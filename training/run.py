"""Portable orchestration around the archived study code. Never downloads data."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
T=Path(__file__).resolve().parent
METHODS=['CDLD','responseSVD','exposureSVD']
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink():
  if dst.resolve()!=src:raise ValueError(f'Conflicting link: {dst}')
 elif dst.exists():
  if dst.resolve()!=src:raise ValueError(f'Existing path: {dst}')
 else:dst.symlink_to(src)
def call(p,*args):
 subprocess.run([sys.executable,str(p),*map(str,args)],cwd=p.parent,check=True)
def module(path,name):
 import importlib.util
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def workspace(work):
 work.mkdir(parents=True,exist_ok=True)
 for p in (T/'legacy').rglob('*'):
  if p.is_file():
   d=work/p.relative_to(T/'legacy');d.parent.mkdir(parents=True,exist_ok=True)
   if d.exists() and d.read_bytes()!=p.read_bytes():raise ValueError(f'Modified source in work directory: {d}')
   if not d.exists():shutil.copy2(p,d)
def base(work,ds):return work/('20261027_EdNet_predictor' if ds=='EdNet' else '20261028_Eedi_predictor')
def prepare(a):
 import numpy as np
 w=a.work;b=base(w,a.dataset);cache=w/f'raw_cache/{a.dataset}';cache.mkdir(parents=True,exist_ok=True)
 if a.dataset=='EdNet':
  c=module(w/'20261028_Eedi_predictor/original_core.py','ednet_core')
  c.RAW_DATA=a.raw.resolve();c.QUESTIONS_FILE=a.questions.resolve();c.CACHE_DIR=cache;c.RUN_MODE='full';c.DATASET_NAME='EdNet'
  _,desc=c.prepare_data()
 else:
  old=w/'20261026_Eedi_57752641';(old/'data').mkdir(exist_ok=True);link(a.raw,old/'data/data.zip');link(cache,old/'cache');call(old/'prepare_eedi.py');desc=json.loads((cache/'description.json').read_text())
  # make_folds.py resolves two levels up from its directory.
  link(cache,w.parent/'자료/Eedi/metadata')
 (b/'features').mkdir(exist_ok=True);link(cache/'responses.npy',b/'features/responses.npy')
 (b/'description.json').write_text(json.dumps(desc))
 shutil.copy2(w/'20261026_Eedi_57752641/config.json',b/'config.json')
 (w/f'{a.dataset}_cache.json').write_text(json.dumps({'cache':str(cache)}))
def split(a):
 import numpy as np
 b=base(a.work,a.dataset)
 if a.dataset=='Eedi':
  # Historical Eedi script creates all five splits with exactly its original RNG order.
  call(b/'prepare.py');return
 core=module(a.work/'20261028_Eedi_predictor/original_core.py','ednet_split')
 data=np.load(b/'features/responses.npy',mmap_mode='r');desc=json.loads((b/'description.json').read_text());cfg=json.loads((b/'config.json').read_text())
 for seed in a.seeds:
  s=core.make_split(data,desc,cfg,seed);f=b/f'features/seed{seed}';i=b/f'inputs/seed{seed}';f.mkdir(parents=True,exist_ok=True);i.mkdir(parents=True,exist_ok=True)
  np.save(f/'train.npy',s['plans']['warm']['train']);np.save(f/'valid.npy',s['plans']['warm']['valid']);np.savez(i/'query.npz',rows=s['query'],people=s['people']);(i/'audit.json').write_text(json.dumps(s['audit']))
def finders(a):
 # The Finder adapter is the original alternating loop with the unrelated
 # initial downstream Predictor omitted. EdNet and Eedi use identical Finder definitions.
 b=base(a.work,a.dataset)
 for n in ['finder_adapter.py','train_finders.py']:
  if a.dataset=='EdNet':shutil.copy2(a.work/'20261028_Eedi_predictor'/n,b/n)
 for seed in a.seeds:call(b/'train_finders.py','--seed',seed)
def ednet_inputs(a):
 import numpy as np,pandas as pd
 from scipy.sparse import csr_matrix
 from scipy.sparse.linalg import svds
 from sklearn.model_selection import StratifiedKFold
 w=a.work;b=base(w,'EdNet');c=module(w/'20261026_Eedi_영역분류/irt_core.py','irt');desc=json.loads((b/'description.json').read_text());cfg=json.loads((b/'config.json').read_text());data=np.load(b/'features/responses.npy',mmap_mode='r')
 old=w/'20261026_EdNet_classification';cache=old/'cache';cache.mkdir(exist_ok=True);shape=(desc['people'],desc['items']);meta=pd.read_csv(w/'raw_cache/EdNet/items.csv');targets=[]
 np.save(cache/'person.npy',data['person']);np.save(cache/'item.npy',data['item']);(cache/'description.json').write_text(json.dumps({'shape':shape,'responses':len(data)}));np.save(b/'inputs/correct.npy',data['correct'].astype('f'))
 for seed in a.seeds:
  f=b/f'features/seed{seed}';i=b/f'inputs/seed{seed}';out=old/f'results/seed{seed}';out.mkdir(parents=True,exist_ok=True);tr=np.load(f/'train.npy');va=np.load(f/'valid.npy');mask=np.zeros(len(data),bool);mask[tr]=True;np.save(cache/f'train_mask{seed}.npy',np.packbits(mask));mask[:]=False;mask[va]=True;np.save(i/'validation_mask.npy',np.packbits(mask))
  count=np.bincount(data['item'][tr],minlength=shape[1]);yes=np.bincount(data['item'][tr],weights=data['correct'][tr],minlength=shape[1]);rate=np.divide(yes,count,out=np.zeros(shape[1]),where=count>0)
  fits={}
  for model in ['Rasch','2PL']:
   fit=c.fit_irt(data,tr,desc,cfg,model,out/f'{model}_checkpoint.npz');assert fit['converged'];fits[model]=fit
  good=(count>0)&(yes>0)&(yes<count)&np.isfinite(np.column_stack([fits['Rasch']['b'],fits['2PL']['b'],np.log(fits['2PL']['a'])])).all(1);ids=np.flatnonzero(good)
  matrix=csr_matrix((data['correct'][tr].astype(float)-rate[data['item'][tr]],(data['person'][tr],data['item'][tr])),shape=shape)
  u,s,vt=svds(matrix,k=64,solver='arpack',tol=1e-6,maxiter=10000,v0=np.random.default_rng(20260922).standard_normal(min(shape)));idx=np.argsort(s)[::-1];u=u[:,idx];s=s[idx];vt=vt[idx]
  assert max((np.linalg.norm(matrix@vt.T-u*s,axis=0)/s).max(),(np.linalg.norm(matrix.T@u-vt.T*s,axis=0)/s).max())<1e-4
  people=np.load(i/'query.npz')['people'];np.savez(i/'response_student_reference.npz',ids=people,coordinates=u[people]*s,singular_values=s)
  for entity,oldname in [('users','user'),('items','item')]:shutil.copy2(b/f'finders/seed{seed}/{entity}.npy',i/f'CDLD_{oldname}_latents.npy')
  np.savez(out/'input.npz',items=ids,y=meta.part.to_numpy()[ids]-1,B=np.column_stack([fits['2PL']['a'],fits['2PL']['b'],rate,np.log1p(count)])[ids],CDLD=np.load(i/'CDLD_item_latents.npy')[ids],responseSVD=(vt.T*s)[ids]);targets.append(ids)
  call(old/'exposure.py','--seed',seed)
 ids=np.unique(np.concatenate(targets));target=meta.iloc[ids].rename(columns={'original_id':'original_id'});y=target.part.to_numpy()-1;outer=np.full(len(ids),-1);inner=np.full((5,len(ids)),-1)
 for o,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261026).split(y,y)):
  outer[te]=o
  for j,(_,iv) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261100+o).split(y[tr],y[tr])):inner[o,tr[iv]]=j
 np.savez(old/'folds.npz',items=ids,y=y,outer=outer,inner=inner);target[['item','original_id','part','bundle_id']].to_csv(old/'target_items.csv',index=False)
 link(old,w/'20261026_EdNet_유형분류');call(w/'20261026_EdNet_bundle/prepare.py')
def features(a):
 b=base(a.work,a.dataset)
 if a.dataset=='EdNet':
  if a.seeds!=list(range(42,47)):raise ValueError('EdNet features requires seeds 42–46 together to construct the common item folds.')
  ednet_inputs(a)
 for seed in a.seeds:call(b/'build_features.py','--seed',seed)
 if a.dataset=='Eedi':call(a.work/'20261026_Eedi_영역분류/make_folds.py')
 link_inputs(a)
def link_inputs(a):
 import numpy as np
 w=a.work;b=base(w,a.dataset)
 for name in ['20261028_revision1','20261029_bilinear_logit_rate']:
  d=w/name/a.dataset
  for key in ['features','inputs']:link(b/key,d/key)
  for key in ['results','claims']:(d/key).mkdir(exist_ok=True)
 data=np.load(b/'features/responses.npy',mmap_mode='r');n=json.loads((b/'description.json').read_text())['items']
 for seed in a.seeds:
  tr=np.load(b/f'features/seed{seed}/train.npy');np.save(w/f'20261028_revision1/{a.dataset}/results/counts_seed{seed}.npy',np.bincount(data['item'][tr],minlength=n))
def predictors(a):
 w=a.work;link_inputs(a)
 for seed in a.seeds:
  if a.variant=='raw-rate':d=w/'20261028_revision1'/a.dataset
  elif a.variant=='logit-rate':d=w/'20261029_bilinear_logit_rate'/a.dataset
  else:d=base(w,a.dataset)
  for method in METHODS:
   if a.variant!='logit-rate':
    if a.variant=='no-rate' and a.dataset=='EdNet':
     call(d/'train.py','--seed',seed,'--method',method)
     if method=='exposureSVD':call(w/'20261027_EdNet_predictor_followup/extend.py','--seed',seed,'--method',method)
    else:call(d/'train_mlp.py','--seed',seed,'--method',method)
   bd=w/'20261027_EdNet_predictor_followup' if a.variant=='no-rate' and a.dataset=='EdNet' else d
   (bd/'claims').mkdir(exist_ok=True)
   call(bd/'bilinear_fast.py','--seed',seed,'--method',method)
  call(bd/'select_extended_grid.py',a.gpu,seed)
def classify(a):
 for seed in a.seeds:
  # Historical rerun raised max_iter only for EdNet item seeds 42/46.
  name='classification_convergence.py' if a.dataset=='EdNet' and a.mode=='item' and seed in [42,46] else 'classification.py'
  call(a.work/'20261028_revision1'/name,'--dataset',a.dataset,'--seed',seed,'--mode',a.mode)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','split','finders','features','predictors','classify']);p.add_argument('--dataset',choices=['EdNet','Eedi'],required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--raw',type=Path);p.add_argument('--questions',type=Path);p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)));p.add_argument('--variant',choices=['raw-rate','no-rate','logit-rate'],default='raw-rate');p.add_argument('--mode',choices=['item','bundle'],default='item');p.add_argument('--gpu',default='0');a=p.parse_args();a.work=a.work.resolve()
 if a.stage=='prepare' and (a.raw is None or (a.dataset=='EdNet' and a.questions is None)):p.error('prepare requires --raw; EdNet also requires --questions')
 if a.dataset=='Eedi' and a.mode=='bundle':p.error('bundle mode is EdNet only')
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;workspace(a.work);globals()[a.stage](a)
if __name__=='__main__':main()
