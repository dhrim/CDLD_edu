import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.sparse import csr_matrix
P=Path(__file__).resolve().parent;S=P.parent/'20261026_EdNet_classification';C=S/'cache';I=P/'inputs';F=P/'features';F.mkdir(exist_ok=True)
desc=json.loads((C/'description.json').read_text());shape=tuple(desc['shape']);people=np.load(C/'person.npy',mmap_mode='r');items=np.load(C/'item.npy',mmap_mode='r');correct=np.load(I/'correct.npy',mmap_mode='r')
# Compact read-only response cache shared by all Predictor jobs.
path=F/'responses.npy'
if not path.exists():
 data=np.lib.format.open_memmap(path,mode='w+',dtype=[('person','i4'),('item','i4'),('correct','f4')],shape=people.shape)
 for start in range(0,len(people),1000000):
  sl=slice(start,start+1000000);data['person'][sl]=people[sl];data['item'][sl]=items[sl];data['correct'][sl]=correct[sl]
 data.flush();del data
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);args=parser.parse_args()
for seed in ([args.seed] if args.seed else range(42,47)):
 out=F/f'seed{seed}';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():continue
 t=time.time();mask=np.unpackbits(np.load(C/f'train_mask{seed}.npy'),count=len(people)).astype(bool);rows=np.flatnonzero(mask)
 valid=np.flatnonzero(np.unpackbits(np.load(I/f'seed{seed}/validation_mask.npy'),count=len(people)));query=np.load(I/f'seed{seed}/query.npz')['rows'];assert not mask[valid].any() and not mask[query].any();assert not np.isin(query,valid).any()
 np.save(out/'train.npy',rows);np.save(out/'valid.npy',valid)
 u=people[rows];i=items[rows];y=correct[rows].astype(float);uc=np.bincount(u,minlength=shape[0]);ic=np.bincount(i,minlength=shape[1]);rate=np.divide(np.bincount(i,weights=y,minlength=shape[1]),ic,out=np.zeros(shape[1]),where=ic>0)
 np.save(out/'item_rate.npy',rate)
 source=np.load(S/f'results/seed{seed}/input.npz');ids=source['items'];assert np.isin(np.flatnonzero(ic),ids).all()
 diagnostics={}
 for method in ['CDLD','responseSVD','exposureSVD']:
  if method=='CDLD':users=np.load(I/f'seed{seed}/CDLD_user_latents.npy').astype(float);v=np.load(I/f'seed{seed}/CDLD_item_latents.npy').astype(float)
  else:
   if method=='responseSVD':
    coords=source['responseSVD'];sing=np.load(I/f'seed{seed}/response_student_reference.npz')['singular_values'];values=y-rate[i]
   else:
    e=np.load(S/f'results/seed{seed}/exposure_svd.npz');assert np.array_equal(e['items'],ids);coords=e['coordinates'];sing=e['singular_values'];values=np.ones(len(i))
   v=np.zeros((shape[1],64));v[ids]=coords;matrix=csr_matrix((values,(u,i)),shape=shape)
   users=matrix@(v/sing)
   if method=='responseSVD':
    ref=np.load(I/f'seed{seed}/response_student_reference.npz');error=np.linalg.norm(users[ref['ids']]-ref['coordinates'])/np.linalg.norm(ref['coordinates']);assert error<1e-4;diagnostics['student_reconstruction_relative_error']=float(error)
   del matrix
  # Identical, invertible coordinate scaling for all three methods; fit on train-active entities only.
  um=users[uc>0].mean(0);us=users[uc>0].std(0);us[us<1e-12]=1;vm=v[ic>0].mean(0);vs=v[ic>0].std(0);vs[vs<1e-12]=1
  np.save(out/f'{method}_users.npy',((users-um)/us).astype(np.float32));np.save(out/f'{method}_items.npy',((v-vm)/vs).astype(np.float32));np.savez(out/f'{method}_scaling.npz',user_mean=um,user_scale=us,item_mean=vm,item_scale=vs)
  print(seed,method,'ready',flush=True)
  del users,v
 diagnostics.update({'seed':seed,'seconds':time.time()-t,'train':len(rows),'validation':len(valid),'query':query.size,'query_overlap':0,'training_rate':float(y.mean())})
 (out/'complete.json').write_text(json.dumps(diagnostics,indent=2));print(diagnostics,flush=True)
