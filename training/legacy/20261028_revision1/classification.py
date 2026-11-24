import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time,warnings,sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.metrics import log_loss,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error',category=ConvergenceWarning)
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',default='item');a=p.parse_args()
P=Path(__file__).resolve().parent;F=P/a.dataset/f'features/seed{a.seed}';O=P/'classification'/a.dataset/a.mode/f'seed{a.seed}';O.mkdir(parents=True,exist_ok=True)
I=P/'classification'/a.dataset/'inputs'/f'seed{a.seed}';I.mkdir(parents=True,exist_ok=True)
def save(path,x):path.write_text(json.dumps(x,indent=2))
if a.dataset=='EdNet':
 old=P.parent/'20261026_EdNet_classification';z=np.load(old/f'results/seed{a.seed}/input.npz');items=z['items'];y=z['y'];base=z['B'];assert np.allclose(base[:,2],np.load(F/'item_rate.npy')[items])
 folds=np.load((old if a.mode=='item' else P.parent/'20261026_EdNet_bundle')/'folds.npz')
else:
 old=P.parent/'20261026_Eedi_영역분류';sys.path.insert(0,str(old));import irt_core as core
 data=np.load(P/'Eedi/features/responses.npy',mmap_mode='r');rows=np.load(F/'train.npy');count=np.load(P/f'Eedi/results/counts_seed{a.seed}.npy');rate=np.load(F/'item_rate.npy');eligible=(rate>0)&(rate<1)&(count>0)
 if not (I/'irt.npz').exists():
  source=P.parent/'20261028_Eedi_predictor';desc=json.loads((source/'description.json').read_text());cfg=json.loads((source/'config.json').read_text());start=time.time()
  r=core.fit_irt(data,rows[eligible[data['item'][rows]]],desc,cfg,'2PL',I/'irt_checkpoint.npz');assert r['converged'];np.savez(I/'irt.npz',a=r['a'],b=r['b']);save(I/'irt_diagnostics.json',{**{k:v for k,v in r.items() if k not in ['a','b','active','history']},'seconds':time.time()-start})
 folds=np.load(old/'folds.npz');items=folds['items'][eligible[folds['items']]];pos=np.searchsorted(folds['items'],items);y=folds['y'][pos];irt=np.load(I/'irt.npz');base=np.column_stack([irt['a'],irt['b'],rate,np.log1p(count)])[items]
pos=np.searchsorted(folds['items'],items);assert np.array_equal(folds['items'][pos],items) and np.array_equal(folds['y'][pos],y)
outer=folds['outer'][pos];inner=folds['inner'][:,pos];classes=np.arange(len(np.unique(y)));features={'frequency':None,'B':base}
for method in ['CDLD','responseSVD','exposureSVD']:
 v=np.load(F/f'{method}_items.npy')[items];features[method]=v;features['B_'+method]=np.column_stack([base,v])
np.savez_compressed(O/'design.npz',items=items,y=y,outer=outer,inner=inner,B=base)
for name,X in features.items():
 if (O/f'metrics_{name}.json').exists():continue
 started=time.time();prob=np.zeros((len(y),len(classes)));choices=[]
 for o in range(5):
  tr=np.flatnonzero(outer!=o);te=np.flatnonzero(outer==o)
  assert len(np.unique(y[tr]))==len(classes)
  if X is None:prob[te]=np.bincount(y[tr],minlength=len(classes))/len(tr);continue
  for j in range(5):assert len(np.unique(y[tr[inner[o,tr]!=j]]))==len(classes)
  model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(solver='lbfgs',max_iter=20000 if a.mode=='bundle' else 3000,tol=1e-6))
  search=GridSearchCV(model,{'logisticregression__C':np.logspace(-4,4,9)},cv=PredefinedSplit(inner[o,tr]),scoring='neg_log_loss',n_jobs=4,error_score='raise',refit=True)
  search.fit(X[tr],y[tr]);prob[te]=search.predict_proba(X[te]);choices.append({'outer':o,'C':search.best_params_['logisticregression__C'],'inner_loss':-search.best_score_})
  save(O/'status.json',{'method':name,'outer_complete':o+1,'seconds':time.time()-started});print(a.dataset,a.seed,a.mode,name,o+1,flush=True)
 assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
 np.savez_compressed(O/f'oof_{name}.npz',items=items,y=y,prob=prob,outer=outer)
 pred=prob.argmax(1);save(O/f'metrics_{name}.json',{'method':name,'n':len(y),'logloss':log_loss(y,prob,labels=classes),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'selected_C':choices,'seconds':time.time()-started,'confusion':confusion_matrix(y,pred,labels=classes).tolist(),'per_class':classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)})
save(O/'complete.json',{'dataset':a.dataset,'seed':a.seed,'mode':a.mode,'n':len(y),'conditions':list(features)})
