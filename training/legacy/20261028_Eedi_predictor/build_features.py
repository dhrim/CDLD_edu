import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();P=Path(__file__).resolve().parent;F=P/f'features/seed{a.seed}'
if (F/'complete.json').exists():raise SystemExit
start=time.time();data=np.load(P/'features/responses.npy',mmap_mode='r');rows=np.load(F/'train.npy');desc=json.loads((P/'description.json').read_text());shape=(desc['people'],desc['items']);u=data['person'][rows];i=data['item'][rows];y=data['correct'][rows].astype(float);uc=np.bincount(u,minlength=shape[0]);ic=np.bincount(i,minlength=shape[1]);rate=np.divide(np.bincount(i,weights=y,minlength=shape[1]),ic,out=np.zeros(shape[1]),where=ic>0);np.save(F/'item_rate.npy',rate)
audit={'seed':a.seed,'training_rate':float(y.mean())}
for method in ['CDLD','responseSVD','exposureSVD']:
 if method=='CDLD':users=np.load(P/f'finders/seed{a.seed}/users.npy').astype(float);items=np.load(P/f'finders/seed{a.seed}/items.npy').astype(float)
 else:
  m=csr_matrix((y-rate[i] if method=='responseSVD' else np.ones(len(rows)),(u,i)),shape=shape)
  U,S,Vt=svds(m,k=64,tol=1e-6,maxiter=10000,v0=np.random.default_rng(20260922).standard_normal(min(shape)),solver='arpack');idx=np.argsort(S)[::-1];U=U[:,idx];S=S[idx];Vt=Vt[idx];users=U*S;items=Vt.T*S
  residual=float(np.linalg.norm(m@Vt.T-users)/np.linalg.norm(S));assert residual<1e-4
  audit[method]={'relative_residual':residual,'student_coordinates':'U Sigma','item_coordinates':'V Sigma'};np.savez(F/f'{method}_singular.npz',singular_values=S);del m,U,Vt
 um=users[uc>0].mean(0);us=users[uc>0].std(0);us[us<1e-12]=1;vm=items[ic>0].mean(0);vs=items[ic>0].std(0);vs[vs<1e-12]=1
 np.save(F/f'{method}_users.npy',((users-um)/us).astype('f'));np.save(F/f'{method}_items.npy',((items-vm)/vs).astype('f'));np.savez(F/f'{method}_scaling.npz',user_mean=um,user_scale=us,item_mean=vm,item_scale=vs)
 print(a.seed,method,'ready',flush=True)
audit['seconds']=time.time()-start;(F/'complete.json').write_text(json.dumps(audit,indent=2))
