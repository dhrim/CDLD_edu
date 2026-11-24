import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
P=Path(__file__).resolve().parent;O=P/'results'/f'seed{a.seed}';C=P/'cache'
if not (O/'exposure_svd.npz').exists():
 start=time.time();desc=json.loads((C/'description.json').read_text());shape=tuple(desc['shape'])
 mask=np.unpackbits(np.load(C/f'train_mask{a.seed}.npy'),count=desc['responses']).astype(bool)
 people=np.load(C/'person.npy',mmap_mode='r')[mask];items=np.load(C/'item.npy',mmap_mode='r')[mask]
 matrix=csr_matrix((np.ones(len(items)),(people,items)),shape=shape);assert matrix.data.max()==1
 print('matrix',a.seed,shape,matrix.nnz,flush=True)
 U,S,Vt=svds(matrix,k=64,solver='arpack',tol=1e-6,maxiter=10000,v0=np.random.default_rng(20260922).standard_normal(min(shape)))
 order=np.argsort(S)[::-1];S=S[order];U=U[:,order];Vt=Vt[order]
 left=np.linalg.norm(matrix@Vt.T-U*S,axis=0)/S;right=np.linalg.norm(matrix.T@U-Vt.T*S,axis=0)/S
 assert max(left.max(),right.max())<1e-4
 ids=np.load(O/'input.npz')['items'];np.savez(O/'exposure_svd.npz',items=ids,coordinates=(Vt.T*S)[ids],singular_values=S)
 (O/'exposure_diagnostics.json').write_text(json.dumps({'seed':a.seed,'seconds':time.time()-start,'shape':shape,'observed':matrix.nnz,'left_residual':left.tolist(),'right_residual':right.tolist(),'coordinates':'V Sigma'},indent=2))
 print('DONE',a.seed,time.time()-start,flush=True)
