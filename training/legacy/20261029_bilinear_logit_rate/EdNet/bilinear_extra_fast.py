import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS']:os.environ[k]='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--method',default='CDLD');p.add_argument('--benchmark',action='store_true');p.add_argument('--selftest',action='store_true');p.add_argument('--reg',type=float,required=True);a=p.parse_args()
for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
tf.keras.utils.set_random_seed(a.seed)
LAM=np.array([a.reg],np.float32);K=len(LAM);D=64
W=tf.Variable(tf.zeros((D,K*D)));A=tf.Variable(tf.zeros((D,K)));B=tf.Variable(tf.zeros((D,K)));C=tf.Variable(tf.zeros(K));G=tf.Variable(tf.zeros(K));variables=[W,A,B,C,G]
def logits(u,v,r):return tf.reduce_sum(tf.reshape(u@W,(-1,K,D))*v[:,None,:],axis=2)+u@A+v@B+C+r[:,None]*G
def penalty():return .5*tf.constant(LAM)*(tf.reduce_sum(tf.reshape(W,(D,K,D))**2,axis=(0,2))+tf.reduce_sum(A*A+B*B,axis=0)+G*G)
if a.selftest:
 rng=np.random.default_rng(123);u=rng.normal(size=(11,D)).astype('f');v=rng.normal(size=(11,D)).astype('f');y=rng.integers(0,2,size=11).astype('f');r=rng.random(11).astype('f')
 for x in variables:x.assign(rng.normal(0,.02,size=x.shape).astype('f'))
 with tf.GradientTape() as tape:
  z=logits(u,v,r);loss=tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.repeat(y[:,None],K,axis=1),logits=z),axis=0)+penalty())
 grads=tape.gradient(loss,variables);wn=W.numpy().reshape(D,K,D);zn=np.einsum('ni,ikj,nj->nk',u,wn,v)+u@A.numpy()+v@B.numpy()+C.numpy()+r[:,None]*G.numpy();pn=1/(1+np.exp(-zn));err=(pn-y[:,None])/len(y)
 assert np.max(abs(grads[4].numpy()-(np.sum(err*r[:,None],axis=0)+LAM*G.numpy())))<2e-6
 expectedW=np.einsum('ni,nk,nj->ikj',u,err,v)+wn*LAM[None,:,None]
 assert np.max(abs(z.numpy()-zn))<2e-6
 assert np.max(abs(grads[0].numpy().reshape(D,K,D)-expectedW))<2e-6
 expectedLoss=np.sum(np.mean(np.logaddexp(0,zn)-y[:,None]*zn,axis=0)+penalty().numpy());assert abs(float(loss)-expectedLoss)<2e-6
 print(json.dumps({'selftest':'passed','logits_max_error':float(np.max(abs(z.numpy()-zn))),'gradient_max_error':float(np.max(abs(grads[0].numpy().reshape(D,K,D)-expectedW)))}));raise SystemExit
P=Path(__file__).resolve().parent;BASE=P;F=BASE/f'features/seed{a.seed}';O=P/f'results/seed{a.seed}/bilinear_{a.method}_lambda{a.reg:g}';O.mkdir(parents=True,exist_ok=True)
if (O/'complete.json').exists():raise SystemExit
start=time.time();users=np.load(F/f'{a.method}_users.npy');items=np.load(F/f'{a.method}_items.npy');U=tf.constant(users);V=tf.constant(items);raw_rates=np.load(F/"item_rate.npy");clipped=np.clip(raw_rates.astype(np.float64),1e-6,1-1e-6);rates=(np.log(clipped)-np.log1p(-clipped)).astype("f");R=tf.constant(rates)
data=np.load(BASE/'features/responses.npy',mmap_mode='r');persons=tf.constant(np.array(data['person']));questions=tf.constant(np.array(data['item']));labels=tf.constant(np.array(data['correct'],dtype=np.float32))
train=np.load(F/'train.npy').astype(np.int32);valid=np.load(F/'valid.npy').astype(np.int32)
mom=[tf.Variable(tf.zeros_like(x)) for x in variables];vel=[tf.Variable(tf.zeros_like(x)) for x in variables]
@tf.function
def step(rows,lr,active,t):
 u=tf.gather(U,tf.gather(persons,rows));v=tf.gather(V,tf.gather(questions,rows));y=tf.gather(labels,rows);r=tf.gather(R,tf.gather(questions,rows))
 with tf.GradientTape() as tape:
  z=logits(u,v,r);bce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.repeat(y[:,None],K,axis=1),logits=z),axis=0);objective=tf.reduce_sum(bce+penalty())
 grads=tape.gradient(objective,variables)
 for index,(x,g,m,s) in enumerate(zip(variables,grads,mom,vel)):
  rate=tf.repeat(lr*active,D)[None,:] if index==0 else (lr*active if index in (3,4) else (lr*active)[None,:])
  m.assign(.9*m+.1*g);s.assign(.999*s+.001*g*g);x.assign_sub(rate*(m/(1-.9**t))/(tf.sqrt(s/(1-.999**t))+1e-7))
 return bce
@tf.function
def evaluate(rows):
 u=tf.gather(U,tf.gather(persons,rows));v=tf.gather(V,tf.gather(questions,rows));y=tf.gather(labels,rows);r=tf.gather(R,tf.gather(questions,rows));z=logits(u,v,r)
 return tf.reduce_sum(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.repeat(y[:,None],K,axis=1),logits=z),tf.float64),axis=0)
@tf.function
def epoch_steps(order,lr,active,stepstart):
 total=tf.zeros(K,dtype=tf.float64)
 for off in tf.range(0,tf.shape(order)[0],65536):
  batch=order[off:off+65536]
  bce=step(batch,lr,active,tf.cast(stepstart+off//65536+1,tf.float32))
  total+=tf.cast(bce,tf.float64)*tf.cast(tf.shape(batch)[0],tf.float64)
 return total
if a.benchmark:
 sample=train[:65536*10];rates=tf.constant(np.full(K,.01,np.float32));flags=tf.constant(np.ones(K,np.float32))
 began=time.time();total=np.zeros(K)
 for i,off in enumerate(range(0,len(sample),65536)):
  batch=sample[off:off+65536];total+=step(batch,rates,flags,tf.constant(float(i+1))).numpy()*len(batch)
 original=[x.numpy().copy() for x in variables];seconds_original=time.time()-began
 for x in variables+mom+vel:x.assign(tf.zeros_like(x))
 began=time.time();compiled=epoch_steps(tf.constant(sample),rates,flags,tf.constant(0)).numpy();seconds_compiled_compile=time.time()-began
 err=max(float(np.max(abs(x.numpy()-v))) for x,v in zip(variables,original));assert err<2e-6,err
 for x in variables+mom+vel:x.assign(tf.zeros_like(x))
 began=time.time();epoch_steps(tf.constant(sample),rates,flags,tf.constant(0)).numpy();seconds_compiled=time.time()-began
 print(json.dumps({'benchmark':'passed','weight_max_error':err,'loss_mean_max_error':float(np.max(abs(compiled-total))/len(sample)),'original_seconds':seconds_original,'compile_seconds':seconds_compiled_compile,'compiled_seconds':seconds_compiled}),flush=True);raise SystemExit
def snapshot(k):return {'W':W.numpy().reshape(D,K,D)[:,k,:].copy(),'a':A.numpy()[:,k].copy(),'b':B.numpy()[:,k].copy(),'c':C.numpy()[k].copy(),'gamma':G.numpy()[k].copy()}
rng=np.random.default_rng(a.seed);order=train.copy();best=np.full(K,np.inf);anchor=best.copy();bestepoch=np.zeros(K,int);stall=np.zeros(K,int);ratecount=np.zeros(K,int);active=np.ones(K,np.float32);lr=np.full(K,.01,np.float32);bestweights=[None]*K;stepnum=0
for epoch in range(1,501):
 if not active.any():break
 epstart=time.time();rng.shuffle(order);tl=np.zeros(K)
 tl=epoch_steps(tf.constant(order),tf.constant(lr),tf.constant(active),tf.constant(stepnum)).numpy();stepnum+=(len(order)+65535)//65536
 vl=np.zeros(K)
 for off in range(0,len(valid),262144):vl+=evaluate(valid[off:off+262144]).numpy()
 vl/=len(valid);tl/=len(order)
 for k in range(K):
  if not active[k]:continue
  if vl[k]<best[k]:best[k]=vl[k];bestepoch[k]=epoch;bestweights[k]=snapshot(k)
  if vl[k]<anchor[k]-1e-5:anchor[k]=vl[k];stall[k]=ratecount[k]=0
  else:
   stall[k]+=1;ratecount[k]+=1
   if ratecount[k]>=3:lr[k]=max(lr[k]*.5,1e-6);ratecount[k]=0
   if stall[k]>=10:active[k]=0
 row={'epoch':epoch,'seconds':time.time()-epstart,'train_logloss':tl.tolist(),'val_logloss':vl.tolist(),'lr':lr.tolist(),'active':active.tolist()}
 with (O/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 (O/'status.json').write_text(json.dumps(row));print(a.seed,a.method,row,flush=True)
assert not active.any(),'Operational limit reached without plateau'
choice=int(np.argmin(best));w=bestweights[choice];np.savez(O/'predictor_weights.npz',**w)
q=np.load(BASE/f'inputs/seed{a.seed}/query.npz');rows=q['rows'];block=data[rows.ravel()];u=users[block['person']].astype(float);v=items[block['item']].astype(float);z=np.sum((u@w['W'])*v,axis=1)+u@w['a']+v@w['b']+w['c']+rates[block['item']].astype(float)*w['gamma'];prob=1/(1+np.exp(-z));np.savez(O/'query_predictions.npz',rows=rows,people=q['people'],y=block['correct'].reshape(rows.shape),prob=prob.reshape(rows.shape))
(O/'complete.json').write_text(json.dumps({'seed':a.seed,'method':a.method,'epochs':sum(1 for _ in (O/'history.jsonl').open()),'lambda_grid':LAM.tolist(),'validation_logloss':best.tolist(),'selected_lambda':float(LAM[choice]),'selected_epoch':int(bestepoch[choice]),'best_epochs':bestepoch.tolist(),'seconds':time.time()-start,'frozen_embeddings_verified':True},indent=2));print('COMPLETE',a.seed,a.method,flush=True)
