import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS']:os.environ[k]='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import original_core as core
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--method',required=True);a=p.parse_args()
P=Path(__file__).resolve().parent;F=P/'features'/f'seed{a.seed}';O=P/'results'/f'seed{a.seed}'/('MLP_'+a.method);O.mkdir(parents=True,exist_ok=True)
if (O/'complete.json').exists():sys.exit(0)
assert (F/'complete.json').exists()
for gpu in core.tf.config.list_physical_devices('GPU'):core.tf.config.experimental.set_memory_growth(gpu,True)
assert core.tf.config.list_physical_devices('GPU')
core.tf.keras.utils.set_random_seed(a.seed)
config={'latent_size':64,'learning_rate':.0005,'batch_size':8192,'predictor_epochs':500}
users=np.load(F/f'{a.method}_users.npy');items=np.load(F/f'{a.method}_items.npy');assert np.isfinite(users).all() and np.isfinite(items).all()
rates=np.load(F/"item_rate.npy").astype(np.float32)
model=core.make_rate_predictor(users,items,config,rates);model.compile(optimizer=core.tf.keras.optimizers.Adam(.0005),loss=core.BatchBinaryCrossentropy())
train=np.load(F/'train.npy');valid=np.load(F/'valid.npy');data=np.load(P/'features/responses.npy',mmap_mode='r')
training=core.ResponseBatches(data,train,8192,True,a.seed);validation=core.ResponseBatches(data,valid,8192,False,a.seed)
start=time.time()
class Progress(core.tf.keras.callbacks.Callback):
 def on_epoch_begin(self,epoch,logs=None):self.started=time.time();self.last=time.time()
 def on_train_batch_end(self,batch,logs=None):
  if time.time()-self.last>40:print(a.seed,a.method,'batch',batch,'loss',logs['loss'],flush=True);self.last=time.time()
 def on_epoch_end(self,epoch,logs=None):
  row={'epoch':epoch+1,'seconds':time.time()-self.started,**{k:float(v) for k,v in logs.items()}}
  with (O/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  (O/'status.json').write_text(json.dumps(row));print(a.seed,a.method,row,flush=True)
stop=core.ReferenceStopping()
h=model.fit(training,validation_data=validation,epochs=500,verbose=0,callbacks=[stop,Progress()])
assert stop.stop_count>=7, 'Operational limit reached before validation plateau'
# Verify embedding arrays were frozen; save only neural Predictor weights.
assert np.array_equal(model.user_latents_layer.get_value(),users) and np.array_equal(model.item_latents_layer.get_value(),items)
assert np.array_equal(model.item_rates_layer.get_value().ravel(),rates)
weights=model.get_layer('Predictor').get_weights();np.savez(O/'predictor_weights.npz',**{f'w{k}':v for k,v in enumerate(weights)})
q=np.load(P/f'inputs/seed{a.seed}/query.npz');rows=q['rows'];block=data[rows.ravel()];inputs=np.column_stack([block['person'],block['item']]).astype(np.float32)
prob=model.predict(inputs,batch_size=8192,verbose=0).ravel();np.savez(O/'query_predictions.npz',rows=rows,people=q['people'],y=block['correct'].reshape(rows.shape),prob=prob.reshape(rows.shape))
(O/'complete.json').write_text(json.dumps({'seed':a.seed,'method':a.method,'epochs':len(h.history['loss']),'best_validation_auc':float(stop.best),'seconds':time.time()-start,'frozen_embeddings_verified':True},indent=2))
print('COMPLETE',a.seed,a.method,time.time()-start,flush=True)
