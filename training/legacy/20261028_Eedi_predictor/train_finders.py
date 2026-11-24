import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS']:os.environ[k]='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import original_core as core
import finder_adapter as adapter
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
P=Path(__file__).resolve().parent;F=P/f'features/seed{a.seed}';O=P/f'finders/seed{a.seed}';O.mkdir(parents=True,exist_ok=True)
if (O/'complete.json').exists():sys.exit(0)
for gpu in core.tf.config.list_physical_devices('GPU'):core.tf.config.experimental.set_memory_growth(gpu,True)
assert core.tf.config.list_physical_devices('GPU');core.CHECKPOINT_DIRECTORY=O
core.run_reproduction_gates();start=time.time();Base=core.TrainingProgress
class Progress(Base):
 def on_epoch_begin(self,epoch,logs=None):self.began=time.time()
 def on_epoch_end(self,epoch,logs=None):
  super().on_epoch_end(epoch,logs)
  row={'phase':self.label,'epoch':epoch+1,'seconds':time.time()-self.began,'elapsed':time.time()-start,**{k:float(v) for k,v in logs.items()}}
  with (O/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  (O/'status.json').write_text(json.dumps(row))
adapter.TrainingProgress=Progress
fit=adapter.fit_finders(np.load(P/'features/responses.npy',mmap_mode='r'),np.load(F/'train.npy'),np.load(F/'valid.npy'),json.loads((P/'description.json').read_text()),json.loads((P/'config.json').read_text()),a.seed)
np.save(O/'users.npy',fit['final_users']);np.save(O/'items.npy',fit['final_items']);(O/'complete.json').write_text(json.dumps({'seed':a.seed,'seconds':time.time()-start,'finder_epochs':100},indent=2))
