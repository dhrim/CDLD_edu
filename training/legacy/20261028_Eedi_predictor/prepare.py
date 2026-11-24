import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;T=P.parent/'20261026_Eedi_57752641';F=P/'features';F.mkdir(exist_ok=True)
source=T/'cache/responses.npy'
if not (F/'responses.npy').exists():(F/'responses.npy').symlink_to(source)
data=np.load(source,mmap_mode='r');desc=json.loads((T/'cache/description.json').read_text());(P/'description.json').write_text(json.dumps(desc,indent=2))
cfg=json.loads((T/'config.json').read_text());(P/'config.json').write_text(json.dumps(cfg,indent=2))
count=np.bincount(data['person'],minlength=desc['people']);eligible=np.flatnonzero(count>=80)
order=np.argsort(data['person'],kind='stable');offset=np.r_[0,count.cumsum()]
for seed in range(42,47):
 out=F/f'seed{seed}';out.mkdir(exist_ok=True);inp=P/f'inputs/seed{seed}';inp.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(seed);people=np.sort(rng.choice(eligible,500,replace=False));q=np.stack([rng.choice(order[offset[u]:offset[u+1]],40,replace=False) for u in people])
 mask=np.random.default_rng(seed).random(len(data),dtype=np.float32)<.1;mask[q.ravel()]=False;valid=np.flatnonzero(mask);mask=~mask;mask[q.ravel()]=False;train=np.flatnonzero(mask)
 assert len(train)+len(valid)+q.size==len(data) and len(np.unique(q))==q.size
 uc=np.bincount(data['person'][train],minlength=desc['people']);ic=np.bincount(data['item'][train],minlength=desc['items']);assert uc[data['person'][q]].min()>0 and ic[data['item'][q]].min()>0
 np.save(out/'train.npy',train);np.save(out/'valid.npy',valid);np.savez(inp/'query.npz',rows=q,people=people)
 audit={'seed':seed,'eligible_students':len(eligible),'train_n':len(train),'validation_n':len(valid),'query_n':q.size,'query_students':len(people),'minimum_train_responses_query_student':int(uc[people].min()),'minimum_train_responses_query_item':int(ic[data['item'][q]].min()),'query_train_validation_overlap':0,'selection':'500 students with at least 80 total responses; 40 random query each; remaining Bernoulli 10% validation; no label/IRT eligibility filters'}
 (inp/'audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
