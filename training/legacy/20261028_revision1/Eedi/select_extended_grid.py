from pathlib import Path
import json,subprocess,sys,os,shutil
import numpy as np
P=Path(__file__).resolve().parent;gpu=int(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
for seed in [int(sys.argv[2])]:
 for method in ['CDLD','responseSVD','exposureSVD']:
  base=P/f'results/seed{seed}/bilinear_{method}';target=P/f'results/seed{seed}/bilinear_final_{method}'
  if (target/'complete.json').exists() or not (base/'complete.json').exists():continue
  claim=P/f'claims/selection_{seed}_{method}'
  try:claim.mkdir()
  except FileExistsError:continue
  d=json.loads((base/'complete.json').read_text());candidates=[{'lambda':l,'validation_logloss':v,'source':str(base.relative_to(P))} for l,v in zip(d['lambda_grid'],d['validation_logloss'])];idx=int(np.argmin(d['validation_logloss']));best=float(d['validation_logloss'][idx]);current=float(d['lambda_grid'][idx]);chosen=base
  boundary=idx==len(d['lambda_grid'])-1
  while boundary and current<1e4:
   nxt=float(f'{current*10:.6g}');extra=P/f'results/seed{seed}/bilinear_{method}_lambda{nxt:g}'
   if not (extra/'complete.json').exists():
    with (P/f'extra_{seed}_{method}_{nxt:g}.log').open('a') as log:subprocess.run([sys.executable,'-u',str(P/'bilinear_extra_fast.py'),'--seed',str(seed),'--method',method,'--reg',str(nxt)],stdout=log,stderr=subprocess.STDOUT,check=True)
   e=json.loads((extra/'complete.json').read_text());value=float(e['validation_logloss'][0]);candidates.append({'lambda':nxt,'validation_logloss':value,'source':str(extra.relative_to(P))})
   if value<best:best=value;current=nxt;chosen=extra
   else:boundary=False
  assert not boundary, 'Regularization grid still at upper boundary'
  target.mkdir(parents=True,exist_ok=True)
  for name in ['predictor_weights.npz','query_predictions.npz']:shutil.copy2(chosen/name,target/name)
  done=json.loads((chosen/'complete.json').read_text());done.update({'selected_lambda':current,'selected_validation_logloss':best,'selected_source':str(chosen.relative_to(P)),'selection_candidates':candidates,'upper_boundary_remaining':boundary})
  (target/'complete.json').write_text(json.dumps(done,indent=2));print(seed,method,'SELECTED',current,best,'boundary',boundary,flush=True)
