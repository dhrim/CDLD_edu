"""Assemble local analysis inputs; outputs include provider data and are not release artifacts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reproduce as r
from run import base
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze',action='store_true');a=p.parse_args();w=a.work.resolve();out=a.output.resolve()
if out==Path(__file__).resolve().parents[1]:p.error('Use a separate output directory; do not overwrite published predictions.')
(out/'predictions').mkdir(parents=True,exist_ok=True)
for ds in ['EdNet','Eedi']:
 b=base(w,ds);data=np.load(b/'features/responses.npy',mmap_mode='r');counts={'counts':np.bincount(data['person'])}
 for seed in range(42,47):
  q=np.load(b/f'inputs/seed{seed}/query.npz');rows=q['rows'];y=data['correct'][rows];rate=np.load(b/f'features/seed{seed}/item_rate.npy');d={'people':q['people'],'y':y};counts[f'seed{seed}']=q['people'];sens={**d,'item_rate':rate[data['item'][rows]]}
  for model in ['MLP','bilinear']:
   for m in r.METHODS:
    for variant in ['rate','original']:
     if variant=='rate':directory=w/f'20261028_revision1/{ds}/results/seed{seed}'/(f'MLP_{m}' if model=='MLP' else f'bilinear_final_{m}')
     elif ds=='Eedi':directory=b/f'results/seed{seed}'/(f'MLP_{m}' if model=='MLP' else f'bilinear_final_{m}')
     elif model=='bilinear':directory=w/f'20261027_EdNet_predictor_followup/results/seed{seed}/bilinear_final_{m}'
     elif m=='exposureSVD':directory=w/f'20261027_EdNet_predictor_followup/results/seed{seed}/exposureSVD_extended'
     else:directory=b/f'results/seed{seed}/{m}'
     z=np.load(directory/'query_predictions.npz');assert np.array_equal(z['y'],y) and np.array_equal(z['rows'],rows) and np.array_equal(z['people'],q['people']);d[f'{model}_{m}_{variant}']=z['prob']
  d['item_rate']=rate[data['item'][rows]];np.savez_compressed(out/f'predictions/{ds}_correctness_{seed}.npz',**d)
  for m in r.METHODS:
   z=np.load(w/f'20261029_bilinear_logit_rate/{ds}/results/seed{seed}/bilinear_final_{m}/query_predictions.npz');assert np.array_equal(z['y'],y) and np.array_equal(z['rows'],rows);sens[m+'_logit']=z['prob'];sens[m+'_raw']=d[f'bilinear_{m}_rate']
  np.savez_compressed(out/f'predictions/{ds}_bilinear_sensitivity_{seed}.npz',**sens)
  for mode in (['item','bundle'] if ds=='EdNet' else ['item']):
   directory=w/f'20261028_revision1/classification/{ds}/{mode}/seed{seed}';z=np.load(directory/'oof_B.npz');items=z['items'];cl={'items':items,'y':z['y'],'groups':items}
   if mode=='bundle':
    folds=np.load(w/'20261026_EdNet_bundle/folds.npz');pos=np.searchsorted(folds['items'],items);assert np.array_equal(folds['items'][pos],items);cl['groups']=folds['bundles'][pos]
   for m in r.CLASSIFIERS:
    z=np.load(directory/f'oof_{m}.npz');assert np.array_equal(z['items'],items) and np.array_equal(z['y'],cl['y']);cl[m]=z['prob']
   np.savez_compressed(out/f'predictions/{ds}_{mode}_{seed}.npz',**cl)
 np.savez_compressed(out/f'predictions/{ds}_response_counts.npz',**counts)
if a.analyze:
 r.ROOT=out;results={}
 for ds,mode in [('EdNet','item'),('EdNet','bundle'),('Eedi','item')]:
  results[f'{ds}_{mode}'],domain=r.classification(ds,mode,out)
  if domain:results['domains']=domain
 for ds in ['EdNet','Eedi']:results[f'{ds}_correctness']=r.correctness(ds,out)
 results['samples']={ds:r.activity(ds) for ds in ['EdNet','Eedi']};(out/'results.json').write_text(json.dumps(results,indent=2));(out/'bilinear_sensitivity.json').write_text(json.dumps({ds:r.bilinear_sensitivity(ds,out) for ds in ['EdNet','Eedi']},indent=2))
print('Exported newly fitted predictions to',out)
