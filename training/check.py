"""CPU synthetic-data checks; does not fit the study models or use participant data."""
import ast,hashlib,json,tempfile,zipfile,sys,subprocess,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
T=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',T/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks={}
for row in json.loads((T/'source_manifest.json').read_text()):assert hashlib.sha256((T/row['file']).read_bytes()).hexdigest()==row['sha256']
for p in T.rglob('*.py'):ast.parse(p.read_text())
checks['source_hashes_and_syntax']='passed'
with tempfile.TemporaryDirectory() as tmp:
 w=Path(tmp)/'work';r.workspace(w)
 # Execute the exact historical split function without importing TensorFlow.
 text=(T/'legacy/20261028_Eedi_predictor/original_core.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='make_split');scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original make_split>','exec'),scope)
 n=600;k=100;data=np.zeros(n*k,dtype=[('person','i4'),('item','i4'),('correct','u1')]);data['person']=np.repeat(np.arange(n),k);data['item']=np.tile(np.arange(k),n);data['correct']=np.random.default_rng(1).integers(0,2,n*k)
 cfg=json.loads((w/'20261028_Eedi_predictor/config.json').read_text());s=scope['make_split'](data,{'people':n,'items':k},cfg,42);tr=s['plans']['warm']['train'];va=s['plans']['warm']['valid'];q=s['query'].ravel();assert len(q)==20000;assert not np.intersect1d(tr,q).size and not np.intersect1d(va,q).size;checks['EdNet_original_split']='passed'
 # Tiny official-layout Eedi archive, enough for the original 500 x 40 query rule.
 frame=pd.DataFrame({'UserId':data['person'],'QuestionId':data['item'],'AnswerId':np.arange(len(data)),'IsCorrect':data['correct']});qmeta=pd.DataFrame({'QuestionId':np.arange(k),'SubjectId':[str([ [32,49,71,101][i%4] ]) for i in range(k)]});subjects=pd.DataFrame({'SubjectId':[32,49,71,101],'ParentId':[0]*4,'Level':[1]*4,'Name':['Number','Algebra','Geometry','Data']})
 raw=Path(tmp)/'data.zip'
 with zipfile.ZipFile(raw,'w') as z:
  for name,df in [('train_data/train_task_1_2.csv',frame),('metadata/question_metadata_task_1_2.csv',qmeta),('metadata/subject_metadata.csv',subjects)]:z.writestr('data/'+name,df.to_csv(index=False))
 from types import SimpleNamespace
 a=SimpleNamespace(work=w,dataset='Eedi',raw=raw,questions=None,seeds=[42]);r.prepare(a);r.split(a)
 b=r.base(w,'Eedi');saved=np.load(b/'inputs/seed42/query.npz');assert saved['rows'].shape==(500,40)
 f=b/'finders/seed42';f.mkdir(parents=True);rng=np.random.default_rng(3);np.save(f/'users.npy',rng.normal(size=(n,64)));np.save(f/'items.npy',rng.normal(size=(k,64)))
 r.features(a);features=b/'features/seed42'
 for method in r.METHODS:
  assert np.load(features/f'{method}_items.npy').shape==(k,64);assert np.load(features/f'{method}_users.npy').shape==(n,64)
 for name in ['20261028_revision1','20261029_bilinear_logit_rate']:
  assert (w/name/'Eedi/features/responses.npy').is_file();assert (w/name/'Eedi/inputs/seed42/query.npz').is_file()
 folds=np.load(w/'20261026_Eedi_영역분류/folds.npz');assert np.all(folds['outer']>=0)
 checks['Eedi_preprocessing_split_SVD_scaling_folds_links']='passed'
 checks['synthetic_Finder_vectors']='random fixtures, not trained models'
checks['full_GPU_retraining']='not run';print(json.dumps(checks,indent=2))
