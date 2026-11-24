"""Prepare official Eedi task1/2 training responses. No test labels or text/images used."""
from pathlib import Path
import zipfile,json,hashlib,ast
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;D=P/'data';C=P/'cache';C.mkdir(exist_ok=True)
zpath=D/'data.zip'
with zipfile.ZipFile(zpath) as z:
 with z.open('data/train_data/train_task_1_2.csv') as f:
  df=pd.read_csv(f,usecols=['QuestionId','UserId','AnswerId','IsCorrect'],dtype={'QuestionId':'int32','UserId':'int32','AnswerId':'int64','IsCorrect':'uint8'})
 with z.open('data/metadata/question_metadata_task_1_2.csv') as f:q=pd.read_csv(f)
 with z.open('data/metadata/subject_metadata.csv') as f:s=pd.read_csv(f)
assert df.IsCorrect.isin([0,1]).all()
assert not df.duplicated(['UserId','QuestionId']).any(),'Unexpected repeated pairs; do not choose an undocumented rule'
people=np.sort(df.UserId.unique());items=np.sort(df.QuestionId.unique())
q=q.set_index('QuestionId').loc[items].reset_index();assert len(q)==len(items)
df=df.sort_values(['UserId','QuestionId']).reset_index(drop=True)
a=np.lib.format.open_memmap(C/'responses.npy',mode='w+',dtype=[('person','<i4'),('item','<i4'),('correct','u1'),('timestamp','<i8')],shape=(len(df),))
a['person']=np.searchsorted(people,df.UserId);a['item']=np.searchsorted(items,df.QuestionId);a['correct']=df.IsCorrect;a['timestamp']=-1;a.flush()
pd.DataFrame({'person':np.arange(len(people)),'original_id':people}).to_csv(C/'people.csv',index=False)
q.insert(0,'item',np.arange(len(q)));q.to_csv(C/'items.csv',index=False);s.to_csv(C/'subjects.csv',index=False)
levels=dict(zip(s.SubjectId,s.Level));parents=dict(zip(s.SubjectId,s.ParentId));names=dict(zip(s.SubjectId,s.Name))
def project(raw):
 out=set()
 for k in ast.literal_eval(raw):
  seen=set()
  while k in levels and levels[k]>1 and k not in seen:seen.add(k);k=parents[k]
  if k in levels and levels[k]==1:out.add(int(k))
 return sorted(out)
labels=q.SubjectId.map(project)
labeldf=q[['item','QuestionId']].copy();labeldf['level1_ids']=labels.map(json.dumps);labeldf['label_count']=labels.map(len);labeldf.to_csv(C/'label_audit.csv',index=False)
counts=pd.Series([k for x in labels for k in x]).value_counts()
pd.DataFrame([{'subject_id':int(k),'name':names[k],'n_items_including_multilabel':int(v)} for k,v in counts.items()]).to_csv(C/'domain_counts.csv',index=False)
record={'dataset':'Eedi','source':'official train_task_1_2.csv only','people':len(people),'items':len(items),'responses':len(df),'duplicates':0,'correct_mean':float(df.IsCorrect.mean()),'label_multiplicity':{str(k):int(v) for k,v in labels.map(len).value_counts().items()},'source_zip_sha256':hashlib.file_digest(open(zpath,'rb'),'sha256').hexdigest(),'item_count_min':int(df.groupby('QuestionId').size().min()),'person_count_min':int(df.groupby('UserId').size().min())}
(C/'description.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
