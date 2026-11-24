from pathlib import Path
import json,ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
P=Path(__file__).resolve().parent
M=P.parents[1]/'자료/Eedi/metadata'
d=pd.read_csv(M/'label_audit.csv');d['ids']=d.level1_ids.map(ast.literal_eval);d['domain']=d.ids.map(lambda x:x[0] if len(x)==1 else -1)
c=d[d.label_count==1].domain.value_counts();keep={32,49,71,101} # Four main domains, explicitly selected by user before classification
target=d[d.domain.isin(keep)].copy().sort_values('item');classes=sorted(keep);target['y']=target.domain.map({k:i for i,k in enumerate(classes)})
y=target.y.to_numpy();outer=np.full(len(y),-1);inner=np.full((5,len(y)),-1)
for o,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261026).split(y,y)):
 outer[te]=o
 for j,(_,iv) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261100+o).split(y[tr],y[tr])):inner[o,tr[iv]]=j
np.savez(P/'folds.npz',items=target.item.to_numpy(),question_ids=target.QuestionId.to_numpy(),y=y,outer=outer,inner=inner)
names=pd.read_csv(M/'subjects.csv').set_index('SubjectId').Name.to_dict()
(P/'labels.json').write_text(json.dumps({'classes':[{'y':i,'subject_id':int(k),'name':names[k],'n':int((target.domain==k).sum())} for i,k in enumerate(classes)],'included_n':len(target),'excluded_n':len(d)-len(target),'excluded_multilabel':int((d.label_count!=1).sum())},indent=2))
target[['item','QuestionId','domain','y']].to_csv(P/'target_items.csv',index=False)
d[~d.index.isin(target.index)][['item','QuestionId','level1_ids','label_count']].to_csv(P/'excluded_items.csv',index=False)
print((P/'labels.json').read_text())
