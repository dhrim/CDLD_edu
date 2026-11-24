from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedGroupKFold
P=Path(__file__).resolve().parent
x=pd.read_csv(P.parent/'20261026_EdNet_유형분류/target_items.csv').sort_values('item')
y=x.part.to_numpy()-1;groups=x.bundle_id.astype(str).to_numpy();outer=np.full(len(x),-1);inner=np.full((5,len(x)),-1)
for o,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=20261026).split(x,y,groups)):
 assert not set(groups[tr])&set(groups[te]);outer[te]=o
 for j,(it,iv) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=20261100+o).split(x.iloc[tr],y[tr],groups[tr])):
  assert not set(groups[tr[it]])&set(groups[tr[iv]]);inner[o,tr[iv]]=j
  assert len(np.unique(y[tr[it]]))==7 and len(np.unique(y[tr[iv]]))==7
np.savez(P/'folds.npz',items=x.item.to_numpy(),y=y,outer=outer,inner=inner,bundles=groups.astype('U'))
x['outer']=outer;x.to_csv(P/'target_items.csv',index=False)
(P/'fold_audit.json').write_text(json.dumps({'items':len(x),'bundles':len(np.unique(groups)),'outer_part_counts':pd.crosstab(outer,y).to_dict(),'outer_inner_bundle_overlap':0},indent=2))
print(pd.crosstab(outer,y))
