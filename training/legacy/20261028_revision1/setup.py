from pathlib import Path
import json,numpy as np,hashlib
P=Path(__file__).resolve().parent
sources={'EdNet':P.parent/'20261027_EdNet_predictor','Eedi':P.parent/'20261028_Eedi_predictor'}
for name,src in sources.items():
 d=P/name
 for key in ['features','inputs']:
  if not (d/key).exists():(d/key).symlink_to(src/key)
 (d/'results').mkdir(exist_ok=True);(d/'claims').mkdir(exist_ok=True)
 (d/'source.json').write_text(json.dumps({'source':str(src)}))
print('linked without modifying source files')
