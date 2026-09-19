from pathlib import Path
import json
from sim2sim.research.finalize import verify_checkpoint
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.replay import shadow
r=Path('results/sprint_exit_20260912');records={}
for name in ['e04_motion_visible','e05_motion_masked']:
 p=r/'runs'/name;suite=r/(name+'_native')/'suite';data=json.loads((suite/'summary.json').read_text())
 trace=next(x['trace'] for x in data['episodes'] if x['case']=='remove_forward_prefix')
 parity=verify_checkpoint(p/'latest.pt',p/'final.onnx',n=256)
 comparison=shadow(trace,suite/'runtime')
 assert parity['passed'] and comparison['passed']
 records[name]=dict(checkpoint=parity,native_shadow=comparison)
atomic_json(r/'candidate_numerical_verification.json',records);print(records,flush=True)
