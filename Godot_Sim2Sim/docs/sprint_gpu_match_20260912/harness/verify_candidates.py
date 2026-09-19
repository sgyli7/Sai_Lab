from pathlib import Path
import json
from sim2sim.research.finalize import verify_checkpoint
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.replay import shadow
r=Path('results/sprint_gpu_match_20260912');records={}
for arm,training,native in [('source','train_source_graph','native_source_graph'),('jolt','train_jolt_graph','native_jolt_graph'),('positive','train_positive','native_positive'),('cat','train_cat','native_cat')]:
 p=r/training;suite=r/native/'suite';data=json.loads((suite/'summary.json').read_text())
 parity=verify_checkpoint(p/'latest.pt',p/'final.onnx',n=256)
 comparisons={}
 for name in ['remove_forward_prefix','sprint_repeat']:
  trace=next(x['trace'] for x in data['episodes'] if x['case']==name)
  comparisons[name]=shadow(trace,suite/'runtime')
 assert parity['passed'] and all(x['passed'] for x in comparisons.values())
 records[arm]=dict(checkpoint=parity,native_shadow=comparisons)
 atomic_json(r/'candidate_numerical_verification.json',records)
print(records,flush=True)
