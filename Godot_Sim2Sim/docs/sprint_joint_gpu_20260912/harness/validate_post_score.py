from pathlib import Path
import json,numbers
import numpy as np
from sim2sim.standalone.replay import deployment,raw_state
from sim2sim.research.queue import atomic_json
from post_score import score_samples
r=Path('results/sprint_joint_gpu_20260912');suite=r/'native_tracking/suite';episodes=json.loads((suite/'summary.json').read_text())['episodes'];robot=deployment(suite/'runtime')['robots']['walk'];records=[]
def difference(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b);return max((difference(a[k],b[k]) for k in a),default=0.)
 if isinstance(a,list):
  assert len(a)==len(b);return max((difference(x,y) for x,y in zip(a,b)),default=0.)
 if isinstance(a,bool) or a is None or isinstance(a,str):assert a==b;return 0.
 return abs(float(a)-float(b))
selected=[e for e in episodes if e['seed']==927001 and not e['case'].endswith('_ordinary')]
assert len(selected)==8
for e in selected:
 trace=json.loads(Path(e['trace']).read_text());rows=trace['rows'];states=[]
 for row in rows:
  b=row['body'];states.append([*b['base_pos'],*b['base_quat'],*row['raw']['base_angvel_local'],*b['base_linvel']])
 final=raw_state(trace['summary']['final_raw'],robot);states.append([*final.base_pos,*final.base_quat_wxyz,*final.base_angvel_local,*final.base_linvel])
 case=json.loads(Path(e['case_path']).read_text());commands=[x['requested_command'] for x in rows];selected=[x['skill']=='sprint' for x in rows]
 actual=score_samples(states,case,commands,selected);error=difference(actual,e['task_metrics']);assert error<1e-5,(e['case'],error)
 try:score_samples(states[:-1],case,commands,selected)
 except ValueError:pass
 else:raise AssertionError('Missing final state was silently scored')
 # Previous helper scored pre-action states; preserve its discrepancy as evidence.
 wrong=score_samples([states[0],*states[:-1]],case,commands,selected)
 numeric_delta=abs(wrong['sustained_mean_vx']-actual['sustained_mean_vx'])
 records.append(dict(case=e['case'],seed=e['seed'],max_error=error,old_pre_step_speed_error=numeric_delta,old_pre_step_success=wrong['success'],native_success=actual['success']))
atomic_json(r/'post_score_validation.json',dict(passed=True,cases=records,sampling='Physical state after each same action, including final state; native acceptance unchanged'))
print(records,flush=True)
