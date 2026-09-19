from pathlib import Path
import json,math,numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_target_gpu_20260912')
rows=[]
for completed in sorted((r/'train').glob('iteration_*/rollouts/completed.json')):
 for e in json.loads(completed.read_text())['rows']:
  case=json.loads(Path(e['case']).read_text())
  if case['case']!='sprint_alternate':continue
  trace=json.loads(Path(e['trace']).read_text());actions=trace['rows'];times=np.array([a['t'] for a in actions]);q=np.array([a['body']['base_quat'] for a in actions]);angles=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
  selected=np.flatnonzero((times>=7.-1e-9)&(times<9.-1e-9))
  command=float(np.mean([a['requested_command'][2] for a in actions if 7.-1e-9<=a['t']<9.-1e-9]));rate=float((angles[selected[-1]]-angles[selected[0]])/(times[selected[-1]]-times[selected[0]])*np.sign(command))
  rows.append(dict(iteration=completed.parents[1].name,seed=case['seed'],signed_yaw_rate=rate,requested=abs(command),below_turn_threshold=rate<.6*abs(command),trace=e['trace']))
result=dict(scope='Diagnostic coverage of stochastic training trajectories only; never acceptance evidence',rows=rows,count=len(rows),below_threshold=sum(x['below_turn_threshold'] for x in rows),minimum=min(x['signed_yaw_rate'] for x in rows),mean=float(np.mean([x['signed_yaw_rate'] for x in rows])))
atomic_json(r/'coverage_audit.json',result);print({k:v for k,v in result.items() if k!='rows'})
