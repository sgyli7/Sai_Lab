from pathlib import Path
import json,numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_target_gpu_20260912');maximum={k:0. for k in ['obs','action','last_action','ctrl','command']};frames=0
for iteration in [1,2,3]:
 old=json.loads((r/'train_warm'/f'iteration_{iteration:03d}'/'rollouts/completed.json').read_text())['rows'];new=json.loads((r/'train_initial_teacher'/f'iteration_{iteration:03d}'/'rollouts/completed.json').read_text())['rows']
 assert len(old)==len(new)==32
 for a,b in zip(old,new):
  assert json.loads(Path(a['case']).read_text())==json.loads(Path(b['case']).read_text())
  assert a['noise_seed']==b['noise_seed'] and a['explore_after']==b['explore_after']
  x=json.loads(Path(a['trace']).read_text())['rows'];y=json.loads(Path(b['trace']).read_text())['rows'];assert len(x)==len(y)
  for k in maximum:maximum[k]=max(maximum[k],float(np.abs(np.array([z[k] for z in x])-np.array([z[k] for z in y])).max()))
  frames+=len(x)
assert max(maximum.values())==0.,maximum
atomic_json(r/'teacher_pair_audit.json',dict(passed=True,frames=frames,episodes=96,maximum=maximum,scope='Both arms have identical observed policy/control trajectories through the first actor update; teacher target is the declared treatment'))
print(frames,maximum,flush=True)
