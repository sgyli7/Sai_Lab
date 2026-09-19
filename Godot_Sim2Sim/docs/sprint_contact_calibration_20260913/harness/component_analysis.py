"""Post-experiment localization; descriptive only, never a new selection gate."""
from pathlib import Path
import json
import numpy as np
from sim2sim.paths import load_robot_json
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_contact_calibration_20260913')
source=json.loads((r/'manifold_probe.json').read_text())['sources']
rows=[json.loads(Path(e['trace']).read_text())['rows'][:101] for e in source]
native=np.array([[x[i]['raw']['q'] for x in rows] for i in range(101)])
cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
import mujoco
m=mujoco.MjModel.from_xml_path(cfg['mjcf'])
names=[m.joint(int(m.actuator_trnid[i,0])).name for i in range(14)]
time=np.arange(101)*.02
out={}
for name in ['rigid','direct20k_hard_contact','direct30k_hard_contact']:
 d=np.load(r/'compliance_direct_probe'/(name+'.npz'))
 phases={}
 for phase,(lo,hi) in [('landing',(.04,.3)),('standing',(.3,1)),('moving',(1,1.3))]:
  mask=(time>=lo-1e-8)&(time<hi-1e-8)
  err=d['joints'][mask,8:]-native[mask,8:]
  rms=np.sqrt(np.mean(err**2,axis=(0,1)))
  phases[phase]=dict(joint_rms_rad=dict(zip(names,map(float,rms))),signed_mean_rad=dict(zip(names,map(float,err.mean(axis=(0,1))))),largest_three=[names[i] for i in np.argsort(rms)[-3:][::-1]])
 out[name]=phases
atomic_json(r/'component_analysis.json',dict(scope='Post-hoc validation-seed localization, not an extra experiment or revised eligibility gate',results=out))
print(json.dumps({k:{p:v['largest_three'] for p,v in phases.items()} for k,phases in out.items()}))
