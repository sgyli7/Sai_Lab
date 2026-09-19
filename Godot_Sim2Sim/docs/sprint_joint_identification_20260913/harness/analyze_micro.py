"""Quantify position increments not represented by solver linear velocity."""
from pathlib import Path
import json
import numpy as np
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913')
info=json.loads((R/'micro_capture/completed.json').read_text());assert info['completed']
positions=[];velocities=[];times=None;names=None
for entry in info['results']:
 rows=json.loads((Path(entry['directory'])/'trace.json').read_text())['joint_micro']
 if names is None:names=[b['name'] for b in rows[0]['poses']];times=np.array([x['t'] for x in rows])
 positions.append([[b['pos'] for b in row['poses']] for row in rows]);velocities.append([[b['linvel'] for b in row['poses']] for row in rows])
p=np.swapaxes(np.array(positions),0,1);v=np.swapaxes(np.array(velocities),0,1)
residual=(p[1:]-p[:-1])-.005*v[1:]
# Compare post-step velocity because Jolt's velocity integration precedes its
# position integration. Same-time pose and velocity are read in one hook.
metrics={}
for phase,(lo,hi) in [('precontact',(.005,.04)),('landing',(.04,.3)),('standing',(.3,1)),('moving',(1,1.3))]:
 mask=(times[1:]>=lo-1e-8)&(times[1:]<hi-1e-8)
 entries=[]
 for i,name in enumerate(names):
  n=np.linalg.norm(residual[mask,:,i],axis=-1)
  entries.append(dict(name=name,median_m=float(np.median(n)),p95_m=float(np.quantile(n,.95)),max_m=float(n.max()),mean_vector_m=residual[mask,:,i].mean(axis=(0,1)).tolist(),solver_speed_rms=float(np.sqrt(np.mean(v[1:][mask,:,i]**2))),equivalent_correction_velocity_rms=float(np.sqrt(np.mean((residual[mask,:,i]/.005)**2)))))
 metrics[phase]=entries
np.savez_compressed(R/'micro_residuals.npz',positions=p,velocities=v,correction=residual,times=times)
atomic_json(R/'micro_analysis.json',dict(completed=True,names=names,metrics=metrics,scope='COM position increment minus post-step solver linear velocity*5ms. Includes any split positional stabilization; not a direct measurement of individual Jolt constraint impulses.'))
print(json.dumps({phase:[x for x in entries if x['name'] in ['trunk_base','ankle_left','ankle_right','ball']] for phase,entries in metrics.items()}))
