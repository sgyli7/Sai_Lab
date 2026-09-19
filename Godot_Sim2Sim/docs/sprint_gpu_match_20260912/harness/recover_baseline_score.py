from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
from sim2sim.standalone.sprint import templates,metrics
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912');out=r/'baseline_source';a=SimpleNamespace(contact='source',feet='source')
z=np.load(out/'trajectories.npz');data=z['states'];acts=z['actions'];obs=z['obs'];cmds=z['commands'];tapes=z['requested'].transpose(1,0,2);selections=z['sprint'].T
assert data.shape==(1250,16,13) and np.isfinite(data).all() and np.isfinite(acts).all()
cases=[dict(name=name,seed=seed,**case) for name,case in templates().items() for seed in [925000,925001]]
n=16;length=1250;elapsed=json.loads((out/'progress.json').read_text())['elapsed_s']
if True:
 results=[]
 for j,case in enumerate(cases):
  m=round(case['seconds']/.02);rows=[];action_rows=[]
  for i in range(m):
   pos=data[i,j,:3];q=data[i,j,3:7].astype(float);v=data[i,j,10:13]
   yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));tilt=np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
   local=[np.cos(yaw)*v[0]+np.sin(yaw)*v[1],-np.sin(yaw)*v[0]+np.cos(yaw)*v[1],v[2]]
   rows.append(dict(xy=pos[:2].tolist(),z=float(pos[2]),tilt=tilt,yaw=yaw,vel=local))
   action_rows.append(dict(t=i*.02,skill='sprint' if selections[j][i] else 'walking',requested_command=tapes[j][i]))
  sel=np.array(selections[j][:m]);edges=np.diff(np.r_[False,sel,False].astype(int));case['sprint_intervals']=[[l*.02,u*.02] for l,u in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
  result=metrics(rows,action_rows,case);results.append(dict(name=case['name'],seed=case['seed'],metrics=result))
 atomic_json(out/'completed.json',dict(status='completed_scoring_recovered',contact=a.contact,feet=a.feet,source_sha256='27ebbf83d63e5e59f125ceb158414cf7d2574990676d73c75b9c9870bcefed6e',worlds=n,seconds_per_world=25,elapsed_rollout_s=None,elapsed_lower_bound_s=elapsed,original_scoring_attempt='failed_signature_then_recovered_from_saved_arrays',device='NVIDIA GB10',passes=sum(x['metrics']['success'] for x in results),falls=sum(x['metrics']['fell'] for x in results),results=results,scope='GPU proxy screening; not native Jolt acceptance or throughput benchmark against competing workloads',initialization='nominal HOME at 0.125 m; second seed joint perturbation sigma .003 rad, not native randomized pose protocol'))
 print(dict(contact=a.contact,passes=sum(x['metrics']['success'] for x in results),falls=sum(x['metrics']['fell'] for x in results),elapsed=elapsed),flush=True)
