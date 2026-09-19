from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from sim2sim.play_input import PlayBrain,TwistLimits
from sim2sim.standalone.sprint import templates,metrics
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
p=argparse.ArgumentParser();p.add_argument('--contact',choices=['source','two_tick'],required=True);p.add_argument('--feet',choices=['source','jolt'],default='source');p.add_argument('--graphs',action='store_true');a=p.parse_args()
r=Path('results/sprint_gpu_match_20260912');out=r/('baseline_'+a.contact+('_jolt_feet' if a.feet=='jolt' else '')+('_graphs' if a.graphs else ''));out.mkdir(exist_ok=False)
assert json.loads((r/'anchor_parity/progress.json').read_text())['s05']['passed']
assert json.loads((r/'gpu_contract_attempt02/completed.json').read_text())['passed']
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
programs=templates();names=list(programs);n=len(names)*2;length=1250
tapes=[];selections=[];cases=[]
for name in names:
 for seed in [925000,925001]:
  case=programs[name];brain=PlayBrain(has_standing=False,has_sprint=True,lim=TwistLimits(**settings['twist_limits']));tape=[];sel=[]
  for i in range(length):
   held=next(s['held'] for s in reversed(case['segments']) if s['at']<=i*.02+1e-9)
   step=brain.tick(set(held),[],.02,press_order=sorted(held));tape.append(step.command);sel.append(step.sprint)
  cases.append(dict(name=name,seed=seed,**case));tapes.append(tape);selections.append(sel)
requested=torch.tensor(np.array(tapes).transpose(1,0,2),device='cuda',dtype=torch.float32)
sprint=torch.tensor(np.array(selections).T,device='cuda',dtype=torch.bool)
world=GpuWorld(n,settings,contact=a.contact,feet=a.feet,graphs=a.graphs);actor=TorchAnchor('results/sprint_20260912/runs/s05_native_handoff/final.onnx').cuda().eval()
perturb=np.stack([np.zeros(14) if c['seed']==925000 else np.random.default_rng(c['seed']).normal(0,.003,14) for c in cases]).astype(np.float32)
world.reset(torch.arange(n,device='cuda'),torch.from_numpy(perturb).cuda())
states=torch.empty((length,n,13),device='cuda');actions=torch.empty((length,n,14),device='cuda');observations=torch.empty((length,n,61),device='cuda');commands=torch.empty_like(requested)
try:
 torch.cuda.synchronize();start=time.monotonic()
 with torch.inference_mode():
  for i in range(length):
   pos,q,ang,v=world.state();states[i]=torch.cat([pos,q,ang,v],dim=1)
   obs,cmd=world.observe(requested[i],sprint[i]);observations[i]=obs;commands[i]=cmd
   action=actor(obs);actions[i]=action;world.step(action)
   if i%250==249:
    atomic_json(out/'progress.json',dict(decisions=i+1,worlds=n,elapsed_s=time.monotonic()-start))
 torch.cuda.synchronize();elapsed=time.monotonic()-start
 data=states.cpu().numpy();acts=actions.cpu().numpy();obs=observations.cpu().numpy();cmds=commands.cpu().numpy()
 assert np.isfinite(data).all() and np.isfinite(acts).all()
 np.savez_compressed(out/'trajectories.npz',states=data,actions=acts,obs=obs,commands=cmds,requested=np.array(tapes).transpose(1,0,2),sprint=np.array(selections).T,perturbation=perturb)
 atomic_json(out/'rollout_completed.json',dict(elapsed_rollout_s=elapsed,steps=length,worlds=n,feet=world.feet))
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
 atomic_json(out/'completed.json',dict(status='completed',contact=a.contact,feet=a.feet,graphs=a.graphs,source_sha256=actor.sha256,worlds=n,seconds_per_world=25,elapsed_rollout_s=elapsed,aggregate_decisions_per_s=length*n/elapsed,device=torch.cuda.get_device_name(),passes=sum(x['metrics']['success'] for x in results),falls=sum(x['metrics']['fell'] for x in results),results=results,scope='GPU proxy screening; not native Jolt acceptance or throughput benchmark against competing workloads',initialization='nominal HOME at 0.125 m; second seed joint perturbation sigma .003 rad, not native randomized pose protocol'))
 print(dict(contact=a.contact,passes=sum(x['metrics']['success'] for x in results),falls=sum(x['metrics']['fell'] for x in results),elapsed=elapsed),flush=True)
finally:world.close()
