"""Replay frozen native starts in the GPU proxy; no model updates or tuning."""
from pathlib import Path
import argparse, json, sys, time, hashlib
import numpy as np
import torch
from sim2sim.play_input import PlayBrain, TwistLimits
from sim2sim.standalone.sprint import metrics
from sim2sim.research.models import NativeAnchor
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.queue import atomic_json
sys.path.insert(0, str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--feet',choices=['source','jolt'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(exist_ok=False,parents=True)
r=Path('results/sprint_joint_gpu_20260912');source=a.ordinary
summary=json.loads((r/'baseline_s05/suite/summary.json').read_text());assert summary['errors']==0
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
cases=[];first=[];tapes=[];selections=[];initial_checks=[]
for e in summary['episodes']:
 case=json.loads(Path(e['case_path']).read_text());data=json.loads(Path(e['trace']).read_text());row=data['rows'][0]
 assert row['t']==0 and row['episode_t']==0 and not any(row['last_action'])
 assert max(abs(v) for v in row['raw']['qd']+row['raw']['base_linvel']+row['raw']['base_angvel_local'])<1e-7
 brain=PlayBrain(has_standing=False,has_sprint=True,lim=TwistLimits(**settings['twist_limits']));tape=[];sel=[]
 for i in range(1250):
  held=next(s.get('held',[]) for s in reversed(case['segments']) if s['at']<=i*.02+1e-9)
  step=brain.tick(set(held),[],.02,press_order=sorted(held));tape.append(step.command);sel.append(step.sprint)
 cases.append(case);first.append(row);tapes.append(tape);selections.append(sel)
 initial_checks.append(dict(case=e['case'],seed=e['seed'],case_sha256=hashlib.sha256(Path(e['case_path']).read_bytes()).hexdigest(),trace=e['trace']))
n=len(cases);assert n==263
requested=torch.tensor(np.array(tapes).transpose(1,0,2),device='cuda',dtype=torch.float32);sprint=torch.tensor(np.array(selections).T,device='cuda',dtype=torch.bool)
world=GpuWorld(n,settings,feet=a.feet,graphs=True);ordinary=TorchAnchor(source).cuda().eval();actor=TorchAnchor(a.actor).cuda().eval()
# Project the zero-velocity native reset onto generalized coordinates. Body
# poses at reset are articulated, unlike an arbitrary contact snapshot.
world.qpos[:,world.qa:world.qa+3]=torch.tensor([x['body']['base_pos'] for x in first],device='cuda')
world.qpos[:,world.qa+3:world.qa+7]=torch.tensor([x['body']['base_quat'] for x in first],device='cuda')
world.qpos[:,world.qi]=torch.tensor([x['raw']['q'] for x in first],device='cuda')
world.qvel.zero_();world.forward()
inputs=np.concatenate([np.random.default_rng(927014).normal(0,.5,(256,61)).astype(np.float32),np.array([x['obs'] for x in first],np.float32)])
with torch.inference_mode():actual=actor(torch.from_numpy(inputs).cuda()).cpu().numpy()
expected=NativeAnchor(a.actor)(inputs);parity=float(np.abs(actual-expected).max());assert parity<1e-5
states=torch.empty((1250,n,13),device='cuda');actions=torch.empty((1250,n,14),device='cuda');observations=torch.empty((1250,n,61),device='cuda');commands=torch.empty_like(requested)
try:
 torch.cuda.synchronize();start=time.monotonic()
 with torch.inference_mode():
  for i in range(1250):
   pos,q,ang,v=world.state();states[i]=torch.cat([pos,q,ang,v],dim=1)
   obs,cmd=world.observe(requested[i],sprint[i]);observations[i]=obs;commands[i]=cmd
   action=torch.where(sprint[i,:,None],actor(obs),ordinary(obs));actions[i]=action;world.step(action)
   if i%250==249:atomic_json(a.out/'progress.json',dict(decisions=i+1,worlds=n,elapsed_s=time.monotonic()-start))
 torch.cuda.synchronize();elapsed=time.monotonic()-start
 data=states.cpu().numpy();acts=actions.cpu().numpy();obs=observations.cpu().numpy();cmds=commands.cpu().numpy()
 assert np.isfinite(data).all() and np.isfinite(acts).all()
 np.savez_compressed(a.out/'trajectories.npz',states=data,actions=acts,obs=obs,commands=cmds,requested=np.array(tapes).transpose(1,0,2),sprint=np.array(selections).T)
 atomic_json(a.out/'rollout_completed.json',dict(elapsed_s=elapsed,initial_starts=initial_checks,feet=world.feet,parity_max_abs=parity))
 results=[]
 for j,case in enumerate(cases):
  rows=[];action_rows=[];count=round(case['seconds']/.02)
  for i in range(count):
   pos=data[i,j,:3];q=data[i,j,3:7].astype(float);v=data[i,j,10:13]
   yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));tilt=np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
   local=[np.cos(yaw)*v[0]+np.sin(yaw)*v[1],-np.sin(yaw)*v[0]+np.cos(yaw)*v[1],v[2]]
   rows.append(dict(xy=pos[:2].tolist(),z=float(pos[2]),tilt=tilt,yaw=yaw,vel=local))
   action_rows.append(dict(t=i*.02,skill='sprint' if selections[j][i] else 'walking',requested_command=tapes[j][i]))
  results.append(dict(case=case['case'],seed=case['seed'],ordinary=case.get('ordinary_control',False),metrics=metrics(rows,action_rows,case)))
 candidates=[x for x in results if not x['ordinary']]
 atomic_json(a.out/'completed.json',dict(status='completed',actor_sha256=actor.sha256,ordinary_sha256=ordinary.sha256,feet=a.feet,worlds=n,elapsed_rollout_s=elapsed,parity_max_abs=parity,passes=sum(x['metrics']['success'] for x in candidates),falls=sum(x['metrics']['fell'] for x in candidates),candidate_count=len(candidates),results=results,scope='Projected native zero-velocity reset and identical input tapes; different contact solver, no native acceptance substitution'))
 print(a.out,'pass',sum(x['metrics']['success'] for x in candidates),'/',len(candidates),'falls',sum(x['metrics']['fell'] for x in candidates),flush=True)
finally:world.close()
