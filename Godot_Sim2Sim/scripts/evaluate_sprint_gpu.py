"""Deterministic-policy GPU proxy evaluation from frozen native reset states.

Never substitutes for native Jolt acceptance. No training, sampling noise,
auto-resets, or rewards are involved in the scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from sim2sim.play_input import PlayBrain, TwistLimits
from sim2sim.research.models import NativeAnchor
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.standalone.sprint import metrics
from sprint_gpu_world import GpuWorld


def score_states(states, case, commands, selections):
    count=round(case['seconds']/.02)
    data=np.asarray(states[1:count+1],float)
    if data.shape!=(count,13) or not np.isfinite(data).all():raise ValueError('Invalid post-action states')
    q=data[:,3:7];q=q/np.linalg.norm(q,axis=1,keepdims=True)
    yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    v=data[:,10:13];cy,sy=np.cos(yaw),np.sin(yaw)
    velocity=np.stack((cy*v[:,0]+sy*v[:,1],-sy*v[:,0]+cy*v[:,1],v[:,2]),axis=1)
    rows=[dict(xy=d[:2].tolist(),z=float(d[2]),tilt=float(tilt[i]),yaw=float(yaw[i]),vel=velocity[i].tolist()) for i,d in enumerate(data)]
    actions=[dict(t=i*.02,skill='sprint' if selections[i] else 'walking',requested_command=commands[i]) for i in range(count)]
    return metrics(rows,actions,case)


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-summary',type=Path,required=True);p.add_argument('--actor',type=Path,required=True);p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads(a.native_summary.read_text());assert summary['errors']==0
    effective_control=json.loads(a.control.read_text());settings=effective_control['walk'];cases=[];first=[];tapes=[];selections=[];sources=[]
    if settings['twist_limits']['sprint_vmax_ang']!=.8:raise ValueError('Native sprint cases require angular limit 0.8')
    for e in summary['episodes']:
        case=json.loads(Path(e['case_path']).read_text());trace=Path(e['trace']);row=json.loads(trace.read_text())['rows'][0]
        # The old native trace supplies only a reset and keyboard program.
        # Apply the explicitly requested evaluation control in both domains.
        case['control_config']=effective_control
        assert row['episode_t']==0 and not any(row['last_action'])
        assert max(abs(v) for v in row['raw']['qd']+row['raw']['base_linvel']+row['raw']['base_angvel_local'])<1e-7
        brain=PlayBrain(has_standing=False,has_sprint=True,lim=TwistLimits(**settings['twist_limits']));commands=[];selected=[]
        for i in range(1250):
            held=next(s['held'] for s in reversed(case['segments']) if s['at']<=i*.02+1e-9)
            step=brain.tick(set(held),[],.02,press_order=sorted(held));commands.append(step.command);selected.append(step.sprint)
        cases.append(case);first.append(row);tapes.append(commands);selections.append(selected)
        sources.append(dict(case=case['case'],seed=case['seed'],trace=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    n=len(cases);requested=np.array(tapes).transpose(1,0,2);selected=np.array(selections).T
    req=torch.tensor(requested,device='cuda',dtype=torch.float32);sel=torch.tensor(selected,device='cuda',dtype=torch.bool)
    actor=TorchAnchor(a.actor).cuda().eval();ordinary=TorchAnchor(a.ordinary).cuda().eval()
    world=GpuWorld(n,settings,feet='jolt',graphs=True,velocity_observer='joint_fd');world.enable_swing_monitor()
    try:
        world.qpos[:,world.qa:world.qa+3]=torch.tensor([x['body']['base_pos'] for x in first],device='cuda')
        world.qpos[:,world.qa+3:world.qa+7]=torch.tensor([x['body']['base_quat'] for x in first],device='cuda')
        world.qpos[:,world.qi]=torch.tensor([x['raw']['q'] for x in first],device='cuda');world.qvel.zero_();world.forward()
        world.observer.reset(torch.arange(n,device='cuda'),world.qpos[:,world.qi],world.qpos[:,world.qa+3:world.qa+7])
        inputs=np.concatenate([np.random.default_rng(929110).normal(0,.5,(128,61)).astype(np.float32),np.array([x['obs'] for x in first],np.float32)])
        with torch.inference_mode():actual=actor(torch.from_numpy(inputs).cuda()).cpu().numpy()
        parity=float(np.abs(actual-NativeAnchor(a.actor)(inputs)).max());assert parity<1e-5
        states=[];actions=[];observations=[];feet=[];contacts=[];air_time=[];clearance=[]
        start=time.monotonic()
        with torch.inference_mode():
            for i in range(1250):
                pos,q,ang,v=world.state();states.append(torch.cat([pos,q,ang,v],dim=-1).clone())
                obs,_=world.observe(req[i],sel[i]);action=torch.where(sel[i,:,None],actor(obs),ordinary(obs))
                observations.append(obs.clone());actions.append(action.clone());world.step(action)
                feet.append(wp.to_torch(world.wd.xipos)[:,world.foot_bodies].clone())
                contacts.append(world.swing.contact.clone());air_time.append(world.swing.air_time.clone());clearance.append(world.swing.clearance.clone())
            pos,q,ang,v=world.state();states.append(torch.cat([pos,q,ang,v],dim=-1).clone())
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        data={k:torch.stack(v).cpu().numpy() for k,v in dict(states=states,actions=actions,obs=observations,feet=feet,contacts=contacts,air_time=air_time,clearance=clearance).items()}
        if not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite rollout')
        np.savez_compressed(a.output/'trajectories.npz',**data,requested=requested,sprint=selected)
        results=[dict(case=case['case'],seed=case['seed'],metrics=score_states(data['states'][:,j],case,tapes[j],selections[j])) for j,case in enumerate(cases)]
        result=dict(completed=True,actor_sha256=actor.sha256,ordinary_sha256=ordinary.sha256,control_sha256=hashlib.sha256(a.control.read_bytes()).hexdigest(),effective_control=effective_control,parity_max_abs=parity,elapsed_s=elapsed,passes=sum(x['metrics']['success'] for x in results),falls=sum(x['metrics']['fell'] for x in results),count=n,results=results,native_start_sources=sources,scope='GPU proxy only; post-action scoring including final state, no resets/no sampling noise. Native source traces supply reset/keys only; requested control overrides their old speed.')
        (a.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['passes','count','falls','elapsed_s']})
    finally:world.close()


if __name__=='__main__':main()
