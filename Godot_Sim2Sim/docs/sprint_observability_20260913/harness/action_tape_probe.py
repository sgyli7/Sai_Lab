"""Same native action tapes, GPU physics only; never an acceptance replay."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch

from sim2sim.research.queue import atomic_json
from sim2sim.research.torch_walking import inverse_rotate
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld

ROOT=Path.cwd()
SESSION=ROOT/'results/sprint_observability_20260913'
OUT=SESSION/'action_tape'


def heading(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def main():
    if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=SESSION:
        raise RuntimeError('Active budget required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(exist_ok=False)
    settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
    summary=json.loads(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json').read_text())
    selected=sorted([e for e in summary['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    assert len(selected)==16
    data=[json.loads(Path(e['trace']).read_text()) for e in selected]
    rows=[[r for r in d['rows'] if 'obs' in r] for d in data]
    n=len(rows);steps=450
    assert all(r[0]['t']==0 and all(abs(v)<1e-7 for v in r[0]['raw']['qd']+r[0]['raw']['base_linvel']+r[0]['raw']['base_angvel_local']) for r in rows)
    actions=np.array([[r[i]['action'] for r in rows] for i in range(steps)],np.float32)
    targets=np.array([[r[i]['ctrl'] for r in rows] for i in range(steps)],np.float32)
    native_pos=np.array([[r[i]['body']['base_pos'] for r in rows] for i in range(steps+1)])
    native_quat=np.array([[r[i]['body']['base_quat'] for r in rows] for i in range(steps+1)])
    native_vel=np.array([[r[i]['body']['base_linvel'] for r in rows] for i in range(steps+1)])
    native_q=np.array([[r[i]['raw']['q'] for r in rows] for i in range(steps+1)])
    native_obs=np.array([[r[i]['obs'] for r in rows] for i in range(steps+1)])
    atomic_json(OUT/'protocol.json',dict(steps=steps,seconds=9,worlds=n,variants=['solver','joint_fd'],
        policy_evaluation=False,physics='Frozen GPU proxy with actual Jolt feet and rotor mapping; no game edits',
        targets='Exact recorded float32 action tapes; verify home+action against native ctrl',
        initial_state='Project native zero-velocity articulated reset; no later state import, no cold restore',
        metrics_at_s=[.02,.1,.2,.5,1,3,6,9],
        limitation='Initial visible-state agreement does not establish equality of solver/contact internal state. Later divergence includes accumulated open-loop error, not a local contact error estimate.',
        sources=[dict(case=e['case'],seed=e['seed'],trace=e['trace'],sha256=hashlib.sha256(Path(e['trace']).read_bytes()).hexdigest()) for e in selected],
        code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/sprint_gpu_world.py',ROOT/'scripts/sprint_gpu_proxy.py',ROOT/'src/sim2sim/research/kinematic_observer.py']}))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for variant in ('solver','joint_fd'):
        world=GpuWorld(n,settings,feet='jolt',graphs=True,velocity_observer=variant)
        try:
            world.qpos[:,world.qa:world.qa+3]=torch.as_tensor(native_pos[0],device='cuda',dtype=torch.float32)
            world.qpos[:,world.qa+3:world.qa+7]=torch.as_tensor(native_quat[0],device='cuda',dtype=torch.float32)
            world.qpos[:,world.qi]=torch.as_tensor(native_q[0],device='cuda',dtype=torch.float32)
            world.qvel.zero_();world.forward()
            if world.observer is not None:world.observer.reset(torch.arange(n,device='cuda'),world.qpos[:,world.qi],world.qpos[:,world.qa+3:world.qa+7])
            action=torch.as_tensor(actions,device='cuda')
            target_error=float(np.abs((action+world.home).cpu().numpy()-targets).max())
            if target_error>1e-6:raise RuntimeError('Actuation targets differ')
            state=torch.empty((steps+1,n,13),device='cuda')
            q=torch.empty((steps+1,n,14),device='cuda')
            observations=torch.empty((steps+1,n,61),device='cuda')
            gravity=torch.tensor([0.,0.,-1.],device='cuda').expand(n,-1)
            native_command=torch.as_tensor(native_obs[:,:,48:],device='cuda',dtype=torch.float32)
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():
                for i in range(steps+1):
                    pos,quat,ang,vel=world.state()
                    state[i]=torch.cat((pos,quat,ang,vel),dim=1)
                    q[i]=world.qpos[:,world.qi]
                    qd=world.qvel[:,world.vi] if world.observer is None else world.observer.qd
                    observations[i]=torch.cat((ang,inverse_rotate(quat,gravity),q[i]-world.home,qd,world.last,native_command[i]),dim=1)
                    if i<steps:world.step(action[i])
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            state=state.cpu().numpy();q=q.cpu().numpy();observations=observations.cpu().numpy()
            if not np.isfinite(state).all():raise RuntimeError('Nonfinite open-loop state')
            initial_obs_error=float(np.abs(observations[0]-native_obs[0]).max())
            last_error=float(np.abs(observations[1:,:,34:48]-actions).max())
            if initial_obs_error>1e-6 or last_error!=0:raise RuntimeError('Initial observation/action history differs')
            pos_error=np.linalg.norm(state[:,:,:3]-native_pos,axis=-1)
            vel_error=np.linalg.norm(state[:,:,10:13]-native_vel,axis=-1)
            joint_error=np.sqrt(np.mean((q-native_q)**2,axis=-1))
            yaw_error=np.abs((heading(state[:,:,3:7])-heading(native_quat)+np.pi)%(2*np.pi)-np.pi)*180/np.pi
            metrics={}
            for seconds in [.02,.1,.2,.5,1,3,6,9]:
                i=round(seconds/.02)
                metrics[str(seconds)]={name:dict(median=float(np.median(v[i])),max=float(v[i].max()),
                    original_seed_927001=float(v[i,1])) for name,v in dict(body_position_m=pos_error,com_velocity_mps=vel_error,joint_rms_rad=joint_error,heading_deg=yaw_error).items()}
            results[variant]=dict(initial_obs_max_abs=initial_obs_error,last_action_max_abs=last_error,
                control_target_max_abs=target_error,elapsed_s=elapsed,metrics=metrics,
                minimum_body_height=float(state[:,:,2].min()),
                scope='CUDA replay of identical recorded actions; not closed-loop policy quality or local contact identification')
            np.savez_compressed(OUT/(variant+'.npz'),gpu_states=state,gpu_joint_q=q,gpu_obs=observations,
                native_pos=native_pos,native_quat=native_quat,native_vel=native_vel,native_joint_q=native_q,
                actions=actions,seeds=np.array([e['seed'] for e in selected]))
            atomic_json(OUT/(variant+'.json'),results[variant])
            print(json.dumps(dict(variant=variant,initial_obs=initial_obs_error,metrics=metrics)),flush=True)
        finally:world.close()
    atomic_json(OUT/'completed.json',dict(results=results,completed=True,game_physics_changed=False,
        new_native_sampling=False,policy_changes=False))


if __name__=='__main__':main()
