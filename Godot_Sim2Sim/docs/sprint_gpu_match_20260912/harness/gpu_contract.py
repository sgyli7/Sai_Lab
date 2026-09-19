from pathlib import Path
import json,sys
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from sim2sim.research.torch_walking import WalkingControl,rotate,inverse_rotate
from sim2sim.motion_control import MotionControl
from sim2sim.standalone.replay import raw_state
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
r=Path('results/sprint_gpu_match_20260912');out=r/'gpu_contract';out.mkdir(exist_ok=False)
torch.set_num_threads(4);settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
trace=json.loads(Path('results/sprint_exit_20260912/state_shadow/trace.json').read_text())
deploy=json.loads(Path('results/sprint_exit_20260912/state_contract/suite/runtime/runtime_assets/deployment.json').read_text());robot=deploy['robots']['walk']
control=WalkingControl(2,settings,'cuda');reference=[MotionControl(settings),MotionControl(settings)];error=0.
for i,row in enumerate(trace['rows']):
 state=raw_state(row['raw'],robot)
 requested=np.array(row['requested_command'],np.float32)
 if i==350:control.reset(torch.tensor([1],device='cuda'));reference[1].reset()
 expected=np.stack([ref.command(requested,state,row['skill']) for ref in reference])
 tensor=lambda a:torch.as_tensor(np.array(a),device='cuda')
 actual=control.command(tensor(np.repeat(requested[None],2,axis=0)),tensor(np.repeat(state.base_pos[None],2,axis=0)),tensor(np.repeat(state.base_quat_wxyz[None],2,axis=0)),tensor(np.repeat(state.base_linvel[None],2,axis=0)),torch.full((2,),row['skill']=='sprint',device='cuda')).cpu().numpy()
 error=max(error,float(np.abs(actual-expected).max()))
assert error<1e-5,error
world=GpuWorld(4,settings)
try:
 snapshots=np.load('results/sprint_exit_20260912/gpu_dynamics_probe/states.npz')
 indices=[0,20,40,60];world.qpos.copy_(tensor(snapshots['qpos'][indices]));world.qvel.copy_(tensor(snapshots['qvel'][indices]))
 pos,q,angular,vel=world.state();errors=[]
 for i in range(4):
  d=mujoco.MjData(world.model);d.qpos[:]=world.qpos[i].cpu().numpy();d.qvel[:]=world.qvel[i].cpu().numpy();mujoco.mj_forward(world.model,d)
  v=np.zeros(6);mujoco.mj_objectVelocity(world.model,d,mujoco.mjtObj.mjOBJ_BODY,world.base,v,0)
  local=d.xmat[world.base].reshape(3,3).T@v[:3]
  errors.append(dict(com_velocity=float(np.abs(vel[i].cpu().numpy()-v[3:]).max()),angular=float(np.abs(angular[i].cpu().numpy()-local).max()),position=float(np.abs(pos[i].cpu().numpy()-d.xpos[world.base]).max())))
 assert max(max(z.values()) for z in errors)<1e-5,errors
 world.reset(torch.arange(4,device='cuda'));obs,cmd=world.observe(torch.zeros((4,13),device='cuda'),torch.zeros(4,device='cuda',dtype=torch.bool));world.step(torch.zeros((4,14),device='cuda'))
 assert torch.isfinite(world.qpos).all() and torch.isfinite(world.qvel).all()
 atomic_json(out/'completed.json',dict(passed=True,real_control_rows=700,streams=2,independent_reset_at=350,control_max_abs=error,com_frame_errors=errors,smoke_physics_steps=4,device=torch.cuda.get_device_name()))
 print(json.loads((out/'completed.json').read_text()),flush=True)
finally:world.close()
