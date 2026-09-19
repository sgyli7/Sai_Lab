from pathlib import Path
import json,sys,time
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.replay import raw_state
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
r=Path('results/sprint_gpu_match_20260912');out=r/'response';out.mkdir(exist_ok=False)
source=Path('results/sprint_exit_20260912/gpu_dynamics_probe');data=np.load(source/'states.npz');selected=np.load(source/'torque_only.npz')['selected']
future=json.loads((source/'targets.json').read_text());index=json.loads((source/'index.json').read_text());spec=json.loads(Path('godot/generated/microduck_ball_stand_fix/robot_spec.json').read_text());base=next(x for x in spec['bodies'] if x['name']=='trunk_base');robot=dict(ipos=base['ipos'],iquat=base['iquat_wxyz'])
expected=[raw_state(future[i]['raw'],robot) for i in selected]
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk'];torch.set_num_threads(4)
def stats(values):
 x=np.array(values);return dict(rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.quantile(np.abs(x),.95)),max_abs=float(np.abs(x).max()))
results={}
for feet,contact in [('source','source'),('jolt','source'),('jolt','two_tick')]:
 world=GpuWorld(len(selected),settings,feet=feet,contact=contact)
 try:
  world.qpos.copy_(torch.tensor(data['qpos'][selected],device='cuda',dtype=world.qpos.dtype));world.qvel.copy_(torch.tensor(data['qvel'][selected],device='cuda',dtype=world.qvel.dtype))
  with wp.ScopedStream(world.stream):mw.forward(world.wm,world.wd)
  action=torch.tensor(data['target'][selected],device='cuda',dtype=torch.float32)-world.home
  world.step(action);torch.cuda.synchronize()
  pos,q,ang,v=world.state();pos=pos.cpu().numpy();q=q.cpu().numpy();v=v.cpu().numpy();jp=world.qpos[:,world.qi].cpu().numpy();jv=world.qvel[:,world.vi].cpu().numpy()
  errors=dict(position=pos-np.array([s.base_pos for s in expected]),com_velocity=v-np.array([s.base_linvel for s in expected]),joint_position=jp-np.array([s.q for s in expected]),joint_velocity=jv-np.array([s.qd for s in expected]))
  groups={}
  for name,mask in [('all',np.ones(len(selected),bool)),('s05',np.array(['state_shadow' in index[i]['trace'] for i in selected])),('donor',np.array(['state_shadow' not in index[i]['trace'] for i in selected]))]:
   groups[name]={key:stats(values[mask]) for key,values in errors.items()}
  key=feet+'_'+contact;results[key]=dict(rows=len(selected),groups=groups,foot_hulls=world.feet)
  atomic_json(out/'progress.json',results);print(key,groups['all'],flush=True)
 finally:world.close()
atomic_json(out/'completed.json',dict(status='completed',results=results,scope='Same 90 prefailure initial states and four 5 ms steps; exact game foot geometry and a separately identified contact time constant',contact_two_tick='0.01 s vs source 0.02 s, damping ratio unchanged at 1',game_changed=False))
