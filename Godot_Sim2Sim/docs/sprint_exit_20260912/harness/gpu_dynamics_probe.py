"""GPU short-horizon response audit; no learning, no Godot processes or edits."""
from pathlib import Path
import hashlib,json,os,sys,time
import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.standalone.replay import raw_state
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_proxy import patch_game_inertia

r=Path('results/sprint_exit_20260912');out=r/'gpu_dynamics_probe'
if not os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR'):raise RuntimeError('Supervisor required')
if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
torch.set_num_threads(1)
data=np.load(out/'states.npz');index=json.loads((out/'index.json').read_text());future=json.loads((out/'targets.json').read_text());projection=json.loads((out/'projection.json').read_text())
first={path:json.loads(Path(path).read_text())['summary'].get('first_fall') for path in {row['trace'] for row in index}}
selected=[i for i,row in enumerate(index) if first[row['trace']] is None or row['t']+.02<first[row['trace']]-1e-9]
qp=data['qpos'][selected];qv=data['qvel'][selected];targets=data['target'][selected];qi=data['joint_qpos_indices'];vi=data['joint_qvel_indices'];base=int(data['base_body']);n=len(selected)
spec=json.loads(Path(projection['godot_spec']).read_text());meta=next(x for x in spec['bodies'] if x['name']=='trunk_base');robot=dict(ipos=meta['ipos'],iquat=meta['iquat_wxyz'])
expected=[raw_state(future[i]['raw'],robot) for i in selected]
limit=1.75*.36601349688984386
def torque(q,v,target):return np.clip(.55*(target-q),-limit,limit)-.053*v-.0048*np.tanh(v/.05)
def stats(x):
 x=np.asarray(x);return dict(rms=float(np.sqrt(np.mean(x*x))),p50_abs=float(np.median(np.abs(x))),p95_abs=float(np.quantile(np.abs(x),.95)),max_abs=float(np.max(np.abs(x))))
def summary(model,positions,velocities):
 d=mujoco.MjData(model);pe=[];ve=[];angles=[];je=[];jve=[]
 for pos,vel,truth in zip(positions,velocities,expected):
  d.qpos[:]=pos;d.qvel[:]=vel;mujoco.mj_forward(model,d)
  v=np.zeros(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_BODY,base,v,0)
  pe.append(d.xpos[base]-truth.base_pos);ve.append(v[3:]-truth.base_linvel)
  rotation=d.xmat[base].reshape(3,3).T@quat_wxyz_to_mat(truth.base_quat_wxyz)
  angles.append(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
  je.append(pos[qi]-truth.q);jve.append(vel[vi]-truth.qd)
 return dict(base_position_m=stats(pe),base_com_velocity_mps=stats(ve),base_rotation_deg=stats(angles),joint_position_rad=stats(je),joint_velocity_radps=stats(jve))
results={};started=time.time()
atomic_json(out/'gpu_started.json',dict(unix=started,device=torch.cuda.get_device_name(),snapshots=n,excluded_after_first_fall=len(index)-n,physics_hz=200,substeps=4,no_learning=True))
for kind in ['torque_only','torque_and_rotor']:
 start=time.monotonic();s=mujoco.MjSpec.from_file(projection['mjcf'])
 for actuator in s.actuators:
  joint=s.joint(str(actuator.target));joint.armature=.0018;joint.damping=np.zeros((3,1));joint.frictionloss=0.
  actuator.set_to_motor();actuator.gear=[1.,0.,0.,0.,0.,0.];actuator.forcelimited=False;actuator.ctrllimited=False
 model=s.compile();model.opt.timestep=.005
 inertia=None
 if kind=='torque_and_rotor':inertia=patch_game_inertia(model,projection['godot_spec'])
 seed=mujoco.MjData(model);mujoco.mj_forward(model,seed)
 with wp.ScopedDevice('cuda:0'):
  wm=mw.put_model(model);wd=mw.put_data(model,seed,nworld=n,nconmax=128,njmax=2048)
  qt=wp.to_torch(wd.qpos);vt=wp.to_torch(wd.qvel);ct=wp.to_torch(wd.ctrl)
  qt.copy_(torch.as_tensor(qp,device='cuda',dtype=qt.dtype));vt.copy_(torch.as_tensor(qv,device='cuda',dtype=vt.dtype));ct.zero_()
  torch.cuda.synchronize();mw.forward(wm,wd);wp.synchronize_device()
  target=torch.as_tensor(targets,device='cuda',dtype=qt.dtype);qidx=torch.as_tensor(qi,device='cuda');vidx=torch.as_tensor(vi,device='cuda')
  for step in range(4):
   speed=vt[:,vidx];ct.copy_((.55*(target-qt[:,qidx])).clamp(-limit,limit)-.053*speed-.0048*torch.tanh(speed/.05))
   torch.cuda.synchronize();mw.step(wm,wd);wp.synchronize_device()
  actual_q=qt.cpu().numpy().copy();actual_v=vt.cpu().numpy().copy()
  if not np.isfinite(actual_q).all() or not np.isfinite(actual_v).all():raise RuntimeError('Nonfinite GPU state')
  np.savez_compressed(out/(kind+'.npz'),qpos=actual_q,qvel=actual_v,selected=selected)
  # Small CPU/GPU numerical audit of the same model. It is not training and
  # does not assert that MuJoCo CPU and Warp contact solvers are bit-identical.
  probes=sorted({0,n//3,2*n//3,n-1});cpu=[];cpu_q=[];cpu_v=[]
  for i in probes:
   d=mujoco.MjData(model);d.qpos[:]=qp[i];d.qvel[:]=qv[i];mujoco.mj_forward(model,d)
   for step in range(4):d.ctrl[:]=torque(d.qpos[qi],d.qvel[vi],targets[i]);mujoco.mj_step(model,d)
   cpu.append(dict(snapshot=selected[i],qpos_max_abs=float(np.abs(d.qpos-actual_q[i]).max()),qvel_max_abs=float(np.abs(d.qvel-actual_v[i]).max())))
  results[kind]=dict(gpu_vs_jolt_20ms=summary(model,actual_q,actual_v),cpu_gpu_same_model=cpu,inertia=inertia,elapsed_including_compile_seconds=time.monotonic()-start)
  # Separate speed-model transfer from ordinary walking, rather than mixing
  # different motion amplitudes into a single error statistic.
  for label in ['s05','donor']:
   mask=[j for j,i in enumerate(selected) if ('state_shadow' in index[i]['trace'])==(label=='s05')]
   saved=expected;expected=[saved[j] for j in mask]
   results[kind][label]=summary(model,actual_q[mask],actual_v[mask]);expected=saved
  atomic_json(out/'gpu_progress.json',results);print(kind,results[kind]['gpu_vs_jolt_20ms'],flush=True)
  del qt,vt,ct,wm,wd,target;torch.cuda.empty_cache()
atomic_json(out/'gpu_completed.json',dict(status='completed',device=torch.cuda.get_device_name(),mujoco=mujoco.__version__,torch=torch.__version__,cuda=torch.version.cuda,rows=n,excluded_after_first_fall=len(index)-n,results=results,scope='Actual CUDA physics for 20 ms from projected Jolt states; no reward optimization, no game physics changes',limitations=['Projection cannot preserve independent rigid-body constraint errors/velocities; see projection.json','MuJoCo/Warp contacts differ from Jolt, even with identical motor equation','Rotor proxy includes non-solid diagonal inertia and is an explicit research approximation','This measures neither long-horizon validity nor learning throughput'],finished_unix=time.time()))
