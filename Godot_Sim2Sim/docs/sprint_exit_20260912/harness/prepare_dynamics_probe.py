"""Project frozen Jolt snapshots into the exact source MJCF; no simulator runs."""
from pathlib import Path
import hashlib,json,os,time
import mujoco
import numpy as np
from sim2sim.coords import quat_wxyz_to_mat,mat_to_quat_wxyz
from sim2sim.paths import load_robot_json
from sim2sim.standalone.replay import raw_state
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'gpu_dynamics_probe';out.mkdir(exist_ok=False)
cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
model=mujoco.MjModel.from_xml_path(cfg['mjcf']);data=mujoco.MjData(model)
spec=json.loads(Path(cfg['godot_spec']).read_text());meta={b['name']:b for b in spec['bodies']}
base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
free=next(j for j in range(model.njnt) if model.jnt_bodyid[j]==base and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE)
qa=int(model.jnt_qposadr[free]);va=int(model.jnt_dofadr[free])
qi=np.array([model.jnt_qposadr[model.actuator_trnid[i,0]] for i in range(model.nu)],int)
vi=np.array([model.jnt_dofadr[model.actuator_trnid[i,0]] for i in range(model.nu)],int)
robot=dict(ipos=meta['trunk_base']['ipos'],iquat=meta['trunk_base']['iquat_wxyz'])
sources=[r/'state_shadow/trace.json',r/'running_zero_shot/suite/fast_source_2.2_923000/attempt_01/trace.json',r/'running_zero_shot/suite/fast_source_0.3_923000/attempt_01/trace.json']
qpos=[];qvel=[];controls=[];future=[];records=[];projection=[]
for path in sources:
 payload=path.read_bytes();trace=json.loads(payload);rows=trace['rows']
 # 20 ms transitions; source policy is held for exactly four physical steps.
 selected=[i for i in range(0,len(rows)-1,10) if not rows[i]['fell'] and not rows[i+1]['fell']]
 for i in selected:
  row=rows[i];state=raw_state(row['raw'],robot)
  mujoco.mj_resetData(model,data)
  # Keep the source scene's other free object away, as the standalone player does.
  for j in range(model.njnt):
   if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and j!=free:
    p=int(model.jnt_qposadr[j]);data.qpos[p:p+7]=[5.,5.,.035,1.,0.,0.,0.]
  data.qpos[qa:qa+7]=np.r_[state.base_pos,state.base_quat_wxyz];data.qpos[qi]=state.q
  data.qvel[vi]=state.qd;mujoco.mj_forward(model,data)
  jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(model,data,jp,jr,base)
  jac=np.vstack([jp,jr]);desired=np.r_[state.base_linvel,quat_wxyz_to_mat(state.base_quat_wxyz)@state.base_angvel_local]
  data.qvel[va:va+6]=np.linalg.solve(jac[:,va:va+6],desired-jac@data.qvel)
  mujoco.mj_forward(model,data)
  root_error=float(np.abs(jac@data.qvel-desired).max())
  if root_error>1e-8:raise RuntimeError('Root velocity reconstruction failed')
  errors={}
  for body in row['raw']['body_states']:
   if body['name']=='ball':continue
   bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body['name'])
   if bid<0:continue
   vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
   angular=data.ximat[bid].reshape(3,3).T@quat_wxyz_to_mat(body['quat'])
   errors[body['name']]=dict(position_m=float(np.linalg.norm(data.xipos[bid]-body['pos'])),rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(angular)-1)/2,-1,1)))),velocity_mps=float(np.linalg.norm(vel[3:]-body['linvel'])))
  projection.append(errors);qpos.append(data.qpos.copy());qvel.append(data.qvel.copy());controls.append(row['ctrl'])
  future.append(rows[i+1]);records.append(dict(trace=str(path),trace_sha256=hashlib.sha256(payload).hexdigest(),index=i,t=row['t'],skill=row['skill']))
np.savez_compressed(out/'states.npz',qpos=qpos,qvel=qvel,target=controls,joint_qpos_indices=qi,joint_qvel_indices=vi,base_body=base,root_qpos=qa,root_qvel=va)
atomic_json(out/'targets.json',future);atomic_json(out/'index.json',records)
summary={}
for name in sorted({name for row in projection for name in row}):
 summary[name]={key:dict(p50=float(np.median(values:=[row[name][key] for row in projection if name in row])),p95=float(np.quantile(values,.95)),maximum=float(max(values))) for key in ['position_m','rotation_deg','velocity_mps']}
atomic_json(out/'projection.json',dict(mujoco=mujoco.__version__,snapshots=len(records),mjcf=cfg['mjcf'],godot_spec=cfg['godot_spec'],model_sha256=hashlib.sha256(Path(cfg['mjcf']).read_bytes()).hexdigest(),spec_sha256=hashlib.sha256(Path(cfg['godot_spec']).read_bytes()).hexdigest(),root_velocity_max_abs_below=1e-8,summary=summary,per_snapshot=projection,scope='Kinematic projection only; finite constraint drift may prevent identical full-body initial states'))
print(dict(snapshots=len(records),summary=summary),flush=True)
