"""009 separately versioned Teacher q/qd/FF profile. Frozen04 adapter/source unchanged.
16 AK axes expose independent FF. XC roll and public fourbar jaw retain q-only.
Total requested torque, not separate PD/FF terms, passes original limit_torque.
"""
from pathlib import Path
import importlib.util
import numpy as np
P=Path(__file__).parent/'robot_source'
spec=importlib.util.spec_from_file_location('immutable_original04_drive',P/'runtime_adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad)
class TeacherMITDrive:
 def __init__(self,package):
  self.original=ad.FrozenPhysics(package)
  from check_refsafe_lock import lock
  lock(self.original.model);self.original.model.opt.timestep=.001;self.original.dt=.001;self.original.substeps=5;assert self.original.drive_dt==.005 and self.original.policy_dt==.02
  self.eligible=np.array([i not in (4,5) and self.original.joints[i]['actuator'].startswith('ak') for i in range(18)]);assert int(self.eligible.sum())==16;self.command_details={};self.mit_ticks=0
 def __getattr__(self,name):return getattr(self.original,name)
 def reset(self,*args,**kwargs):
  out=self.original.reset(*args,**kwargs);self.command_details={};self.mit_ticks=0;return out
 def step(self,action):
  self.command_details={'profile':'original_q_only','tau_ff18_Nm':np.zeros(18).tolist(),'qd_ref18_rad_s':np.zeros(18).tolist()};return self.original.step(action)
 def step_motor(self,q_ref18_rad,qd_ref18_rad_s,tau_ff18_Nm):
  import mujoco as mj
  e=self.original;qref=np.asarray(q_ref18_rad,float);qdref=np.asarray(qd_ref18_rad_s,float);ff=np.asarray(tau_ff18_Nm,float)
  assert qref.shape==qdref.shape==ff.shape==(18,) and np.isfinite(np.r_[qref,qdref,ff]).all()
  if np.any(abs(qdref[~self.eligible])>1e-12) or np.any(abs(ff[~self.eligible])>1e-12):raise ValueError('XC/fourbar independent qd/FF not admitted')
  qref=np.clip(qref,e.ranges[:,0],e.ranges[:,1]);qdref=np.clip(qdref,-e.slew_speed,e.slew_speed);ff=np.clip(ff,-e.cont,e.cont)
  before=float(e.data.time)
  for _ in range(e.ticks):
   old=e.target.copy();e.target+=np.clip(qref-e.target,-e.slew_speed*e.drive_dt,e.slew_speed*e.drive_dt);rate=(e.target-old)/e.drive_dt;qp=e.data.qpos[e.qidx];qv=e.data.qvel[e.vidx];pd=e.kp*(e.target-qp)+e.kd*(qdref-qv);total=pd+ff;actual=e.limit_torque(total,e.data.qvel[e.drive_vidx]);e.last_tau=actual.copy();e.data.ctrl[:]=actual
   self.command_details={'profile':'teacher_mit_q_qd_ff_v1','requested_q_ref18_rad':qref.tolist(),'actual_slewed_q_ref18_rad':e.target.tolist(),'actual_q_ref_derivative18_rad_s':rate.tolist(),'qd_ref18_rad_s':qdref.tolist(),'qd_ref_minus_q_slew_derivative18_rad_s':(qdref-rate).tolist(),'tau_ff18_Nm':ff.tolist(),'PD18_Nm':pd.tolist(),'combined_requested_tau18_Nm':total.tolist(),'limited_total_tau18_Nm':actual.tolist(),'eligible16_mask':self.eligible.tolist(),'no_independent_root_or_contact_forces':True,'FF_not_in_position_target':True}
   mj.mj_step(e.model,e.data,nstep=e.substeps);e.controller_updates+=1;self.mit_ticks+=1
   if not np.isfinite(e.data.qpos).all() or not np.isfinite(e.data.qvel).all():raise FloatingPointError('Nonfinite state')
   if any(w.number for w in e.data.warning):raise FloatingPointError('MuJoCo warning')
  e.policy_updates+=1
  if abs(float(e.data.time)-before-e.policy_dt)>1e-10:raise RuntimeError('Motor profile20ms timing mismatch')
  return e.state()
