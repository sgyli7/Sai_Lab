"""Bounded .2s C2 entry within, never before, the declared4.90625s clock.
Reference-only scratch geometry; no runtime state/target/clock reset. Needs a
separate inverse/contact certificate for this actual capture context.
"""
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

def smooth(t):
 u=np.clip(t,0.,1.);return 10*u**3-15*u**4+6*u**5
class CapturePlanEntry:
 def __init__(self,plan,capture_context,foot_slip_R_L,duration=.5):
  if abs(duration-.5)>1e-12:raise ValueError('only declared .5s entry')
  self.plan=plan;self.model=plan.model;self.context=capture_context;self.entry=np.array(capture_context['actual_qpos']);self.adapter_target=np.array(capture_context['original_env_target']);self.held_qd=np.array(capture_context['held_motor_command']['qd_ref18_rad_s']);self.held_FF=np.array(capture_context['held_motor_command']['tau_ff18_Nm']);self.clock=float(capture_context['clock_s']);self.duration=duration;self.foot_slip=np.asarray(foot_slip_R_L,float);self.entry_bias=self.adapter_target-self.entry[plan.qidx];self.root_entry_velocity=np.zeros(self.model.nv);self.root_entry_velocity[:6]=np.array(capture_context['actual_qvel'])[:6]
 def pose(self,t):
  r=self.plan.sample(t);q=r['qpos'].copy();s=smooth(t/self.duration);initial=self.entry.copy();mj.mj_integratePos(self.model,initial,self.root_entry_velocity,float(t))
  if t>=self.duration:return q
  for j in range(self.model.njnt):
   qi=int(self.model.jnt_qposadr[j]);typ=int(self.model.jnt_type[j])
   if typ==int(mj.mjtJoint.mjJNT_FREE):
    q[qi:qi+3]=(1-s)*initial[qi:qi+3]+s*q[qi:qi+3];R0=Rotation.from_quat(initial[qi+3:qi+7][[1,2,3,0]]);Rp=Rotation.from_quat(q[qi+3:qi+7][[1,2,3,0]]);quat=(R0*Rotation.from_rotvec(s*(R0.inv()*Rp).as_rotvec())).as_quat();q[qi+3:qi+7]=quat[[3,0,1,2]]
   else:q[qi]=(1-s)*self.entry[qi]+s*q[qi]
  return q
 def sample(self,t,h=1e-5):
  q=self.pose(t)
  if t<=1e-12:v=self.root_entry_velocity.copy();a=np.zeros(self.model.nv)
  elif t>=self.duration+2*h:
   r=self.plan.sample(t);r['qd_ref18']=r['qvel'][self.plan._vidx()].copy();r['qd_ref18'][:6]=0.;return r
  else:
   lo=max(0.,t-h);hi=min(self.plan.time[-1],t+h);vm=np.zeros(self.model.nv);vp=np.zeros(self.model.nv);mj.mj_differentiatePos(self.model,vm,t-lo,self.pose(lo),q);mj.mj_differentiatePos(self.model,vp,hi-t,q,self.pose(hi));v=.5*(vm+vp);a=2*(vp-vm)/(hi-lo)
  s=smooth(t/self.duration);bias=(1-s)*self.entry_bias+s*self.plan.bias18;qr=q[self.plan.qidx]+bias
  # qd remains diagnostic until computed from SAME qref+bias curve. Front6
  # original q-only qd0; actual adapter clips qd and slews target normally.
  if t<=1e-12:qd=self.held_qd.copy()
  else:
   lo=max(0.,t-h);hi=min(self.plan.time[-1],t+h)
   def cmd(tt):return self.pose(tt)[self.plan.qidx]+(1-smooth(tt/self.duration))*self.entry_bias+smooth(tt/self.duration)*self.plan.bias18
   qd=(cmd(hi)-cmd(lo))/(hi-lo);qd[:6]=0.
  return dict(time=t,qpos=q,qvel=v,qacc=a,qref18=qr,qd_ref18=qd,entry_phase=float(s),entry_end_s=self.duration,backward_order_clock_start=self.clock,entry_clock_counts_within_plan=True,carry_foot_slip_R_L=self.foot_slip.copy(),reference_geometry_only_not_state_write=True,certificate_required_before_actual=True)

 def command(self,t):
  r=self.sample(t);return r['qref18'],r['qd_ref18'],r

 def certificate_sampler_sha256(self,t):
  import hashlib,inspect
  from pathlib import Path
  obj=self if t<self.duration else self.plan
  return hashlib.sha256(Path(inspect.getfile(type(obj))).read_bytes()).hexdigest()
 def join_FF(self,t,inverse_FF18):
  s=smooth(t/self.duration);return (1-s)*self.held_FF+s*np.asarray(inverse_FF18,float)
