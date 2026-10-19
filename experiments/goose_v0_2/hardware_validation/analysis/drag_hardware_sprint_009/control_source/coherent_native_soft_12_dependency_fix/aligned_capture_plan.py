"""One observed horizontal rigid-frame alignment, not per-tick target chasing.
Whole plan pose is transformed; joint q/v/a and all timing/stride stay intact.
Actual capture world is never modified. Floor Z remains0.
"""
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
class AlignedCapturePlan:
 def __init__(self,base,capture_context):
  self.base=base;self.model=base.model;self.time=base.time;self.qidx=base.qidx;self.bias18=base.bias18
  actual=np.array(capture_context['actual_qpos']);a=mj.MjData(self.model);a.qpos[:]=actual;mj.mj_forward(self.model,a);b=mj.MjData(self.model);b.qpos[:]=base.sample(0.)['qpos'];mj.mj_forward(self.model,b)
  sides=['right','left'];src=np.array([b.geom_xpos[self.model.geom(s+'_actual_sole').id] for s in sides]);dst=np.array([a.geom_xpos[self.model.geom(s+'_actual_sole').id] for s in sides]);u=src[1,:2]-src[0,:2];v=dst[1,:2]-dst[0,:2];yaw=float(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v));self.R=Rotation.from_euler('z',yaw);self.T=np.r_[dst[:,:2].mean(0)-self.R.apply(src)[:,:2].mean(0),0.];fit=self.R.apply(src)+self.T
  self.rod_anchor=a.geom_xpos[self.model.geom('cart_bite_bar').id].copy();self.front_bias=self.bias18[:5].copy();self.report=dict(one_time_alignment=True,yaw_rad=yaw,translation_world_m=self.T.tolist(),original_sole_centers_R_L=src.tolist(),observed_capture_sole_centers_R_L=dst.tolist(),aligned_nominal_sole_centers_R_L=fit.tolist(),fit_error_R_L_m=(fit-dst).tolist(),fit_error_norm_R_L_m=np.linalg.norm(fit-dst,axis=1).tolist(),actual_root=actual[:7].tolist(),aligned_nominal_root=self.transform_q(base.sample(0.)['qpos'])[:7].tolist(),actual_fixed_rod_anchor=self.rod_anchor.tolist(),old_cold_rod_anchor=capture_context['cold_rod_fixed_anchor'],cumulative_slip_must_not_reset=True,no_actual_state_write=True)
 def _vidx(self):return self.base._vidx()
 def transform_q(self,q):
  q=q.copy()
  for j in range(self.model.njnt):
   if int(self.model.jnt_type[j])!=int(mj.mjtJoint.mjJNT_FREE):continue
   k=int(self.model.jnt_qposadr[j]);q[k:k+3]=self.R.apply(q[k:k+3])+self.T;quat=(self.R*Rotation.from_quat(q[k+3:k+7][[1,2,3,0]])).as_quat();q[k+3:k+7]=quat[[3,0,1,2]]
  return q
 def sample(self,t):
  x=self.base.sample(t);x['qpos']=self.transform_q(x['qpos'])
  for j in range(self.model.njnt):
   if int(self.model.jnt_type[j])==int(mj.mjtJoint.mjJNT_FREE):
    k=int(self.model.jnt_dofadr[j]);x['qvel'][k:k+3]=self.R.apply(x['qvel'][k:k+3]);x['qacc'][k:k+3]=self.R.apply(x['qacc'][k:k+3])
  x['one_time_capture_frame_aligned']=True;return x
