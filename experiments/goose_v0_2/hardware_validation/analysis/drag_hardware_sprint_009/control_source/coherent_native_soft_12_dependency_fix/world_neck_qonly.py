"""Extracted original09 front5 IK only. No leg IK or mixed qdd construction.
Read actual robot pose in scratch; fixed observed capture rod anchor moves only
by planned backward progress, never current rod/body retreat feedback.
"""
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import complete_cycle_plan as plan
LOCAL=np.array([.13750001972253326,0,-.03849996489687674])
def command(env,actual_q,elapsed,alignment,entry_front_target):
 m=env.model;n=mj.MjData(m);n.qpos[:]=actual_q;h=m.body('head_roll').id;back=plan.travel(min(elapsed,plan.END))[0];world=alignment.rod_anchor+alignment.R.apply([-back,0,0])
 def res(z):
  n.qpos[env.qidx[:5]]=z;mj.mj_kinematics(m,n);H=n.xmat[h].reshape(3,3);ang=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[n.xpos[h]+H@LOCAL-world,.08*(ang[1]-.174532925),.08*ang[2]]
 so=least_squares(res,np.clip(np.asarray(actual_q)[env.qidx[:5]],env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),bounds=(env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),max_nfev=40,ftol=1e-9,gtol=1e-9,xtol=1e-9);qr=np.r_[so.x+alignment.front_bias,0.];z=float(plan.smooth(elapsed/.5));qr=(1-z)*np.asarray(entry_front_target)+(z*qr)
 return qr,dict(world_TCP_target=world.tolist(),neck_IK_error_m=float(np.linalg.norm(res(so.x))),front6_q_only_qd0_FF0=True,actual_root_world_neck_feedback_only_not_leg_qdd=True,fixed_rod_anchor=alignment.rod_anchor.tolist(),actual_rod_not_reanchored_each_tick=True,entry_C2_blend=z)
