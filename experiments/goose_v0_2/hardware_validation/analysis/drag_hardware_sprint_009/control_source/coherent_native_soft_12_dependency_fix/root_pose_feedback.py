"""Root task only: native tangent pose error; freejoint angular velocity body-local.
No integration/state write. Does not replace original 200Hz motor PD.
"""
import numpy as np
import mujoco as mj

def root_acceleration_goal(model,actual_q,actual_v,desired_q,desired_v,desired_a):
 error=np.zeros(model.nv);mj.mj_differentiatePos(model,error,1.,actual_q,desired_q)
 Ra=np.zeros(9);Rd=np.zeros(9);mj.mju_quat2Mat(Ra,actual_q[3:7]);mj.mju_quat2Mat(Rd,desired_q[3:7]);T=Ra.reshape(3,3).T@Rd.reshape(3,3)
 vd=np.asarray(desired_v[:6]).copy();ad=np.asarray(desired_a[:6]).copy();vd[3:6]=T@vd[3:6];ad[3:6]=T@ad[3:6]-np.cross(actual_v[3:6],vd[3:6])
 kp=np.array([25.,25.,25.,64.,64.,64.]);kd=np.array([10.,10.,10.,16.,16.,16.]);recovery=kp*error[:6]+kd*(vd-actual_v[:6]);goal=ad+recovery
 return dict(goal=goal.tolist(),native_pose_error=error[:6].tolist(),velocity_error=(vd-actual_v[:6]).tolist(),transported_curve_acceleration=ad.tolist(),recovery=recovery.tolist(),Kp=kp.tolist(),Kd=kd.tolist(),angular_frame='actual freejoint local tangent; desired angular velocity transported Ractual.T Rdesired',no_native_joint_PD_root_acceleration_added=True)
