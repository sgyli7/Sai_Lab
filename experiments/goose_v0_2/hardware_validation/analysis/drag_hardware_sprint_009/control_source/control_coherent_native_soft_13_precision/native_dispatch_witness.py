"""Existing native-state oracle algebra, now on truly DISPATCHED blendedFF.
One scratch forward, no integration or live writes. Released native contacts
without constraint rows are reported, not enumerated as other contacts.
"""
import numpy as np,mujoco as mj
from soft_phase_model import caps
def witness(env,qpos,qvel,total_tau18):
 m=env.model;d=mj.MjData(m);d.qpos[:]=qpos;d.qvel[:]=qvel;d.ctrl[:]=total_tau18;mj.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mj.mj_fullM(m,d,M);J=d.efc_J.reshape(d.nefc,m.nv);f=d.efc_force[:d.nefc];B=np.zeros((m.nv,18))
 for i in range(18):B[m.jnt_dofadr[m.actuator_trnid[i,0]],i]=m.actuator_gear[i,0]
 balance=M@d.qacc+d.qfrc_bias-d.qfrc_passive-B@d.actuator_force-J.T@f;grad=J@d.qacc+d.efc_R[:d.nefc]*f-d.efc_aref[:d.nefc];eq=np.flatnonzero(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY));cm=[];dry=[]
 for i,c in enumerate(d.contact):
  if c.efc_address<0:continue
  ids=np.arange(c.efc_address,c.efc_address+3);F=f[ids];g=grad[ids];mu=float(c.friction[0]);fn=float(F[0]);tn=float(np.linalg.norm(F[1:]))
  if fn<1e-8:mode='apex';vi=max(0.,mu*np.linalg.norm(g[1:])-g[0]);vi=max(vi,tn)
  elif tn<mu*fn-1e-7:mode='inside';vi=max(abs(g))
  else:mode='boundary';u=F[1:]/max(tn,1e-30);vi=max(abs(g[0]+mu*u@g[1:]),abs(np.array([-u[1],u[0]])@g[1:]),max(0.,u@g[1:]))
  cm.append(dict(original_contact_index=i,mode=mode,mode_residual=float(vi),circle_excess=float(max(0,tn-mu*fn))))
 for i in np.flatnonzero(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF)):
  cap=float(m.dof_frictionloss[d.efc_id[i]]);fi=float(f[i]);gi=float(grad[i]);dry.append(float(abs(gi) if abs(fi)<cap-1e-7 else max(0,gi*np.sign(fi))))
 cap=caps(env,d);tau=np.asarray(total_tau18);power=float(np.maximum(tau*d.qvel[env.drive_vidx],0).sum());source_diff=float(max(abs(env.limit_torque(tau,d.qvel[env.drive_vidx])-tau)));r=dict(physics_steps=0,actual_dispatched_blended_FF_not_raw_LP=True,force_balance_max=float(max(abs(balance))),equality_acceleration_max=float(max(abs(grad[eq]),default=0)),contact_KKT_max=float(max((c['mode_residual'] for c in cm),default=0)),contact_circle_excess_max=float(max((c['circle_excess'] for c in cm),default=0)),dry_friction_mode_max=float(max(dry,default=0)),source_total_difference=source_diff,source_cap_excess=float(max(abs(tau)-cap)),positive_power_W=power,total_tau18=tau.tolist(),qacc=d.qacc.tolist(),native_force=f.tolist(),contact_modes=cm,active_contact_indices=[i for i,c in enumerate(d.contact) if c.efc_address>=0],all_finite=bool(np.isfinite(np.r_[tau,d.qacc,f,balance,grad]).all()),source_legal=bool(source_diff<=1e-6 and max(abs(tau)-cap)<=1e-8 and power<=350+1e-8))
 return r
