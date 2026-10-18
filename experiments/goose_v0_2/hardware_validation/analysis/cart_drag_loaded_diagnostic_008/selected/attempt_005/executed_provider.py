"""Bounded measured-support Teacher. Scratch IK only, joint targets via immutable adapter."""
from pathlib import Path
import json,importlib.util
import numpy as np
R=Path(__file__).parent
sp=importlib.util.spec_from_file_location('old_gait_provider',R/'gait_target_provider.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
sp=importlib.util.spec_from_file_location('foot_ik',R/'narrow_stance_readiness/foot_ik_reference.py');ik=importlib.util.module_from_spec(sp);sp.loader.exec_module(ik)
class AlternatingBackwardTeacher(old.AlternatingBackwardTeacher):
 def __init__(self,program,cold_rod_world,initial_q18):
  super().__init__(program,cold_rod_world,initial_q18);self.duration=75.;self.walk_time=70.;self.warmup=2.;self.state='settle';self.enter=0.;self.stepindex=0;self.unload_s=0.;self.last_t=0.;self.finished=False;self.previous_meta={}
 def action(self,env,t,actual_rod_world=None):
  import mujoco as mj
  from scipy.spatial.transform import Rotation
  if not hasattr(self,'root0'):
   self.root0=env.data.qpos[:3].copy();self.anchors={s:env.data.geom_xpos[env.model.geom(s+'_actual_sole').id].copy() for s in ['left','right']};self.footrots={s:env.data.geom_xmat[env.model.geom(s+'_actual_sole').id].reshape(3,3).copy() for s in self.anchors};self.legscratch=mj.MjData(env.model)
  dt=max(0.,t-self.last_t);self.last_t=t;d,m=env.data,env.model;Ns={s:0. for s in self.anchors}
  for ci,c in enumerate(d.contact):
   for s in Ns:
    if m.geom(s+'_actual_sole').id in (c.geom1,c.geom2) and m.geom('ground').id in (c.geom1,c.geom2):
     force=np.zeros(6);mj.mj_contactForce(m,d,ci,force);fw=c.frame.reshape(3,3).T@force[:3];Ns[s]+=max(0.,fw[2] if c.geom2==m.geom(s+'_actual_sole').id else -fw[2])
  stance='left' if self.stepindex%2==0 else 'right';swing='right' if stance=='left' else 'left';sg=1 if stance=='left' else -1;elapsed=t-self.enter;COM=d.subtree_com[m.body('torso').id].copy();angles=Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)).as_euler('xyz');abort=None
  if self.state=='settle' and elapsed>=2.:self.state='transfer';self.enter=t;elapsed=0.
  if self.state=='transfer':
   # Actual support progress gates phase. Never advance merely because a clock expires.
   inside=sg*(COM[1]-self.anchors[stance][1])>=-.025
   self.unload_s=self.unload_s+dt if Ns[swing]<15. and Ns[stance]>80. and inside else 0.
   if self.unload_s>=.1:self.state='swing';self.enter=t;elapsed=0.;self.swing_start=self.anchors[swing].copy()
   elif elapsed>=3.:abort='feedback_transfer_timeout_no_actual_unload'
  elif self.state=='swing' and elapsed>=1.5:
   self.state='landing';self.enter=t;elapsed=0.
  elif self.state=='landing':
   if Ns[swing]>15. and elapsed>.1:
    self.anchors[swing]=d.geom_xpos[m.geom(swing+'_actual_sole').id].copy();self.stepindex+=1;self.state='transfer';self.enter=t;elapsed=0.;self.unload_s=0.
   elif elapsed>=1.:abort='feedback_landing_timeout'
  actualback=float(self.root0[0]-d.qpos[0]);progress=max(0.,actualback)
  if progress>=.55 and self.stepindex>=8 and self.state=='transfer':self.state='stop_hold';self.enter=t;elapsed=0.
  if self.state=='stop_hold' and elapsed>=3.:abort='feedback_task_complete_hold3s'
  # Bounded root reference is converted only into original leg joint targets.
  a=float(np.clip(elapsed/1.5,0,1));a=a*a*(3-2*a)
  side_goal=0. if self.state in ('settle','stop_hold') else sg*.060*a if self.state=='transfer' else sg*.060
  com_goal=self.anchors[stance][1]-sg*.015
  ycorr=0. if self.state in ('settle','stop_hold') else float(np.clip(.5*(com_goal-COM[1]),-.015,.015))
  roll_goal=0. if self.state in ('settle','stop_hold') else -sg*.08*(a if self.state=='transfer' else 1.)
  roll_ref=float(np.clip(roll_goal+.4*(roll_goal-angles[0]),-.10,.10));pitch_ref=float(np.clip(-.4*angles[1],-.04,.04))
  requested_back=min(.56,.020*(self.stepindex+ (a if self.state=='transfer' else 1.))) if self.state!='settle' else 0.
  vroot=self.root0+[-requested_back+float(np.clip(.3*(-requested_back- (d.qpos[0]-self.root0[0])),-.01,.01)),float(np.clip(side_goal+ycorr,-.075,.075)),float(np.clip(.3*(self.root0[2]-d.qpos[2]),-.005,.005))]
  # Reuse old neck IK with actual-body anchoring, not time-driven .56m neck pull.
  self.phase=lambda tt:{'u':0.,'cycle':self.stepindex//2,'progress_m':min(.56,progress+.005),'phase':self.state,'stance':[stance] if self.state in ('swing','landing') else ['left','right']}
  action,meta=super().action(env,t,actual_rod_world);q=np.array(meta['requested_joint_targets18']);tau=np.array(self.frames[0]['static_tau18_Nm']);scratch=self.legscratch;scratch.qpos[:]=d.qpos;scratch.qvel[:]=0;scratch.qpos[:3]=vroot;quat=Rotation.from_euler('xyz',[roll_ref,pitch_ref,0.]).as_quat();scratch.qpos[3:7]=np.r_[quat[3],quat[:3]];errors={}
  for s in self.anchors:
   target=self.anchors[s].copy()
   if s==swing and self.state in ('swing','landing'):
    u=float(np.clip(elapsed/1.5,0,1)) if self.state=='swing' else 1.;target=self.swing_start+[-.04*u,0,.015*np.sin(np.pi*u)]
   names=[s+'_'+n for n in ik.LEG_JOINT_SUFFIXES];ix=[env.names.index(n) for n in names];sol=ik.solve_foot_ik(m,scratch,s,target,self.footrots[s],q[ix],max_nfev=25);weight=min(1.,Ns[s]/max(1.,sum(Ns.values())/2.));q[ix]=np.array(sol['q_target6_rad'])+tau[ix]*weight*min(1.,t/2.)/env.kp[ix];errors[s]=sol['equivalent_IK_error_m']
  q=np.clip(q,env.ranges[:,0],env.ranges[:,1]);meta.update(requested_joint_targets18=q.tolist(),controller_variant='measured_support_feedback_v1',measured_normal_N=Ns,actual_COM_world=COM.tolist(),actual_body_roll_pitch=angles[:2].tolist(),virtual_body_reference=vroot.tolist(),virtual_roll_pitch=[roll_ref,pitch_ref],actual_body_back_m=actualback,body_based_TCP_preview_m=.005,measured_unload_streak_s=self.unload_s,actual_support_steps_completed=self.stepindex,foot_IK_errors=errors,teacher_abort_reason=abort,static_bias_policy='cold_DS_tau scaled down with actual foot load; not single-support feedforward claim')
  return (q-env.neutral)/env.scale,meta
