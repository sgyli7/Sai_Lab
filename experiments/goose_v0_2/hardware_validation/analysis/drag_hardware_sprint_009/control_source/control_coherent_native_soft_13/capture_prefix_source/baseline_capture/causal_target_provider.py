"""009 2x2 causal settle. Only joint targets; independently selectable leg/neck channels."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
R=Path(__file__).parent
spec=importlib.util.spec_from_file_location('local_foot_ik',R/'foot_ik_reference.py');ik=importlib.util.module_from_spec(spec);spec.loader.exec_module(ik)
class AlternatingBackwardTeacher:
 def __init__(self,program,cold_rod_world,initial_q18):
  p=json.loads(Path(program).read_text());self.frame=p['frames'][0];self.initial=np.asarray(initial_q18).copy();self.cold=np.asarray(cold_rod_world).copy();self.tau=np.array(self.frame['static_tau18_Nm']);self.frames=[self.frame];self.duration=4.;self.warmup=4.;self.walk_time=0.;self.cycles=0;self.cycle_back=0.;self.cell=json.loads((R/'causal_design.json').read_text())['cells'][int(sys.argv[1])-1];self.last_neck=None;self.scratch=None
 def action(self,env,t,actual_rod_world=None):
  import mujoco as mj
  from scipy.optimize import least_squares
  from scipy.spatial.transform import Rotation
  m,d=env.model,env.data
  if self.scratch is None:
   self.scratch=mj.MjData(m);self.legs=mj.MjData(m);self.root0=d.qpos[:3].copy();self.anchors={s:d.geom_xpos[m.geom(s+'_actual_sole').id].copy() for s in ['left','right']};self.rotations={s:d.geom_xmat[m.geom(s+'_actual_sole').id].reshape(3,3).copy() for s in self.anchors}
  N={s:0. for s in self.anchors}
  for ci,c in enumerate(d.contact):
   for s in N:
    g=m.geom(s+'_actual_sole').id
    if g in (c.geom1,c.geom2) and m.geom('ground').id in (c.geom1,c.geom2):
     f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);world=c.frame.reshape(3,3).T@f[:3];N[s]+=max(0.,world[2] if c.geom2==g else -world[2])
  ramp=min(1.,max(0.,t/2.));ff=self.tau*ramp;nominal=self.initial.copy();actualback=float(self.root0[0]-d.qpos[0]);target=self.cold.copy()
  if self.cell['neck']=='new_body_led_preview5mm':target[0]-=max(0.,actualback)+.005
  scratch=self.scratch;scratch.qpos[:]=d.qpos;scratch.qvel[:]=d.qvel;idx=env.qidx[:5];bounds=env.ranges[:5];head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674])
  def residual(z):
   scratch.qpos[idx]=z;mj.mj_kinematics(m,scratch);H=scratch.xmat[head].reshape(3,3);e=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[scratch.xpos[head]+H@local-target,.08*(e[1]-.174532925),.08*e[2]]
  seed=d.qpos[idx] if self.last_neck is None else self.last_neck
  sol=least_squares(residual,np.clip(seed,bounds[:,0]+.05,bounds[:,1]-.05),bounds=(bounds[:,0]+.05,bounds[:,1]-.05),max_nfev=25,ftol=1e-7,xtol=1e-7,gtol=1e-7);self.last_neck=sol.x.copy();nominal[:5]=sol.x;errors={};weights={s:1. for s in N};vroot=self.root0.copy();angles=Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)).as_euler('xyz');pitchref=rollref=0.
  if self.cell['legs']=='new_settle_body_foot_feedback':
   rollref=float(np.clip(-.4*angles[0],-.10,.10));pitchref=float(np.clip(-.4*angles[1],-.04,.04));vroot=self.root0+[float(np.clip(-.3*(d.qpos[0]-self.root0[0]),-.01,.01)),0.,float(np.clip(.3*(self.root0[2]-d.qpos[2]),-.005,.005))]
   leg=self.legs;leg.qpos[:]=d.qpos;leg.qvel[:]=0.;leg.qpos[:3]=vroot;quat=Rotation.from_euler('xyz',[rollref,pitchref,0.]).as_quat();leg.qpos[3:7]=np.r_[quat[3],quat[:3]]
   for s in N:
    names=[s+'_'+n for n in ik.LEG_JOINT_SUFFIXES];ii=[env.names.index(n) for n in names];seed=self.initial[ii]+self.tau[ii]*ramp/env.kp[ii];z=ik.solve_foot_ik(m,leg,s,self.anchors[s],self.rotations[s],seed,max_nfev=25);nominal[ii]=z['q_target6_rad'];weights[s]=min(1.,N[s]/max(1.,sum(N.values())/2.));ff[ii]*=weights[s];errors[s]=z['equivalent_IK_error_m']
  q=nominal+ff/env.kp;q[5]=0.;q=np.clip(q,env.ranges[:,0],env.ranges[:,1]);meta={'causal_cell':self.cell,'phase':'causal_settle_no_gait','stance':['left','right'],'requested_joint_targets18':q.tolist(),'nominal_target18':nominal.tolist(),'actual_applied_position_bias_tau18_Nm':ff.tolist(),'bias_implementation':'tau/kp inserted into slew-limited position target; NOT independent motor feedforward','cold_rod_anchor_world':self.cold.tolist(),'tcp_target_world':target.tolist(),'actual_body_back_m':actualback,'actual_rod_monitor_only':None if actual_rod_world is None else np.asarray(actual_rod_world).tolist(),'scratch_IK_error_equivalent_m':float(np.linalg.norm(residual(sol.x))),'foot_IK_errors':errors,'measured_normal_N':N,'leg_bias_weights':weights,'virtual_body_reference':vroot.tolist(),'virtual_roll_pitch':[rollref,pitchref],'actual_body_roll_pitch':angles[:2].tolist(),'actual_COM_world':d.subtree_com[m.body('torso').id].tolist(),'prepositioned_true_physical_grip_not_active_capture':True,'progress_m':0.,'cycle':0,'u':0.}
  return (q-env.neutral)/env.scale,meta
