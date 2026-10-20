from capture_reference import tcp as capture_tcp,jaw as capture_jaw
"""009 whole-body position/impedance Teacher candidate; no dynamics integration/state correction.
Original200Hz Kp/Kd retained. Explicit leg FF contains cold support torque once plus bounded model gravity residual.
Plant to real cold bottom normal/yaw; actual support gate before 1.5Hz stepping. Pending Owner physics admission.
"""
from pathlib import Path
import plan
import importlib.util,json,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
R=Path(__file__).parent
sp=importlib.util.spec_from_file_location('old_stable_teacher',R/'causal_target_provider.py');legacy=importlib.util.module_from_spec(sp);sp.loader.exec_module(legacy);ik=legacy.ik
class AlternatingBackwardTeacher(legacy.AlternatingBackwardTeacher):
 def __init__(self,*args):
  super().__init__(*args);self.duration=9.;self.warmup=0.;self.walk_time=12.;self.boot=False;self.plant_integral={'right':np.zeros(3),'left':np.zeros(3)};self.last_t=None;self.active_swing=None;self.air={'right':0.,'left':0.};self.air_seen={'right':False,'left':False};self.landing_events=[];self.completed=[];self.hold_start=None;self.back_start=None
 def action(self,env,t,actual_rod_world=None):
  m,d=env.model,env.data
  if not self.boot:
   self.coldq=d.qpos.copy();self.root0=d.qpos[:3].copy();self.feet={s:m.geom(s+'_actual_sole').id for s in ['right','left']};self.anchors={s:d.geom_xpos[g].copy() for s,g in self.feet.items()};self.initial_anchors={s:p.copy() for s,p in self.anchors.items()};self.coldR={s:d.geom_xmat[g].reshape(3,3).copy() for s,g in self.feet.items()};self.native_normal={s:q.T@np.array([0.,0.,1.]) for s,q in self.coldR.items()};self.nominal=self.initial.copy();self.scratch=mj.MjData(m);self.gravity_scratch=mj.MjData(m);self.gravity_scratch.qpos[:]=self.coldq;self.gravity_scratch.qvel[:]=0.;mj.mj_forward(m,self.gravity_scratch);self.g0=self.gravity_scratch.qfrc_bias.copy();self.boot=True
  action,base=super().action(env,t,actual_rod_world)
  if not hasattr(self,'com0'):self.motion_start=float('inf');self.com0=d.subtree_com[m.body('torso').id].copy();self.previous_nominal=self.initial.copy();self.phase_wait_start=None
  u0=float(np.clip(t/.2,0,1));entry=10*u0**3-15*u0**4+6*u0**5
  if not hasattr(self,'COM_kinematic_calibration'):
   def sample(dy,dr):
    q=mj.MjData(m);q.qpos[:]=self.coldq;q.qvel[:]=0.;q.qpos[1]+=dy;quat=Rotation.from_euler('x',dr).as_quat();q.qpos[3:7]=np.r_[quat[3],quat[:3]]
    for side in self.feet:
     ii=[env.names.index(side+'_'+n) for n in ik.LEG_JOINT_SUFFIXES];ik.solve_foot_ik(m,q,side,self.anchors[side],self.coldR[side],self.initial[ii],max_nfev=30)
    
    from scipy.optimize import least_squares
    hh=m.body('head_roll').id
    def nr(z):
     q.qpos[env.qidx[:5]]=z;mj.mj_kinematics(m,q);HH=q.xmat[hh].reshape(3,3);ee=Rotation.from_matrix(HH).as_euler('ZYX');return np.r_[q.xpos[hh]+HH@np.array([.13750001972253326,0,-.03849996489687674])-self.cold,.08*(ee[1]-.174532925),.08*ee[2]]
    zz=least_squares(nr,q.qpos[env.qidx[:5]],bounds=(env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),max_nfev=30,ftol=1e-8,gtol=1e-8,xtol=1e-8);nr(zz.x)
    mj.mj_forward(m,q);return float(q.subtree_com[m.body('torso').id,1])
   ky=(sample(.001,0)-sample(-.001,0))/.002;kr=(sample(0,.001)-sample(0,-.001))/.002;self.COM_kinematic_calibration={'rootY_to_totalCOMY':ky,'rootRoll_to_totalCOMY_m_rad':kr,'physics_steps':0,'method':'four finite FK+footIK samples; mass-weighted compiled whole robot'}
   if abs(ky)<.2:return action,dict(base,teacher_abort_reason='wholebody_COM_kinematic_calibration_degenerate',COM_kinematic_calibration=self.COM_kinematic_calibration)

  dt=.02 if self.last_t is None else max(0.,t-self.last_t);self.last_t=t;N={s:0. for s in self.feet};clear={};grounds={m.geom(n).id for n in json.loads((R/'foot_ground_names.json').read_text())}
  for ci,c in enumerate(d.contact):
   for s,g in self.feet.items():
    if any(x in (c.geom1,c.geom2) for x in grounds) and g in (c.geom1,c.geom2):
     f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);F=c.frame.reshape(3,3).T@f[:3];N[s]+=max(0.,F[2]*(1 if c.geom2==g else -1))
  for s,g in self.feet.items():
   mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];clear[s]=float(world[:,2].min());self.air[s]=self.air[s]+dt if N[s]<=2 and clear[s]>.001 else 0.
   if self.air[s]>=.05:self.air_seen[s]=True
  jawN=np.zeros(2)
  for ci,c in enumerate(d.contact):
   names=[m.geom(c.geom1).name,m.geom(c.geom2).name]
   for k,n in enumerate(['upper_bill_envelope','lower_bill_envelope']):
    if n in names and 'cart_bite_bar' in names:
     f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);jawN[k]+=max(0.,f[0])
  if self.hold_start is None:
   plant=getattr(env,'actual_DS_gate',{});angles0=Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)).as_euler('xyz');co=d.subtree_com[m.body('torso').id]
   ready=t>=.8 and min(N.values())>20 and min(jawN)>=5 and np.linalg.norm(co[:2]-self.com0[:2])<=.005 and abs(co[2]-self.com0[2])<=.005 and np.linalg.norm(angles0[:2])<=np.deg2rad(5) and plant.get('continuous_DS_conditions_s',0.)>=.2 and max(getattr(env,'cumulative_foot_slip',[0.,0.]))<=.005
   if ready:self.hold_start=t;self.motion_start=float("inf");env.loaded_hold_metrics={}
   elif t>=3.:return action,dict(base,teacher_abort_reason='actual_plant_establishment_timeout3s_no_hold_started',actual_plant_gate=plant,actual_jaw_normal_N=jawN.tolist())
  if t>=self.motion_start and not hasattr(self,'neck_lock'):
   hm=getattr(env,'loaded_hold_metrics',{});plant=getattr(env,'actual_plant_gate',{});self.hold_certificate=hm.copy()
   if hm.get('samples',0)<4995 or hm.get('max_COM_height_error_m',1.)>.005 or hm.get('max_COM_XY_error_m',1.)>.005 or hm.get('max_tilt_deg',30.)>5. or hm.get('min_jaw_normal_N',0.)<5. or hm.get('min_foot_normal_N',0.)<20. or max(getattr(env,'cumulative_foot_slip',[1.,1.]))>.005:return action,dict(base,teacher_abort_reason='loaded_hold5s_or_original5mm_qualification_not_passed',loaded_hold5s_metrics=hm)
   self.neck_lock=np.array(base['requested_joint_targets18'][:6]);self.phase_wait_start=t;head=m.body('head_roll').id;tcp=d.xpos[head]+d.xmat[head].reshape(3,3)@np.array([.13750001972253326,0,-.03849996489687674]);self.grasp_local=d.xmat[m.body('torso').id].reshape(3,3).T@(tcp-d.xpos[m.body('torso').id])
  elapsed=max(0.,t-self.motion_start);cy,vy,ay=plan.lateral(elapsed);distance,speed,ax=plan.travel(elapsed);rollnom=-plan.RATIO*cy;duration=plan.SWING;startbridge=float(plan.STARTS[0]);phi=plan.phase(elapsed);amp=1.
  if t<self.motion_start:cy=vy=ay=rollnom=distance=speed=ax=0.
  comref=self.com0+[-distance,cy,0.];Tx=plan.RAMP
  swing=self.active_swing
  if self.active_swing and self.air_seen[self.active_swing] and N[self.active_swing]>15:
   s=self.active_swing;back=float(self.actual_lift0[0]-d.geom_xpos[self.feet[s],0]);self.landing_events.append({'side':s,'time_s':t,'air50ms':True,'actual_back_m':back});self.anchors[s]=d.geom_xpos[self.feet[s]].copy();self.completed.append(s);self.active_swing=None;self.air_seen[s]=False;self.phase_wait_start=t
   if back<.01:return action,dict(base,teacher_abort_reason='actual_landing_not_backward10mm',actual_landing_events=self.landing_events)
  # Owner-approved removal of the artificial lowN-before-lift gate. Only the predeclared LEFT target is requested, through realPD; real support/hardstops remain.
  if not self.active_swing and not self.completed and t>=self.motion_start+plan.STARTS[0]:
   self.active_swing='left';self.lift_time=self.motion_start+plan.STARTS[0];self.actual_lift0=d.geom_xpos[self.feet['left']].copy();self.lift0=self.anchors['left'].copy();self.lift_end=plan.feet_targets(float(plan.ENDS[0]),self.initial_anchors)[0]['left'].copy();self.phase_wait_start=t;self.back_start=self.lift_time
  for observed in getattr(env,'actual_foot_landing_events',[]):
   if observed['side']=='left' and observed['time_s']>=self.motion_start and observed['actual_back_from_cold_m']>=.01:
    self.actual_first_step=observed;self.completed=['left'];self.active_swing=None
  swing=self.active_swing
  if swing and self.phase_wait_start is not None and t-self.phase_wait_start>3.:return action,dict(base,teacher_abort_reason='actual_support_transition_timeout3s',actual_landing_events=self.landing_events)
  # Body target is an IK reference only. True root/feet are never written or forced.
  blend=entry;actualCOM=d.subtree_com[m.body('torso').id];feedback=np.clip(.5*(comref-actualCOM),[-.015,-.015,-.005],[.015,.015,.005]);angles=Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)).as_euler('xyz');roll=rollnom+float(np.clip(-.4*(angles[0]-rollnom),-.04,.04));pitch=float(np.clip(-.4*angles[1],-.04,.04));rootY=(cy-self.COM_kinematic_calibration['rootRoll_to_totalCOMY_m_rad']*rollnom)/self.COM_kinematic_calibration['rootY_to_totalCOMY'];feedback[1]=np.clip(feedback[1]/self.COM_kinematic_calibration['rootY_to_totalCOMY'],-.015,.015);virtualroot=self.root0+[-distance,rootY,0.]+blend*feedback
  leg=self.scratch;leg.qpos[:]=d.qpos;leg.qvel[:]=0.;leg.qpos[:3]=virtualroot;qq=Rotation.from_euler('xyz',[roll*blend,pitch*blend,0.]).as_quat();leg.qpos[3:7]=np.r_[qq[3],qq[:3]];nominal=self.initial.copy();errs={};foot_normals={};actualroot_left_solution=None;actualroot_left_blend=0.
  for side,g in self.feet.items():
   actualnormal=d.geom_xmat[g].reshape(3,3)@self.native_normal[side];error=np.cross(actualnormal,[0.,0.,1.]);error[2]=0.;foot_normals[side]=actualnormal.tolist()
   if N[side]>20:self.plant_integral[side]=np.clip(self.plant_integral[side]+2.*dt*error,-.03,.03)
   correction=np.clip(self.plant_integral[side]+.5*error,-.03,.03)*blend;rotation=Rotation.from_rotvec(correction).as_matrix()@self.coldR[side];point=self.anchors[side].copy()
   if side=='left' and t>=self.motion_start+plan.STARTS[0]:
    z=float(np.clip((t-self.lift_time)/duration,0,1));sw=10*z**3-15*z**4+6*z**5;point=self.lift0+(self.lift_end-self.lift0)*sw;point[2]+=.010*64*z**3*(1-z)**3
   names=[side+'_'+n for n in ik.LEG_JOINT_SUFFIXES];ii=[env.names.index(n) for n in names];sol=ik.solve_foot_ik(m,leg,side,point,rotation,self.previous_nominal[ii],max_nfev=30);nominal[ii]=np.array(sol['q_target6_rad']);errs[side]=sol['equivalent_IK_error_m']
   if side=='left' and t>=self.motion_start+plan.STARTS[0]:
    # One authorized change only: world swing target solved in ACTUAL root,
    # while right stance/body correction and old virtual gravity scratch survive.
    if not hasattr(self,'actualroot_swing_scratch'):self.actualroot_swing_scratch=mj.MjData(m)
    ar=self.actualroot_swing_scratch;ar.qpos[:]=d.qpos;ar.qvel[:]=0.;ar_sol=ik.solve_foot_ik(m,ar,side,point,rotation,nominal[ii],max_nfev=40);actualroot_left_solution=np.array(ar_sol['q_target6_rad']);errs['left_actualroot']=ar_sol['equivalent_IK_error_m'];u_ar=float(np.clip((t-self.lift_time)/(.25*duration),0,1));actualroot_left_blend=10*u_ar**3-15*u_ar**4+6*u_ar**5

  nominal=self.initial+blend*(nominal-self.initial);self.previous_nominal=nominal.copy();
  if actualroot_left_solution is not None:nominal[12:18]+=actualroot_left_blend*(actualroot_left_solution-nominal[12:18])
  qref=self.initial+entry*(np.array(base['nominal_target18'])-self.initial);qref[:5]+=entry*self.tau[:5]/env.kp[:5];qref[5]=capture_jaw(t);qref[6:]=nominal[6:];qdref=np.zeros(18)
  # World reference from the declared PLANNED body progress. Actual body pose enters IK, never the target anchor.
  if not hasattr(self,'neckscratch'):self.neckscratch=mj.MjData(m);self.last_world_neck=self.initial[:5].copy()
  neck=self.neckscratch;neck.qpos[:]=d.qpos;neck.qvel[:]=d.qvel;head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674]);world_tcp=capture_tcp(self.cold,t)
  from scipy.optimize import least_squares
  def neckres(z):
   neck.qpos[env.qidx[:5]]=z;mj.mj_kinematics(m,neck);H=neck.xmat[head].reshape(3,3);ang=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[neck.xpos[head]+H@local-world_tcp,.08*(ang[1]-.174532925),.08*ang[2]]
  solneck=least_squares(neckres,np.clip(self.last_world_neck,env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),bounds=(env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),max_nfev=30,ftol=1e-8,xtol=1e-8,gtol=1e-8);self.last_world_neck=solneck.x.copy();qref[:5]=solneck.x+entry*self.tau[:5]/env.kp[:5];qref[5]=capture_jaw(t)
  if elapsed>0 and hasattr(self,'previous_sent_qref'):qdref[6:]=(qref[6:]-self.previous_sent_qref[6:])/max(dt,1e-6)
  self.previous_sent_qref=qref.copy()
  # Cold full-support inverse-dynamics anchor + bounded model gravity difference; not a force-feasibility proof.
  mj.mj_forward(m,leg);g=leg.qfrc_bias;ff=np.zeros(18);delta=np.zeros(18)
  for j in range(6,18):
   ids=[env.vidx[j]] if env.vidx[j]==env.drive_vidx[j] else [env.vidx[j],env.drive_vidx[j]];delta[j]=np.clip(sum(g[k]-self.g0[k] for k in ids),-.5,.5);ff[j]=entry*(self.tau[j]+delta[j])
  abort=None
  if elapsed>0 and np.max(abs(qdref[6:])-env.slew_speed[6:])>1e-8:abort='IK_reference_velocity_over100percent_source_slew_not_actual_speed_clamp'
  if np.min(np.minimum(qref[6:]-env.ranges[6:,0],env.ranges[6:,1]-qref[6:]))<.02:abort='IK_public_ROM_prediction_margin_under20mrad'
  if max(errs.values())>.0005:abort='IK_reference_error_over0p5mm'
  if np.max(abs(ff)-env.cont)>1e-8:abort='Owner_software_FF_component_guard_not_hardware_infeasibility'
  targetTCP=world_tcp.copy()
  slewed=env.target+np.clip(qref-env.target,-env.slew_speed*.005,env.slew_speed*.005);first_tau=env.limit_torque(env.kp*(slewed-d.qpos[env.qidx])+env.kd*(qdref-d.qvel[env.vidx])+ff,d.qvel[env.drive_vidx])
  meta=dict(base,actualroot_swing_mapping_only=True,actualroot_swing_C2_blend_s=.25*duration,actualroot_swing_blend=actualroot_left_blend,original_virtual_gravity_scratch_preserved=True,phase='actual_plant_establish_from_t0' if self.hold_start is None else 'loaded_stable_grip_hold5s' if t<self.motion_start else 'wholebody_actual_support_IK_phase',stance=['left','right'] if swing is None else ['left' if swing=='right' else 'right'],requested_joint_targets18=qref.tolist(),tcp_target_world=targetTCP.tolist(),force_prediction={'total_tau18_Nm':first_tau.tolist(),'scope':'only first5ms PD+FF algebra; later200Hz PD correction expected, not held total torque'},COM_reference_world=comref.tolist(),COM_velocity_reference_world=[-speed,vy,0.],COM_acceleration_reference_world=[-ax,ay,0.],actual_Fz_N=N,actual_sole_clearance_m=clear,native_sole_normals_world=foot_normals,cold_flat_geom_R_targets={s:v.tolist() for s,v in self.coldR.items()},plant_integral_world_rotation={s:v.tolist() for s,v in self.plant_integral.items()},virtual_body_reference=virtualroot.tolist(),motion_start_time_s=self.motion_start,actual_landing_events=self.landing_events,planned_swing=swing,active_swing=self.active_swing,IK_errors_m=errs,original200Hz_impedance_retained=True,body_back_start_actual_unload_s=self.back_start,body_speed_ramp_s=1.40625,body_peak_acceleration_m_s2=.2,swing_C2_duration_s=duration,planned_first_side='left',C2_start_bridge_s=startbridge,reference_path_time_s=elapsed,DS_bridge_time_scale=2.,no_swing_causal_trial=False,only_first_left_prefix=True,source_no_qdd_FF_added=True,reference_tail_end_s=plan.XEND,actual_first_step=getattr(self,'actual_first_step',None),declared_phase=phi,world_neck_IK_error_m=float(np.linalg.norm(neckres(solneck.x))),swing_reference_start_world=None if not self.active_swing else self.lift0.tolist(),swing_reference_end_world=None if not self.active_swing else self.lift_end.tolist(),opposite_stance_foot_actual_world=None if not self.active_swing else d.geom_xpos[self.feet['left' if self.active_swing=='right' else 'right']].tolist(),entry_C2_quintic_0p2s=entry,hold_start_actual_certificate_s=self.hold_start,COM_kinematic_calibration=self.COM_kinematic_calibration,leg_tau_bias_in_position=False,gravity_support_anchor18_Nm=self.tau.tolist(),bounded_gravity_delta18_Nm=delta.tolist(),source_FF_not_fullscene_feasibility_claim=True,qd_ref_policy='hold zero desired velocity with original damping; moving phase finite derivative of sent q_ref, original slew discrepancy logged',teacher_abort_reason=abort,motor_command={'q_ref18_rad':qref.tolist(),'qd_ref18_rad_s':qdref.tolist(),'tau_ff18_Nm':ff.tolist()})
  return (qref-env.neutral)/env.scale,meta
