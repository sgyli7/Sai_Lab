"""12: same planned leg trajectory + ORIGINAL actual-native soft FF model.
No inverse-total decomposition; solver FF already appears WITH real PD in hard
balance. Front6 q-only actual-root worldneck remains an explicit boundary.
"""
from pathlib import Path
import numpy as np,mujoco as mj
import complete_cycle_plan as plan
from coherent_plan_sampler import CoherentPlan
from aligned_capture_plan import AlignedCapturePlan
from capture_plan_entry import CapturePlanEntry
from motion_stop_reference import MotionStopReference
from teacher_drive_segment import make_packet,sample_packet,compute_drive_tick
from soft_phase_model import solve,dispatch_or_validated_witness,smooth
from world_neck_qonly import command as neck_command
from native_dispatch_witness import witness as native_witness
class NativeCoherentMotion:
 def __init__(self,env,capture_context,foot_slip_R_L):
  self.env=env;self.context=capture_context;self.base=CoherentPlan(env.model,Path(__file__).parent/'complete_reference.npz',env.qidx);self.alignment=AlignedCapturePlan(self.base,capture_context);from rebuild_capture_plan import rebuild
  artifact=Path(getattr(env,'reference_output_dir',Path(__file__).parent/'readbacks'))/'observed_capture_reference.npz';artifact.parent.mkdir(exist_ok=True,parents=True)
  self.rebuild_report=rebuild(env,self.base,self.alignment,capture_context,artifact);import json
  artifact.with_suffix('.readback.json').write_text(json.dumps(self.rebuild_report,indent=2));self.rebuilt=CoherentPlan(env.model,artifact,env.qidx)
  if self.rebuild_report['max_IK']>.0005 or self.rebuild_report['minimum_leg_ROM_margin']<.02 or self.rebuild_report['max_leg_reference_source_slew_ratio']>1.+1e-8:raise ValueError(('observed-capture reference source/IK/ROM gate',self.rebuild_report))
  self.entry=CapturePlanEntry(self.rebuilt,capture_context,foot_slip_R_L);self.curve=MotionStopReference(self.entry);self.last_ff=np.array(capture_context['held_motor_command']['tau_ff18_Nm']);self.consecutive=0;self.fallback_total=0
 def prepare(self,elapsed,reference_only=True):
  e=self.env;m,d=e.model,e.data;ref=self.curve.sample(float(elapsed));packet=make_packet(self.curve,float(elapsed));front,frontmeta=neck_command(e,d.qpos,elapsed,self.alignment,self.entry.adapter_target[:6])
  # Front feedback is the original50Hz actual-world IK/q-only command, held
  # across20ms; only legs use the planned200Hz segment. No future actualstate.
  for seg in packet['segments']:
   cc=np.array(seg['bernstein_q18']);cc[:,:6]=front;seg['bernstein_q18']=cc.tolist()
  qr,qd,_=sample_packet(packet,elapsed);preview_target,_,pm=compute_drive_tick(e,packet,elapsed,e.target,d.qpos,d.qvel,np.zeros(18));pd=np.array(pm['PD18']);pd[:6]=e.limit_torque(pd,d.qvel[e.drive_vidx])[:6];grounds={m.geom(n).id for n in ['ground','sole_ground_box']};N={s:0. for s in ['left','right']}
  for k,c in enumerate(d.contact):
   for s in N:
    g=m.geom(s+'_actual_sole').id
    if g in (c.geom1,c.geom2) and any(gg in (c.geom1,c.geom2) for gg in grounds):
     f=np.zeros(6);mj.mj_contactForce(m,d,k,f);F=c.frame.reshape(3,3).T@f[:3];N[s]+=max(0.,F[2]*(1 if c.geom2==g else -1))
  ffprevious=np.array(getattr(e,'command_details',{}).get('tau_ff18_Nm',self.last_ff));force=plan.support_force_objective(min(elapsed,plan.END),sum(N.values()));rootref=(ref['qpos'],ref['qvel'],ref['qacc']);why=None
  if frontmeta['neck_IK_error_m']>.0005:why='ACTUAL_COMMANDED_neck_IK_over0p5mm'
  margin=float(min(np.minimum(qr-e.ranges[:,0],e.ranges[:,1]-qr)))
  if margin<-1e-8:why='ACTUAL_COMMANDED_public_ROM_outside_source'
  if min(np.minimum(qr[6:]-e.ranges[6:,0],e.ranges[6:,1]-qr[6:]))<.02:why='ACTUAL_COMMANDED_reference_ROM_under20mrad'
  result=None;decision=None;ff=ffprevious.copy()
  if not why and not reference_only:
   result=solve(e,d,ref['qacc'],pd,ffprevious,force['normal_soft_target_N'],100*force['weight_blend'],root_reference=rootref,unload_side=force['unload_side']);decision=dispatch_or_validated_witness(result,ffprevious,self.consecutive,self.fallback_total)
   if not decision['accepted']:why=decision['reason']
   else:
    self.consecutive=decision['consecutive'];self.fallback_total=decision['total'];rawff=np.array(decision['FF18']);z=smooth(elapsed/.5);ff=rawff if decision['witness_hold'] else (1-z)*self.entry.held_FF+z*rawff;ff[:6]=0.
    # Existing model/source readback is a dispatch requirement, not a claim
    # that the actual body will track the desired acceleration.
    if result['success']:
     if result['source_applied_model_error']>1e-6:why='native_LP_source_applied_model_error'
     if result['hard_equality_residual']>1e-6 or result['hard_inequality_residual']>1e-6 or result['maximum_contact_circle_excess']>1e-6:why='native_LP_hard_physical_readback'
    if max(abs(ff)-e.cont)>1e-8:why='conservative_FF_component_software_guard_not_hardware'
  target,total,dispatch=compute_drive_tick(e,packet,elapsed,e.target,d.qpos,d.qvel,ff);raw=np.array(dispatch['requested_total18']);power=float(np.maximum(total*d.qvel[e.drive_vidx],0).sum())
  if not np.isfinite(np.r_[target,total,ff]).all() or power>350.+1e-9:why='original_source_nonfinite_or_power'
  dispatched_witness=native_witness(e,d.qpos,d.qvel,total)
  if not dispatched_witness['all_finite'] or not dispatched_witness['source_legal']:why='blended_dispatch_native_source_or_finite'
  if max(dispatched_witness[k] for k in ['force_balance_max','equality_acceleration_max','contact_KKT_max','contact_circle_excess_max','dry_friction_mode_max'])>1e-6:why='blended_dispatch_native_hard_witness'
  if not why:self.last_ff=ff.copy()
  scratch=mj.MjData(m);scratch.qpos[:]=ref['qpos'];mj.mj_forward(m,scratch);COM=scratch.subtree_com[m.body('torso').id].copy();Jcom=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,scratch,Jcom,m.body('torso').id);COMv=Jcom@ref['qvel'];hh=1e-4;Jpair=[]
  for delta in [-hh,hh]:
   zz=mj.MjData(m);zz.qpos[:]=ref['qpos'];mj.mj_integratePos(m,zz.qpos,ref['qvel'],delta);mj.mj_forward(m,zz);JJ=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,zz,JJ,m.body('torso').id);Jpair.append(JJ)
  COMa=Jcom@ref['qacc']+(Jpair[1]-Jpair[0])@ref['qvel']/(2*hh);active=next((w['side'] for w in plan.WINDOWS if w['start']<=elapsed<=w['end']),None);stance=['left','right'] if active is None else ['right' if active=='left' else 'left']
  meta=dict(teacher_abort_reason=why,phase='coherent_planned_leg_native_soft' if elapsed<plan.END else 'final_DS_hold_no_return',stance=stance,active_swing=active,planned_swing=active,soft_phase=result,validated_witness_hold=decision,trajectory_packet=packet,motor_command=dict(q_ref18_rad=qr.tolist(),qd_ref18_rad_s=qd.tolist(),tau_ff18_Nm=ff.tolist()),segment_motor_command=dict(packet=packet,tau_ff18_Nm=ff.tolist()),front_worldneck=frontmeta,reference_qpos=ref['qpos'].tolist(),reference_qvel=ref['qvel'].tolist(),reference_qdd=ref['qacc'].tolist(),root_reference_feedback_gains_translation=[25,10],root_reference_feedback_gains_rotation=[64,16],frontend_acceleration_boundary='original soft model tracks fresh actualnative front6/internals; does not claim front follows same planned qdd',leg_qref_qd_qdd_same_C2_sampler=True,no_actual_virtual_root_leg_IK_difference=True,FF_semantics='original native soft LP FF directly; hard model includes actual PD once. NO inverse-minus-nominalPD decomposition.',FF_C2_entry_blend=float(smooth(elapsed/.5)),actual_first_tick_preview=dispatch,actual_dispatched_native_witness=dispatched_witness,actual_first_tick_power_W=power,actual_foot_normal_N=N,COM_reference_world=COM.tolist(),COM_velocity_reference_world=COMv.tolist(),COM_acceleration_reference_world=COMa.tolist(),tcp_target_world=frontmeta['world_TCP_target'],planned_progress_m=plan.travel(min(elapsed,plan.END))[0],actual_steps_not_inferred_from_phase=True,reference_governance=dict(raw_position_reference_derivative18=pm['q_reference_derivative18'],dispatch_qd_ref18=qd.tolist(),physical_target_slew_and_source_limits_unchanged=True),alignment=self.alignment.report,rebuild_reference_report={k:v for k,v in self.rebuild_report.items() if k!='checks'},cumulative_slip_at_capture_R_L=self.entry.foot_slip.tolist())
  packet['FF_hold_Hz']=200;packet['high_level_reference_Hz']=50;meta['FF_refresh_Hz']=200;meta['nonrealtime_teacher_only']=True;meta['FF50Hz_prepare_does_not_solve']=True;self.current_drive_meta=meta;e.teacher_drive_FF_refresh=self.refresh_drive_FF
  return {n:float(qr[i]) for i,n in enumerate(e.names)},meta

 def refresh_drive_FF(self,packet,elapsed,current_target):
  """Only legFF refreshed at drive; front/packet remains last50Hz command."""
  import time
  from drive_ff200 import refresh
  return refresh(self,packet,elapsed,current_target)
