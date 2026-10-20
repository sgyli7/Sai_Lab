"""Candidate13 only: existing nativeLP legFF at each real5ms drive tick.
50Hz packet/front q-only are never resampled from future actual observations.
Zero integration; caller applies one final source-limited total and old fineguard.
"""
import time,numpy as np,mujoco as mj
from teacher_drive_segment import compute_drive_tick
from soft_phase_model import solve,dispatch_or_validated_witness,smooth
from native_dispatch_witness import witness
import complete_cycle_plan as plan

def refresh(motion,packet,elapsed,current_target):
 e=motion.env;m,d=e.model,e.data;before=time.perf_counter();previous=np.array(getattr(e,'command_details',{}).get('tau_ff18_Nm',motion.last_ff));target,_,preview=compute_drive_tick(e,packet,elapsed,current_target,d.qpos,d.qvel,previous);pd=np.array(preview['PD18']);pd[:6]=e.limit_torque(pd,d.qvel[e.drive_vidx])[:6]
 # Current observed state, previous actual heldctrl; scratch only. Existing
 # same-command fresh native modes are computed by unchanged solve below.
 sensed=mj.MjData(m);sensed.qpos[:]=d.qpos;sensed.qvel[:]=d.qvel;sensed.ctrl[:]=d.ctrl;mj.mj_forward(m,sensed);N={s:0. for s in ['left','right']};grounds={m.geom(n).id for n in ['ground','sole_ground_box']}
 for i,c in enumerate(sensed.contact):
  f=np.zeros(6);mj.mj_contactForce(m,sensed,i,f);F=c.frame.reshape(3,3).T@f[:3]
  for s in N:
   g=m.geom(s+'_actual_sole').id
   if g in (c.geom1,c.geom2) and any(z in (c.geom1,c.geom2) for z in grounds):N[s]+=max(0.,F[2]*(1 if c.geom2==g else -1))
 ref=motion.curve.sample(float(elapsed));force=plan.support_force_objective(min(elapsed,plan.END),sum(N.values()));tic=time.perf_counter();result=solve(e,sensed,ref['qacc'],pd,previous,force['normal_soft_target_N'],100*force['weight_blend'],root_reference=(ref['qpos'],ref['qvel'],ref['qacc']),unload_side=force['unload_side']);wall=time.perf_counter()-tic;decision=dispatch_or_validated_witness(result,previous,motion.consecutive,motion.fallback_total);why=None;ff=previous.copy();z=float(smooth(elapsed/.5))
 if not decision['accepted']:why=decision['reason']
 else:
  motion.consecutive=decision['consecutive'];motion.fallback_total=decision['total'];raw=np.array(decision['FF18']);ff=raw if decision['witness_hold'] else (1-z)*motion.entry.held_FF+z*raw;ff[:6]=0.
  if result['success']:
   if result['source_applied_model_error']>1e-6:why='native_LP_source_applied_model_error'
   if result['hard_equality_residual']>1e-6 or result['hard_inequality_residual']>1e-6 or result['maximum_contact_circle_excess']>1e-6:why='native_LP_hard_physical_readback'
  if max(abs(ff)-e.cont)>1e-8:why='conservative_FF_component_software_guard_not_hardware'
 _,total,dispatched=compute_drive_tick(e,packet,elapsed,current_target,d.qpos,d.qvel,ff);fresh=witness(e,d.qpos,d.qvel,total)
 if not fresh['all_finite'] or not fresh['source_legal']:why='blended_dispatch_native_source_or_finite'
 if max(fresh[k] for k in ['force_balance_max','equality_acceleration_max','contact_KKT_max','contact_circle_excess_max','dry_friction_mode_max'])>1e-6:why='blended_dispatch_native_hard_witness'
 if not why:motion.last_ff=ff.copy()
 record=dict(elapsed_s=float(elapsed),actual_clock_s=float(d.time),LP_wall_s=wall,whole_refresh_wall_s=time.perf_counter()-before,FF18=ff.tolist(),previous_actual_FF18=previous.tolist(),PD18=pd.tolist(),preview_total18=total.tolist(),entry_C2_blend=z,soft_phase=result,validated_witness_hold=decision,actual_blended_native_witness=fresh,abort_reason=why,FF_refresh_Hz=200,Teacher_reference_Hz=50,nonrealtime_simulation_only=True,front6_qonly_FF0=True,fallback_max_consecutive_drive_ticks=2,fallback_max_time_s=.010,raw_LP_qacc_lambda_not_blended_actual=True,observed_native_N=N)
 if hasattr(motion,'current_drive_meta'):
  motion.current_drive_meta.update(soft_phase=result,validated_witness_hold=decision,drive_FF_refresh=record,actual_dispatched_native_witness=fresh,FF_C2_entry_blend=z)
 return record
