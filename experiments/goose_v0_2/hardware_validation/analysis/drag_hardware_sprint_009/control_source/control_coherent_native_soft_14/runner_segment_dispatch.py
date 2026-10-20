"""Private Teacher consumer for existing1ms guarded runner, never autoruns.
The caller must supply the EXISTING logged(nstep=5) fine-guard callback;
there is deliberately no direct mj_step/default integration path here.
"""
import numpy as np
class DriveFFRefreshAbort(RuntimeError):pass
from teacher_drive_segment import compute_drive_tick

def dispatch_segment_20ms(env,packet,held_FF18,motion_start,guarded_five_fine):
 if guarded_five_fine is None:raise ValueError('original immediate1ms guard callback required')
 e=env.original
 if abs(e.drive_dt-.005)>1e-12 or abs(e.policy_dt-.02)>1e-12 or abs(e.model.opt.timestep-.001)>1e-12:raise ValueError('clock mismatch')
 start=float(e.data.time);ff=np.asarray(held_FF18,float)
 if np.max(abs(ff[:6]))>1e-12:raise ValueError('front6 q-only')
 for sub in range(4):
  clock=float(e.data.time)-float(motion_start);old=e.target.copy()
  if not hasattr(env,'teacher_drive_FF_refresh'):raise DriveFFRefreshAbort('missing_declared_200Hz_FF_callback')
  refreshed=env.teacher_drive_FF_refresh(packet,clock,old)
  if refreshed.get('abort_reason'):raise DriveFFRefreshAbort(refreshed['abort_reason'])
  ff=np.array(refreshed['FF18']);target,total,details=compute_drive_tick(env,packet,clock,old,e.data.qpos,e.data.qvel,ff)
  if max(abs(total-np.array(refreshed['preview_total18'])))>1e-9:raise DriveFFRefreshAbort('same_drive_PD_FF_source_shadow_mismatch')
  e.target[:]=target;e.last_tau=total.copy();e.data.ctrl[:]=total
  env.command_details=dict(profile='teacher_coherent_plan_segment13_FF200_drive200_physics1000',requested_q_ref18_rad=details['raw_q_ref18'],actual_slewed_q_ref18_rad=target.tolist(),actual_q_ref_derivative18_rad_s=((target-old)/.005).tolist(),qd_ref18_rad_s=details['dispatch_qd18'],qd_ref_minus_q_slew_derivative18_rad_s=(np.array(details['dispatch_qd18'])-(target-old)/.005).tolist(),tau_ff18_Nm=details['held_FF18'],PD18_Nm=details['PD18'],combined_requested_tau18_Nm=details['requested_total18'],limited_total_tau18_Nm=details['limited_total18'],packet_details=details,FF_drive_refresh=refreshed,no_independent_root_or_contact_forces=True,source_total_limiter_once=True)
  # Existing callback iterates five1ms ticks; any first abort propagates and
  # leaves remaining drive/command substeps unexecuted. No try/continue here.
  guarded_five_fine(e.model,e.data,nstep=5);e.controller_updates+=1;env.mit_ticks+=1
 e.policy_updates+=1
 if abs(float(e.data.time)-start-.02)>1e-10:raise RuntimeError('original20ms command duration mismatch')
 return env.state()
