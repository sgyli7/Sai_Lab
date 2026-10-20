"""12 single same-world open capture ->5s hold -> native coherent plan.
No reset, no live q/v writes, no environment/drive/source modification.
"""
from pathlib import Path
import sys,importlib.util,numpy as np
from native_coherent_motion import NativeCoherentMotion
R=Path(__file__).parent;C=R/'capture_prefix_source';sys.path.insert(0,str(C));sp=importlib.util.spec_from_file_location('unchanged_capture_prefix12',C/'capture_provider.py');cp=importlib.util.module_from_spec(sp);sp.loader.exec_module(cp);sys.path.remove(str(C))
class AlternatingBackwardTeacher:
 def __init__(self,program,cold_rod,initial):
  self.capture=cp.AlternatingBackwardTeacher(program,cold_rod,initial);self.motion=None;self.motion_start=None;self.duration=15.;self.warmup=0.;self.walk_time=4.90625;self.frames=self.capture.frames;self.cycles=4;self.cycle_back=.125;self.capture_complete=False;self.landing_events=[];self.actual_first_step=None;self.complete_requested_motion_hold=False
 def __getattr__(self,n):return getattr(self.capture,n)
 def context(self,e,t):
  b=self.capture.b;m,d=e.model,e.data
  return dict(actual_qpos=d.qpos.tolist(),actual_qvel=d.qvel.tolist(),original_env_target=e.target.tolist(),held_motor_command=dict(e.command_details),clock_s=float(t),cold_rod_fixed_anchor=b.cold.tolist(),restoring_stance_anchors={s:x.tolist() for s,x in b.anchors.items()},observed_actual_foot_anchors={s:d.geom_xpos[m.geom(s+'_actual_sole').id].tolist() for s in ['left','right']},observed_foot_rotations={s:d.geom_xmat[m.geom(s+'_actual_sole').id].tolist() for s in ['left','right']},hold_start=b.hold_start,hold_complete=self.capture.capture_complete,observed_root_world=d.qpos[:3].tolist(),observed_cart_world=d.xpos[m.body('cart_base').id].tolist(),observed_cargo_world=d.xpos[m.body('probe_cargo').id].tolist())
 def action(self,e,t,actual_rod=None):
  if self.motion is None:
   action,meta=self.capture.action(e,t,actual_rod)
   if not meta.get('capture_hold_complete'):return action,meta
   self.capture_complete=True;self.motion_start=float(t);self.capture_context=self.context(e,t);self.carry_slip=np.array(e.cumulative_foot_slip).copy();self.motion=NativeCoherentMotion(e,self.capture_context,self.carry_slip)
  elapsed=float(t-self.motion_start)
  if elapsed>self.walk_time+3.-1e-9:
   self.complete_requested_motion_hold=True;return {},dict(complete_requested_motion_hold=True,teacher_abort_reason=None,motion_start_time_s=self.motion_start)
  action,meta=self.motion.prepare(elapsed);self.landing_events=list(getattr(e,'actual_foot_landing_events',[]));self.actual_first_step=self.landing_events[0] if self.landing_events else None
  meta.update(capture_prefix_hold_complete=True,capture_hold_complete=False,active_capture_not_pregripped=True,motion_start_time_s=self.motion_start,motion_elapsed_s=elapsed,command_to_stop_plan_s=self.walk_time,requested_stop_hold_s=3.,capture_context=self.capture_context,cumulative_foot_slip_at_motion_start_R_L=self.carry_slip.tolist(),cumulative_foot_slip_current_R_L=list(e.cumulative_foot_slip),no_material_slip_or_clock_reset=True,actual_landing_events=self.landing_events,motion_reference_heading=self.motion.alignment.R.apply([1.,0.,0.]).tolist())
  return action,meta
