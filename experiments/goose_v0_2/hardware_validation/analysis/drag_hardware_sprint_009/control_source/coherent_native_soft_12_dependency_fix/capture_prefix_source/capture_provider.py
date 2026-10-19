from pathlib import Path
import sys,json,importlib.util,numpy as np,mujoco as mj
from capture_reference import blend
R=Path(__file__).parent;sys.path.insert(0,str(R/'baseline_capture'));sp=importlib.util.spec_from_file_location('capture_standing_teacher',R/'baseline_capture/legacy_position_teacher.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);sys.path.remove(str(R/'baseline_capture'))
class AlternatingBackwardTeacher:
 def __init__(self,program,cold_rod,initial):
  argv=sys.argv[:];sys.argv=['capture','1']
  try:self.b=mod.AlternatingBackwardTeacher(R/'baseline_capture/causal_program.json',cold_rod,initial)
  finally:sys.argv=argv
  self.duration=8.;self.warmup=0.;self.walk_time=0.;self.frames=self.b.frames;self.cycles=0;self.cycle_back=0.;self.open_tau=np.array(json.loads((R/'static_rebalance_free16.json').read_text())['new_tau18_Nm']);self.closed_tau=np.array(json.loads((R/'closed_static_rebalance.json').read_text())['new_tau18_Nm']);self.capture_complete=False;self.landing_events=[];self.grefs=None
 def __getattr__(self,n):return getattr(self.b,n)
 def action(self,env,t,actual_rod=None):
  if self.grefs is None:
   self.grefs=[]
   for q in [np.load(R/'cold_state.npz')['qpos'],np.load(R/'closed_reference_cold.npz')['qpos']]:
    d=mj.MjData(env.model);d.qpos[:]=q;mj.mj_forward(env.model,d);self.grefs.append(d.qfrc_bias.copy())
  z=blend(t);self.b.tau=(1-z)*self.open_tau+z*self.closed_tau
  if self.b.boot:self.b.g0=(1-z)*self.grefs[0]+z*self.grefs[1]
  action,meta=self.b.action(env,t,actual_rod);meta.update(active_capture_not_pregripped=True,prepositioned_true_physical_grip_not_active_capture=False,active_capture_reference_finished=t>=.8,capture_hold_start=self.b.hold_start,capture_only_no_pull=True,reference_start_TCP=[.300347296355334,0,.4219696155060244])
  if self.b.hold_start is not None and t-self.b.hold_start>=5.-1e-9:
   hm=getattr(env,'loaded_hold_metrics',{});ok=hm.get('samples',0)>=4995 and hm.get('min_jaw_normal_N',0)>=5 and hm.get('min_foot_normal_N',0)>=20 and hm.get('max_COM_XY_error_m',1)<=.005 and hm.get('max_COM_height_error_m',1)<=.005 and hm.get('max_tilt_deg',30)<=5 and max(getattr(env,'cumulative_foot_slip',[1,1]))<=.005
   self.capture_complete=bool(ok);meta['capture_hold_complete']=bool(ok)
   if not ok:meta['teacher_abort_reason']='actual_capture_hold5s_qualification_failed'
  return action,meta
