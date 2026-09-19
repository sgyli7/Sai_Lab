from pathlib import Path
import json,subprocess,shutil
import numpy as np
import torch
from sim2sim.research.kinematic_observer import KinematicObserver
from sim2sim.research.queue import atomic_json
from sim2sim.godot_proc import _headless_overlay
r=Path('results/sprint_stop_state_20260912').resolve();p=r/'diagnostic/runtime';f=p/'standalone/stop_state_probe.gd';s=f.read_text();s=s.replace('var captured = false','var captured = false\nvar sensor_rows = []')
s=s.replace(' if not ready_to_run: return',' if not ready_to_run: return\n if probe_mode=="sensors": _record_sensor_sample()',1)
s+='''
func _record_sensor_sample():
 _send_state("step")
 var body=Contract.body_state(local_reply,robot_config)
 sensor_rows.append({"t":_t,"q":local_reply.q.duplicate(),"qd":local_reply.qd.duplicate(),"quat":body.base_quat.duplicate(),"omega":local_reply.base_angvel_local.duplicate()})

func _refresh_after_physics(delta: float) -> void:
 var advanced=_kin_after_tick
 super._refresh_after_physics(delta)
 if probe_mode=="sensors" and advanced: _record_sensor_sample()

func _finish() -> void:
 if probe_mode=="sensors":
  var file=FileAccess.open(snapshot_path+".sensors.json",FileAccess.WRITE)
  file.store_string(JSON.stringify(sensor_rows))
  file.close()
 super._finish()
'''
f.write_text(s);o=r/'native_sensor_trace';o.mkdir(exist_ok=False);overlay=_headless_overlay(p)
try:
 cmd=['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(r/'reproduce_0.json'),'--trace='+str(o/'trace.json'),'--probe-mode=sensors','--snapshot='+str(o/'state')]
 with (o/'player.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=35)
finally:shutil.rmtree(overlay)
a=json.loads((o/'trace.json').read_text())['rows'];b=json.loads((r/'cold_probe/warm/trace.json').read_text())['rows'];parity={k:float(np.max(np.abs(np.array([x[k] for x in a])-np.array([x[k] for x in b])))) for k in ['obs','action','ctrl','last_action','command']};assert max(parity.values())<1e-5
rows=json.loads((o/'state.sensors.json').read_text());assert len(rows)==2401,len(rows);metrics={}
for dtype in [torch.float64,torch.float32]:
 ob=KinematicObserver(1,'cpu',dtype=dtype);first=rows[0];ob.reset(torch.arange(1),torch.tensor([first['q']],dtype=dtype),torch.tensor([first['quat']],dtype=dtype));maxq=maxw=0.
 for row in rows[1:]:
  quat=torch.tensor([row['quat']],dtype=dtype);ob.update(torch.tensor([row['q']],dtype=dtype),quat)
  maxq=max(maxq,float(np.max(np.abs(ob.qd.numpy()[0]-row['qd']))));maxw=max(maxw,float(np.max(np.abs(ob.angular_local(quat).numpy()[0]-row['omega']))))
 metrics[str(dtype)]=dict(joint_velocity_max_abs=maxq,body_angular_velocity_max_abs=maxw)
assert max(metrics['torch.float64'].values())<1e-7,metrics
atomic_json(r/'native_observer_validation.json',dict(passed=True,substeps=2400,observer='FD at200Hz then EMA tau5ms, matching Jolt',capture_policy_parity=parity,numeric_error=metrics))
print(metrics)
