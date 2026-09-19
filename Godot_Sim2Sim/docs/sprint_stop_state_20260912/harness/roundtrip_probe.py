from pathlib import Path
import json,subprocess,shutil
import numpy as np
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.score import score
from sim2sim.godot_proc import _headless_overlay
r=Path('results/sprint_stop_state_20260912').resolve();p=r/'diagnostic/runtime';f=p/'standalone/stop_state_probe.gd';s=f.read_text()
s=s.replace(' print("[DEBUG-stop-state] captured "+str(session.steps))',' print("[DEBUG-stop-state] captured "+str(session.steps))\n return snapshot')
s=s.replace('  if probe_mode=="warm": _capture_probe()','  if probe_mode in ["warm","inplace","freeze"]:\n   var state=_capture_probe()\n   if probe_mode=="freeze": _freeze(true)\n   if probe_mode in ["inplace","freeze"]: _restore_probe(state)')
f.write_text(s);out=r/'roundtrip';out.mkdir(exist_ok=False);results={}
for name in ['warm','inplace','freeze']:
 o=out/name;o.mkdir();overlay=_headless_overlay(p)
 try:
  command=['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(r/'reproduce_0.json'),'--trace='+str(o/'trace.json'),'--probe-mode='+name,'--snapshot='+str(o/'state.bin')]
  with (o/'player.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=30)
 finally:shutil.rmtree(overlay)
 results[name]=score(o/'trace.json',r/'reproduce_0.json');atomic_json(o/'metrics.json',results[name]);print(name,results[name]['task_metrics']['stops'],flush=True)
a=json.loads((out/'warm/trace.json').read_text())['rows'];errors={}
for name in ['inplace','freeze']:
 b=json.loads((out/name/'trace.json').read_text())['rows'];errors[name]={key:float(np.max(np.abs(np.array([x[key] for x in a[:451]])-np.array([x[key] for x in b[:451]])))) for key in ['obs','command','action','ctrl','last_action']};assert max(errors[name].values())<1e-5
state=json.loads((r/'cold_probe/state.bin.json').read_text());j=sorted([x for x in state['joints'] if x['act_index']>=0],key=lambda x:x['act_index']);delta=[x['qd_solver']-x['qd_kin'] for x in j]
atomic_json(r/'roundtrip_completed.json',dict(completed=True,results=results,through_cut_control_parity=errors,joint_solver_minus_observer_qd=delta,max_joint_speed_difference=float(np.max(np.abs(delta))),scope='Diagnostic state roundtrip; freeze is excluded from deliverable behavior and acceptance. No mass, gravity, collision or actuator parameter changed.'))
