from pathlib import Path
import json,subprocess,shutil
import numpy as np
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.score import score
from sim2sim.godot_proc import _headless_overlay
r=Path('results/sprint_stop_state_20260912').resolve();p=r/'diagnostic/runtime';case=r/'reproduce_0.json';out=r/'cold_probe';out.mkdir(exist_ok=False)
results={}
for name,mode in [('warm','warm'),('cold_1','cold'),('cold_2','cold')]:
 o=out/name;o.mkdir();overlay=_headless_overlay(p)
 cmd=['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(case),'--trace='+str(o/'trace.json'),'--probe-mode='+mode,'--snapshot='+str(out/'state.bin')]
 try:
  with (o/'player.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=35)
 finally:shutil.rmtree(overlay)
 results[name]=score(o/'trace.json',case);atomic_json(o/'metrics.json',results[name]);print(name,results[name]['task_metrics']['stops'],flush=True)
a=json.loads((out/'warm/trace.json').read_text())['rows'];original=json.loads(Path(json.loads((r/'source.json').read_text())['episode']['trace']).read_text())['rows']
parity={}
for key in ['obs','command','action','ctrl','last_action']:
 error=float(np.max(np.abs(np.array([x[key] for x in a])-np.array([x[key] for x in original]))));assert error<1e-5,(key,error);parity[key]=error
cold=[]
for name in ['cold_1','cold_2']:
 b=json.loads((out/name/'trace.json').read_text())['rows'];first={k:float(np.max(np.abs(np.array(a[450][k])-np.array(b[450][k])))) for k in ['obs','command','action','ctrl','last_action']};assert max(first.values())<1e-5,first;cold.append(first)
atomic_json(r/'cold_probe_completed.json',dict(completed=True,results=results,instrumented_warm_parity=parity,cold_initial_policy_parity=cold,scope='All rigid link transforms/solver velocities, FD/EMA observer cache, controller history restored; Jolt contact and solver warmstart not transferred. No rebake of joints on restoration.'))
