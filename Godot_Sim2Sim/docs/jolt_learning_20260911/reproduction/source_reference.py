"""Three-way diagnostic of factory policy under identical keyboard task semantics."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS,DT
from sim2sim.research.world import World
from sim2sim.research.evaluate import record
from sim2sim.standalone.cases import standard_cases
from sim2sim.standalone.score import brake_metrics
root=Path.cwd(); p=root/'results/jolt_learning_20260911'
def episode(job):
 backend,profile,case_name,seed=job;case=standard_cases()[case_name]
 actor=NativeAnchor(p/'baseline/roller.onnx')
 w=World(replace(TASKS['roller'],seconds=case['seconds']),backend=backend,reference_profile=profile,roller_contract=True)
 try:
  obs=w.reset(seed,'keyboard_'+case_name);rows=[]
  for i in range(round(case['seconds']/DT)):
   action=actor(obs[None])[0];obs=w.step(action);rows.append(record(w,action))
  metrics=brake_metrics(rows,case['brake_times'][0],case['seconds'])
  row=dict(backend=backend,profile=profile,case=case_name,seed=seed,source_sha256=actor.sha256,metrics=metrics,physics=w.physics)
  trace=p/'source_references'/f'{backend}_{profile}_{case_name}_{seed}.json';trace.parent.mkdir(exist_ok=True)
  trace.write_text(json.dumps(dict(result=row,trajectory=[{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items()} for r in rows])))
  row['trace']=str(trace);print(json.dumps(row),flush=True);return row
 finally:w.close()
jobs=[(backend,profile,case,seed) for backend,profile in [('mujoco','xml'),('mujoco','source_play'),('godot','xml')] for case in ['roller_brake_1s','roller_brake_3s','roller_turn_brake'] for seed in [917000,917001]]
with ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(episode,jobs))
(p/'source_reference_summary.json').write_text(json.dumps(result,indent=2))
