from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS
from sim2sim.research.models import NativeAnchor
p=Path('results/jolt_learning_20260911')
s=json.loads((p/'task_smoke_native/summary.json').read_text());reports=[]
actor=NativeAnchor(p/'task_initial.onnx')
for ep in s['episodes']:
 if 'crouch' in ep['case']:continue
 native=json.loads(Path(ep['trace']).read_text())['rows']
 w=World(replace(TASKS['roller'],seconds=len(native)*.02),roller_contract=True,state_input=actor.state_input,task_input=actor.task_input)
 maxima=dict(obs=0.,action=0.,command=0.); first=None
 try:
  obs=w.reset(ep['seed'],'keyboard_'+ep['case'])
  for i,row in enumerate(native):
   action=actor(obs[None])[0]
   for name,value in [('obs',obs),('action',action),('command',w.command())]:
    error=float(np.max(np.abs(value-np.asarray(row[name]))));maxima[name]=max(maxima[name],error)
    if error>1e-5 and first is None:first=dict(index=i,t=w.t,field=name,error=error)
   obs=w.step(action)
  reports.append(dict(case=ep['case'],steps=len(native),max_abs=maxima,first_difference=first,passed=first is None))
 finally:w.close()
(p/'training_native_contract.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
