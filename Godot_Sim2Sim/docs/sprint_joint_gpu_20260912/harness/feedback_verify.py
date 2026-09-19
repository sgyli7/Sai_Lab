from pathlib import Path
import json
import numpy as np
import torch
from sim2sim.standalone.replay import shadow
from sim2sim.research.torch_walking import WalkingControl
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');suite=r/'native_feedback/suite';s=json.loads((suite/'summary.json').read_text());results=[]
for name,seed in [('sprint_alternate',927002),('sprint_right',927012),('sprint_turn_release',927000),('sprint_alternate_ordinary',927002)]:
 e=next(e for e in s['episodes'] if (e['case'],e['seed'])==(name,seed));verification=shadow(e['trace'],suite/'runtime');assert verification['passed']
 c=WalkingControl(1,e['control_config']['walk'],'cpu');rows=json.loads(Path(e['trace']).read_text())['rows'];error=0.
 for row in rows:
  body=row['body'];out=c.command(torch.tensor([row['requested_command']],dtype=torch.float32),torch.tensor([body['base_pos']],dtype=torch.float64),torch.tensor([body['base_quat']],dtype=torch.float64),torch.tensor([body['base_linvel']],dtype=torch.float64),torch.tensor([row['skill']=='sprint']))
  error=max(error,float(np.max(np.abs(out.numpy()[0]-np.array(row['command'],np.float32)))))
 assert error<1e-5
 results.append(dict(case=name,seed=seed,shadow=verification,torch_control_max_abs=error))
atomic_json(r/'feedback_verification.json',dict(passed=True,records=results))
