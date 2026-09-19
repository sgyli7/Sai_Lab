from pathlib import Path
import json,subprocess,math
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
assert json.loads((r/'baseline_completed.json').read_text())['completed']
base=['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python','scripts/sprint_gpu_train.py','--feet','jolt','--cuda-graphs','--constraints','cat','--controller','shared','--teacher-replay',str(r/'teacher/manifest.json'),'--teacher-weight','.02','--envs','512','--steps','64','--seed','953001']
for name,iterations,resume in [('smoke_shared',8,None),('smoke_shared_resume',16,r/'smoke_shared/latest.pt')]:
 command=[*base,'--output',str(r/name),'--iterations',str(iterations)]
 if resume:command += ['--resume',str(resume)]
 subprocess.run(command,check=True,timeout=100)
 c=json.loads((r/name/'completed.json').read_text());records=[json.loads(x) for x in (r/name/'metrics.jsonl').read_text().splitlines()]
 assert c['status']=='completed' and c['parity_max_abs']<1e-5 and c['iterations']==iterations
 assert all(x['actor_samples']==32768 and math.isfinite(x['teacher_kl']) for x in records)
 assert any(x['teacher_kl']>0 and x['actor_updates']>0 for x in records)
atomic_json(r/'smoke_completed.json',dict(completed=True,shared_actor_samples=32768,resume_iterations=16))
