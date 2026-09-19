from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
assert json.loads((r/'baseline_completed.json').read_text())['completed']
assert json.loads((r/'smoke_shared/completed.json').read_text())['parity_max_abs']<1e-5
assert json.loads((r/'smoke_shared_resume/completed.json').read_text())['iterations']==16
for command in json.loads((r/'training_protocol.json').read_text())['commands']:
 out=Path(command[command.index('--output')+1]);atomic_json(r/'training_progress.json',dict(arm=out.name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=360)
 s=json.loads((out/'completed.json').read_text());assert s['samples']==8388608 and s['parity_max_abs']<1e-5
atomic_json(r/'training_completed.json',dict(completed=True,finished_unix=time.time()))
