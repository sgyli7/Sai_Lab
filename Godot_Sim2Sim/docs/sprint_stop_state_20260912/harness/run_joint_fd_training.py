from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912');p=json.loads((r/'training_protocol.json').read_text())
for phase,key,timeout in [('smoke','smoke_command',100),('training','command',500)]:
 atomic_json(r/'training_progress.json',dict(phase=phase,started_unix=time.time()))
 subprocess.run(p[key],check=True,timeout=timeout)
 out=Path(p[key][p[key].index('--output')+1]);s=json.loads((out/'completed.json').read_text());assert s['collection_device']==s['learner_device']=='cuda' and s['parity_max_abs']<1e-5
atomic_json(r/'training_completed.json',dict(completed=True,finished_unix=time.time()))
