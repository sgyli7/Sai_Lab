from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912')
assert json.loads((r/'smoke_cat/completed.json').read_text())['parity_max_abs']<1e-5
assert json.loads((r/'smoke_cat_resume/completed.json').read_text())['iterations']==16
assert json.loads((r/'smoke_positive/completed.json').read_text())['parity_max_abs']<1e-5
for command in json.loads((r/'cat_pair_protocol.json').read_text())['commands']:
 out=Path(command[command.index('--output')+1]);atomic_json(r/'cat_pair_progress.json',dict(arm=out.name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=240)
 result=json.loads((out/'completed.json').read_text());assert result['samples']==4194304 and result['parity_max_abs']<1e-5
atomic_json(r/'cat_pair_completed.json',dict(completed=True,finished_unix=time.time()))
