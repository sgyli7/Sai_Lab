from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912')
assert json.loads((r/'cat_pair_completed.json').read_text())['completed']
for command in json.loads((r/'cat_native_protocol.json').read_text())['commands']:
 out=Path(command[command.index('--out')+1]);atomic_json(r/'cat_native_progress.json',dict(arm=out.name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=400)
 s=json.loads((out/'completed.json').read_text());assert s['candidate_count']==128 and s['errors']==0
atomic_json(r/'cat_native_completed.json',dict(completed=True,finished_unix=time.time()))
