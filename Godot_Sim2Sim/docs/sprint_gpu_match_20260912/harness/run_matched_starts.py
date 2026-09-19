from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912')
assert json.loads((r/'native_completed.json').read_text())['completed']
for command in json.loads((r/'matched_start_protocol.json').read_text())['commands']:
 out=Path(command[command.index('--out')+1]);atomic_json(r/'matched_start_progress.json',dict(arm=out.name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=180)
 assert json.loads((out/'completed.json').read_text())['worlds']==257
atomic_json(r/'matched_start_completed.json',dict(completed=True,finished_unix=time.time()))
