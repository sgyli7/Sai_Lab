from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912')
assert json.loads((r/'smoke_graph/completed.json').read_text())['parity_max_abs']<1e-5
assert json.loads((r/'smoke_graph_resume/completed.json').read_text())['iterations']==16
commands=json.loads((r/'gpu_pair_graph_protocol.json').read_text())['commands']
for command in commands:
 out=Path(command[command.index('--output')+1])
 atomic_json(r/'pair_graph_progress.json',dict(stage='training',arm=out.name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=300)
 completed=json.loads((out/'completed.json').read_text())
 assert completed['iterations']==128 and completed['samples']==4194304 and completed['parity_max_abs']<1e-5
atomic_json(r/'pair_graph_completed.json',dict(completed=True,finished_unix=time.time()))
