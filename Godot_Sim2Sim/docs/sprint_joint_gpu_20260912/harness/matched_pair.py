from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
assert json.loads((r/'training_completed.json').read_text())['completed']
for arm,ordinary in [('split',Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')),('shared',r/'train_shared/final.onnx')]:
 atomic_json(r/'matched_progress.json',dict(arm=arm,started_unix=time.time(),scope='Quality comparison; concurrent native evaluation, no throughput claim'))
 subprocess.run(['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts.py'),'--actor',str(r/('train_'+arm)/'final.onnx'),'--ordinary',str(ordinary),'--feet','jolt','--out',str(r/('matched_'+arm))],check=True,timeout=150)
atomic_json(r/'matched_completed.json',dict(completed=True,finished_unix=time.time()))
