from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
assert json.loads((r/'training_completed.json').read_text())['completed']
for name,actor,ordinary in [('native_split',r/'train_split/final.onnx',Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')),('native_shared',r/'train_shared/final.onnx',r/'train_shared/final.onnx')]:
 atomic_json(r/'native_progress.json',dict(arm=name,started_unix=time.time()))
 subprocess.run(['.venv/bin/python',str(r/'native_eval.py'),'--actor',str(actor),'--ordinary',str(ordinary),'--out',str(r/name)],check=True,timeout=420)
 s=json.loads((r/name/'completed.json').read_text());assert s['errors']==0 and s['candidate_count']==128 and s['regression_count']==7
subprocess.run(['.venv/bin/python',str(r/'compare.py'),'baseline_s05','baseline_cat_split','frozen_cat_joint','native_split','native_shared'],check=True,timeout=30)
atomic_json(r/'native_completed.json',dict(completed=True,finished_unix=time.time()))
