from pathlib import Path
import subprocess,json,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');s05='results/sprint_20260912/runs/s05_native_handoff/final.onnx';cat='results/sprint_gpu_match_20260912/train_cat/final.onnx'
for name,sprint,ordinary in [('baseline_s05',s05,s05),('baseline_cat_split',cat,s05),('frozen_cat_joint',cat,cat)]:
 atomic_json(r/'baseline_progress.json',dict(arm=name,started_unix=time.time()))
 command=['.venv/bin/python',str(r/'native_eval.py'),'--actor',sprint,'--ordinary',ordinary,'--out',str(r/name)]
 subprocess.run(command,check=True,timeout=420)
 s=json.loads((r/name/'completed.json').read_text());assert s['errors']==0 and s['candidate_count']==128 and s['regression_count']==7
atomic_json(r/'baseline_completed.json',dict(completed=True,finished_unix=time.time()))
