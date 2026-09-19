from pathlib import Path
import json,subprocess,time,hashlib
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');s05=Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx');cat=Path('results/sprint_gpu_match_20260912/train_cat/final.onnx')
assert json.loads((r/'post_score_validation.json').read_text())['passed']
arms=[('baseline_s05',s05,s05),('baseline_cat_split',cat,s05),('frozen_cat_joint',cat,cat),('native_split',r/'train_split/final.onnx',s05),('native_shared',r/'train_shared/final.onnx',r/'train_shared/final.onnx'),('native_tracking',r/'train_tracking/final.onnx',r/'train_tracking/final.onnx'),('native_tracking_split',r/'train_tracking/final.onnx',s05)]
atomic_json(r/'post_scoring_protocol.json',dict(arms=[dict(name=n,actor=str(a),ordinary=str(o)) for n,a,o in arms],change='Store the final state; score state[i+1] against action[i]. Native scoring and training unchanged.',validation='Eight native trace differential cases, max discrepancy1.53e-13; missing final state rejected',previous_pre_step_diagnostics='Retained but superseded for cross-engine comparison',helper_sha256=hashlib.sha256((r/'gpu_native_starts_post.py').read_bytes()).hexdigest(),scorer_sha256=hashlib.sha256((r/'post_score.py').read_bytes()).hexdigest()))
for name,actor,ordinary in arms:
 atomic_json(r/'post_scoring_progress.json',dict(arm=name,started_unix=time.time()))
 subprocess.run(['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts_post.py'),'--actor',str(actor),'--ordinary',str(ordinary),'--feet','jolt','--out',str(r/('post_'+name))],check=True,timeout=100)
atomic_json(r/'post_scoring_completed.json',dict(completed=True,finished_unix=time.time()))
