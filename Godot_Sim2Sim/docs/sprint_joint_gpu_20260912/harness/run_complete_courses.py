from pathlib import Path
import json,subprocess,time,hashlib
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');p=json.loads((r/'complete_courses_protocol.json').read_text());assert hashlib.sha256(Path('scripts/sprint_gpu_train.py').read_bytes()).hexdigest()==p['code_sha256']
for phase,command,timeout in [('smoke',p['smoke_command'],60),('training',p['command'],220),
 ('native',['.venv/bin/python',str(r/'native_eval.py'),'--actor',str(r/'train_complete_courses/final.onnx'),'--ordinary',str(r/'train_complete_courses/final.onnx'),'--out',str(r/'native_complete_courses')],300),
 ('gpu',['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts_post.py'),'--actor',str(r/'train_complete_courses/final.onnx'),'--ordinary',str(r/'train_complete_courses/final.onnx'),'--feet','jolt','--out',str(r/'post_native_complete_courses')],80)]:
 atomic_json(r/'complete_courses_progress.json',dict(phase=phase,started_unix=time.time()));subprocess.run(command,check=True,timeout=timeout)
 if phase=='smoke':
  c=json.loads((r/'smoke_complete_courses/config.json').read_text());s=json.loads((r/'smoke_complete_courses/completed.json').read_text());assert len(c['course_names'])==16 and s['parity_max_abs']<1e-5
  rows=[json.loads(x) for x in (r/'smoke_complete_courses/metrics.jsonl').read_text().splitlines()];assert all(x['actor_samples']==32768 for x in rows);assert all(sum(x['course_samples'][j] for x in rows)>0 for j in range(16))
subprocess.run(['.venv/bin/python',str(r/'compare.py'),'baseline_s05','baseline_cat_split','frozen_cat_joint','native_split','native_shared','native_tracking','native_tracking_split','native_trained_tracking_split','native_complete_courses'],check=True,timeout=30)
atomic_json(r/'complete_courses_completed.json',dict(completed=True,finished_unix=time.time()))
