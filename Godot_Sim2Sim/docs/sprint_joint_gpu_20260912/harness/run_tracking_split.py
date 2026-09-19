from pathlib import Path
import json,subprocess,time,hashlib
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');command=json.loads((r/'tracking_split_protocol.json').read_text())['command']
for f,sha in json.loads((r/'train_tracking/config.json').read_text())['code_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha
for phase,cmd,timeout in [('training',command,260),
 ('native',['.venv/bin/python',str(r/'native_eval.py'),'--actor',str(r/'train_tracking_split/final.onnx'),'--ordinary','results/sprint_20260912/runs/s05_native_handoff/final.onnx','--out',str(r/'native_trained_tracking_split')],420),
 ('gpu',['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts_post.py'),'--actor',str(r/'train_tracking_split/final.onnx'),'--ordinary','results/sprint_20260912/runs/s05_native_handoff/final.onnx','--feet','jolt','--out',str(r/'post_native_trained_tracking_split')],150)]:
 atomic_json(r/'tracking_split_progress.json',dict(phase=phase,started_unix=time.time()));subprocess.run(cmd,check=True,timeout=timeout)
subprocess.run(['.venv/bin/python',str(r/'compare.py'),'baseline_s05','baseline_cat_split','frozen_cat_joint','native_split','native_shared','native_tracking','native_tracking_split','native_trained_tracking_split'],check=True,timeout=30)
atomic_json(r/'tracking_split_completed.json',dict(completed=True,finished_unix=time.time()))
