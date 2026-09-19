from pathlib import Path
import json,subprocess,time,hashlib
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
assert json.loads((r/'native_completed.json').read_text())['completed']
assert json.loads((r/'matched_completed.json').read_text())['completed']
protocol=json.loads((r/'tracking_protocol.json').read_text())
assert hashlib.sha256(Path('scripts/sprint_gpu_train.py').read_bytes()).hexdigest()==protocol['code_sha256']
base=protocol['command']
for phase,command,timeout in [
 ('smoke',protocol['smoke_command'],100),('training',base,260),
 ('native',['.venv/bin/python',str(r/'native_eval.py'),'--actor',str(r/'train_tracking/final.onnx'),'--ordinary',str(r/'train_tracking/final.onnx'),'--out',str(r/'native_tracking')],420),
 ('matched',['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts.py'),'--actor',str(r/'train_tracking/final.onnx'),'--ordinary',str(r/'train_tracking/final.onnx'),'--feet','jolt','--out',str(r/'matched_tracking')],150)]:
 atomic_json(r/'tracking_progress.json',dict(phase=phase,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=timeout)
 if phase in ('smoke','training'):
  out=r/('smoke_tracking' if phase=='smoke' else 'train_tracking');c=json.loads((out/'completed.json').read_text());assert c['parity_max_abs']<1e-5
subprocess.run(['.venv/bin/python',str(r/'compare.py'),'baseline_s05','baseline_cat_split','frozen_cat_joint','native_split','native_shared','native_tracking'],check=True,timeout=30)
atomic_json(r/'tracking_completed.json',dict(completed=True,finished_unix=time.time()))
