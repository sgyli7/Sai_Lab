from pathlib import Path
import json,subprocess,importlib.util
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912');assert json.loads((r/'training_completed.json').read_text())['completed'];assert json.loads((r/'joint_contract_validation.json').read_text())['passed']
subprocess.run(['.venv/bin/python','results/sprint_joint_gpu_20260912/native_eval.py','--actor',str(r/'train_joint_fd/final.onnx'),'--ordinary','results/sprint_20260912/runs/s05_native_handoff/final.onnx','--out',str(r/'native_joint_fd')],check=True,timeout=420)
spec=importlib.util.spec_from_file_location('joint_comparison','results/sprint_joint_gpu_20260912/compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.compare(Path('results/sprint_joint_gpu_20260912/baseline_s05'),r/'native_joint_fd');b=m.compare(Path('results/sprint_joint_gpu_20260912/native_trained_tracking_split'),r/'native_joint_fd')
atomic_json(r/'native_comparison.json',dict(completed=True,vs_s05=a,vs_previous_best=b));print(a['counts'],a['development_eligible'],a['failed'],flush=True)
