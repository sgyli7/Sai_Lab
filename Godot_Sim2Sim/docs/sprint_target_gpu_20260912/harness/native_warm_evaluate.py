from pathlib import Path
import argparse, json, subprocess, importlib.util
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_target_gpu_20260912')
p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,required=True);a=p.parse_args()
assert a.iteration in json.loads((r/'warm_prefix_protocol.json').read_text())['endpoints']
assert json.loads((r/'train_warm/completed.json').read_text())['completed']
actor=r/'train_warm'/f'iteration_{a.iteration:03d}'/'actor.onnx';out=r/f'native_warm_{a.iteration:03d}'
subprocess.run(['.venv/bin/python','results/sprint_joint_gpu_20260912/native_eval.py','--actor',str(actor),'--ordinary','results/sprint_20260912/runs/s05_native_handoff/final.onnx','--out',str(out)],check=True,timeout=420)
spec=importlib.util.spec_from_file_location('comparison','results/sprint_joint_gpu_20260912/compare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
comparisons={key:m.compare(Path(base),out) for key,base in [('vs_s05','results/sprint_joint_gpu_20260912/baseline_s05'),('vs_prior_best','results/sprint_joint_gpu_20260912/native_trained_tracking_split'),('vs_initial','results/sprint_stop_state_20260912/native_joint_fd')]}
atomic_json(out/'comparison.json',comparisons)
print(json.dumps({k:{x:v[x] for x in ['counts','falls','lost_baseline_successes','fixed_baseline_failures','development_eligible','failed']} for k,v in comparisons.items()}),flush=True)
