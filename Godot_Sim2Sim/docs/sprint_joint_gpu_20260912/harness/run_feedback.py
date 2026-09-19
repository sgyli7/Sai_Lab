from pathlib import Path
import json,subprocess,time
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');base=['.venv/bin/python',str(r/'feedback_eval.py'),'--actor',str(r/'train_tracking/final.onnx'),'--ordinary','results/sprint_20260912/runs/s05_native_handoff/final.onnx']
atomic_json(r/'feedback_progress.json',dict(phase='disabled_canary',started_unix=time.time()))
subprocess.run([*base,'--out',str(r/'feedback_disabled_canary'),'--control','results/sprint_joint_20260912/delivery/control.json','--seeds','1'],check=True,timeout=120)
old={(e['case'],e['seed']):e for e in json.loads((r/'native_tracking_split/suite/summary.json').read_text())['episodes']}
new=json.loads((r/'feedback_disabled_canary/suite/summary.json').read_text());assert new['errors']==0 and len(new['episodes'])==23
assert all(e['task_metrics']==old[(e['case'],e['seed'])]['task_metrics'] for e in new['episodes'])
atomic_json(r/'feedback_disabled_parity.json',dict(passed=True,episodes=23,metrics='bit-identical to frozen pre-change runtime'))
atomic_json(r/'feedback_progress.json',dict(phase='enabled_full_suite',started_unix=time.time()))
subprocess.run([*base,'--out',str(r/'native_feedback'),'--control',str(r/'feedback_control.json')],check=True,timeout=420)
for script in ['feedback_verify.py','compare_feedback.py']:subprocess.run(['.venv/bin/python',str(r/script)],check=True,timeout=90)
atomic_json(r/'feedback_completed.json',dict(completed=True,finished_unix=time.time()))
