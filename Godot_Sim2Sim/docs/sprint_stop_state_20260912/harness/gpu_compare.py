from pathlib import Path
import json,subprocess
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912');actor=r/'train_joint_fd/final.onnx';ordinary=Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx');result={}
for label,model,observer in [('new_observer',actor,'joint_fd'),('new_actor_old_observer',actor,'solver'),('old_actor_new_observer',Path('results/sprint_joint_gpu_20260912/train_tracking_split/final.onnx'),'joint_fd')]:
 cmd=['env','PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(r/'gpu_native_starts.py'),'--actor',str(model),'--ordinary',str(ordinary),'--feet','jolt','--velocity-observer',observer,'--out',str(r/('gpu_'+label))]
 subprocess.run(cmd,check=True,timeout=110);x=json.loads((r/('gpu_'+label)/'completed.json').read_text());rows=x.pop('results');x['ordinary_pass']=sum(e['metrics']['success'] for e in rows if e['ordinary']);x['native_failure_case']=next(e for e in rows if e['case']=='sprint_alternate' and e['seed']==927001 and not e['ordinary']);result[label]=x;print(label,x['passes'],x['ordinary_pass'],flush=True)
atomic_json(r/'gpu_comparison.json',dict(completed=True,arms=result,final_seeds_used=False))
