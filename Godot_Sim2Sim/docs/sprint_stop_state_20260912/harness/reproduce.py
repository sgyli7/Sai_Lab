from pathlib import Path
import json
from sim2sim.standalone.suite import run_case
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912').resolve();source=json.loads((r/'source.json').read_text())['episode'];project=Path('results/sprint_joint_gpu_20260912/native_trained_tracking_split/suite/runtime').resolve()
rows=[]
for i in range(3):
 e=run_case(r/f'reproduce_{i}.json',r/'reproduce',source['models'],project=project,timeout=30)
 assert e['completed'],e
 assert not e['task_metrics']['success'] and not e['task_metrics']['fell'],e['task_metrics']
 assert e['task_metrics']['stops']==source['task_metrics']['stops'],e['task_metrics']
 rows.append(e)
atomic_json(r/'reproduce_completed.json',dict(completed=True,red_capable=True,reproduced=3,expected_failure='No sustained stop after9s release',episodes=rows))
print('RED reproduced3/3: stop onset=None, no falls')
