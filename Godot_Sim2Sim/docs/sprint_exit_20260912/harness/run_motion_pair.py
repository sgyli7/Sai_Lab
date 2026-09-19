import json,subprocess,time
from pathlib import Path
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912')
assert json.loads((r/'state_contract/completed.json').read_text())['anchor_max_abs']==0.
assert json.loads((r/'state_shadow/result.json').read_text())['passed']
protocol=json.loads((r/'motion_pair_protocol.json').read_text())
for command in protocol['commands']:
 name=command[command.index('--name')+1]
 atomic_json(r/'motion_pair_progress.json',dict(stage='training',name=name,started_unix=time.time()))
 subprocess.run(command,check=True,timeout=1200)
 complete=json.loads((r/'runs'/name/'completed.json').read_text())
 assert complete['iterations']==96 and complete['status']=='iteration_limit'
 assert complete['final_parity']['passed']
 atomic_json(r/'motion_pair_progress.json',dict(stage='native',name=name,started_unix=time.time()))
 subprocess.run(['.venv/bin/python',str(r/'motion_native.py'),'--actor',str(r/'runs'/name/'final.onnx'),'--out',str(r/(name+'_native'))],check=True,timeout=420)
atomic_json(r/'motion_pair_completed.json',dict(completed=True,finished_unix=time.time()))
