from pathlib import Path
import json,subprocess,time
R=Path(__file__).resolve().parent
I=Path('/home/ethan/Projects/MicroDuck-SpeedControls')
O=Path('/home/ethan/Projects/MicroDuck/sim2sim')
training=json.loads((R/'training_completed.json').read_text());assert len(training)==4
actors=[('a402',I/'src/sim2sim/assets/microduck_sprint_v1/Sprint_Godot.onnx')]+[(x['label'],R/x['label']/'final.onnx') for x in training]
summary=O/'results/sprint_input_diagnosis_20260914/ceiling_gate/baseline/suite/summary.json'
records=[]
for label,actor in actors:
 args=['env',f'PYTHONPATH={I}/src',f'SIM2SIM_ROOT={O}','/home/ethan/Projects/microduck_rl/.venv/bin/python',str(I/'scripts/evaluate_sprint_gpu.py'),'--native-summary',str(summary),'--actor',str(actor),'--ordinary',str(I/'src/sim2sim/assets/microduck_sprint_v1/Walk_Godot.onnx'),'--control',str(R/'control.json'),'--output',str(R/('gpu_'+label))]
 with (R/('gpu_'+label+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
 done=json.loads((R/('gpu_'+label)/'completed.json').read_text());records.append(dict(label=label,domain='gpu',passes=done['passes'],count=done['count'],falls=done['falls']));print(records[-1],flush=True)
 (R/'evaluation_progress.json').write_text(json.dumps(records,indent=2)+'\n')
for label,actor,speed in [('default030',actors[0][1],.3)]+[(label,actor,.45) for label,actor in actors]:
 args=['env',f'PYTHONPATH={I}/src',f'SIM2SIM_ROOT={O}',str(O/'.venv/bin/python'),str(I/'scripts/evaluate_sprint_native.py'),'--actor',str(actor),'--project',str(R/'runtime'),'--control',str(R/'control.json'),'--output',str(R/('native_'+label)),'--speed',str(speed),'--seeds','4']
 with (R/('native_'+label+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)
 done=json.loads((R/('native_'+label)/'completed.json').read_text());records.append(dict(label=label,domain='native',passes=done['passes'],count=done['count'],falls=done['falls'],long_speed=done['long_speed']));print(records[-1],flush=True)
 (R/'evaluation_progress.json').write_text(json.dumps(records,indent=2)+'\n')
(R/'evaluation_completed.json').write_text(json.dumps(records,indent=2)+'\n')
