"""Success-only frozen-teacher occupancy; separate training seeds, no held-out starts."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from sim2sim.standalone.cases import write_cases
from sim2sim.standalone.suite import run
p=Path('results/jolt_learning_20260911').resolve();names=['roller_brake_1s','roller_brake_2s','roller_brake_3s','roller_brake_5s','roller_turn_brake','roller_repeated_brake','roller_crouch_brake']
seeds=list(range(71010000,71010016))
recipe=dict(registered_unix=time.time(),seeds=seeds,cases=names,teacher='Frozen feedback candidate with zero-increment 68D wrapper',inclusion='Whole episode must pass unchanged active-brake scorer; retain only roller negative-throttle pre-action observations.',use='Training-only teacher regularization; never final acceptance',phase_sampling='Three bins by brake elapsed: <1s, 1–2s, >=2s, sampled equally in regularizer')
(p/'teacher_replay_recipe.json').write_text(json.dumps(recipe,indent=2))
files=write_cases(p/'teacher_training_cases',seeds,names)
s=run(files,p/'teacher_training_suite',workers=4,executable=p/'contract_export_package/MicroDuck.arm64')
if s['errors']:raise RuntimeError('Teacher data generation incomplete')
obs=[];groups=[];episodes=[]
for index,e in enumerate(s['episodes']):
 if not e['brake_success']:continue
 rows=json.loads(Path(e['trace']).read_text())['rows'];count=0
 for r in rows:
  if r['skill']!='roller' or r['command'][0]>=-.01:continue
  x=np.asarray(r['obs'],np.float32)
  if x.shape!=(68,) or not np.isfinite(x).all():raise RuntimeError('Invalid teacher task observation')
  obs.append(x);groups.append(0 if x[61]<1 else (1 if x[61]<2 else 2));episodes.append(index);count+=1
 if not count:raise RuntimeError('Successful teacher episode has no braking observations')
obs=np.stack(obs);groups=np.asarray(groups,np.int64)
if len(set(groups))!=3:raise RuntimeError('Teacher replay lacks a task phase')
path=p/'teacher_replay.npz';np.savez_compressed(path,obs=obs,phase=groups,episode=np.asarray(episodes,np.int64))
report=dict(data=path.name,anchor_sha256=hashlib.sha256((p/'task_anchor.onnx').read_bytes()).hexdigest(),recipe=recipe,source_summary=str(p/'teacher_training_suite/summary.json'),teacher_sha256=s['models']['roller'],episodes=len(s['episodes']),successful_episodes=sum(e['brake_success'] for e in s['episodes']),frames=len(obs),phase_counts=np.bincount(groups).tolist(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(p/'teacher_replay_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
