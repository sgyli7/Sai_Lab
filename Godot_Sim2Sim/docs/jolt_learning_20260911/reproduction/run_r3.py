"""Paired teacher-retention experiment with fixed data and final-only selection."""
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys,time,traceback
from pathlib import Path
from sim2sim.research.budget import remaining
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.candidate import evaluate
root=Path.cwd();p=root/'results/jolt_learning_20260911';recipe=json.loads((p/'r3_recipe.json').read_text())
start_wait=time.monotonic()
while not (p/recipe['wait_for']).exists():
 if time.monotonic()-start_wait>11000:raise RuntimeError('Preceding paired experiments did not finish')
 time.sleep(5)
def train(seed,mode):
 item=dict(name=f'r3_{mode}_s{seed}_1m',seed=seed,mode=mode,started_unix=time.time(),segments=[])
 parent=None
 try:
  for updates in recipe['segments']:
   name=f'r3_{mode}_s{seed}_'+('prefix' if parent is None else '1m')
   args=list(recipe['fixed_training_arguments']);args[args.index('--iterations')+1]=str(updates);args[args.index('--minutes')+1]='35'
   command=[sys.executable,'-m','sim2sim.research.train','--name',name,*args,'--seed',str(seed),'--teacher-mode',mode]
   if mode=='replay_kl':command+=['--teacher-replay',recipe['teacher_replay']]
   if parent is not None:command+=['--resume',str(p/'runs'/parent/'latest.pt')]
   segment=dict(name=name,command=command)
   with (p/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=2250,check=True)
   done=json.loads((p/'runs'/name/'completed.json').read_text())
   if done['samples']!=updates*4096 or not done['final_parity']['passed']:raise RuntimeError('Incomplete segment or numerical export failure')
   segment['training']=done;item['segments'].append(segment);atomic_json(p/(item['name']+'_training.json'),item);parent=name
  item.update(training=done,cumulative_samples=1048576,training_complete=True)
 except Exception:item.update(completed=False,error=traceback.format_exc())
 atomic_json(p/(item['name']+'_training.json'),item);return item
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for seed in recipe['training_seeds']:
  if remaining(p,reserve=5400)<3400:raise RuntimeError('Reserved closeout prevents starting another complete pair')
  pair=[pool.submit(train,seed,mode) for mode in recipe['teacher_modes']];pair=[f.result() for f in pair]
  for item in pair:
   if item.get('training_complete'):
    try:
     out=p/'candidate_evaluations'/item['name']
     evaluate(root/'results/research_20260911/delivery/candidate_models','roller',p/'runs'/item['name']/'final.onnx',p/'development_cases',out,p/'baseline_candidate/summary.json',workers=8,control_config=root/'results/research_20260911/final_candidate/control.json',base_project=p/'runtime_r1')
     cmp=json.loads((out/'comparison.json').read_text())
     item.update(completed=True,comparison=str(out/'comparison.json'),eligible=cmp['eligible_for_quality_review'],passes=sum(c['brake_passes_after'] for c in cmp['cases'].values()),baseline_passes=51,regressions=len(cmp['regressions']))
    except Exception:item.update(completed=False,error=traceback.format_exc())
   item['elapsed_s']=time.time()-item['started_unix'];rows.append(item);atomic_json(p/'r3_progress.json',rows);print('R3_RESULT '+json.dumps(item),flush=True)
  if not all(x.get('completed') for x in pair):raise SystemExit(1)
atomic_json(p/'r3_completed.json',rows)
