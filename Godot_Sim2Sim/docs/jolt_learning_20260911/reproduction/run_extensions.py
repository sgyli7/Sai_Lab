"""Fixed 1M endpoints; train each matched pair concurrently, then evaluate."""
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys,time,traceback
from pathlib import Path
from sim2sim.research.budget import remaining
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.candidate import evaluate
root=Path.cwd();p=root/'results/jolt_learning_20260911';recipe=json.loads((p/'extension_recipe.json').read_text())
wait_started=time.monotonic()
while not (p/'r2_completed.json').exists():
 if time.monotonic()-wait_started>6000:raise RuntimeError('Preceding experiments did not finish; no extension started')
 time.sleep(5)
def train(seed,family,mode):
 parent=f'{family}_{mode}_s{seed}';name=parent+'_1m';started=time.time()
 arguments=json.loads((p/f'{family}_recipe.json').read_text())['fixed_training_arguments']
 arguments[arguments.index('--iterations')+1]='192';arguments[arguments.index('--minutes')+1]='35'
 command=[sys.executable,'-m','sim2sim.research.train','--name',name,*arguments,'--seed',str(seed),'--resume',str(p/'runs'/parent/'latest.pt')]
 item=dict(name=name,parent=parent,seed=seed,command=command,started_unix=started)
 try:
  with (p/(name+'.log')).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=2250,check=True)
  done=json.loads((p/'runs'/name/'completed.json').read_text())
  if done['samples']!=786432 or done['iterations']!=256 or not done['final_parity']['passed']:raise RuntimeError('Incomplete extension or parity failure')
  item.update(training=done,cumulative_samples=1048576,training_complete=True)
 except Exception:item.update(completed=False,error=traceback.format_exc())
 atomic_json(p/(name+'_training.json'),item);return item
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for seed in recipe['training_seeds']:
  if remaining(p,reserve=5400)<2550:raise RuntimeError('Insufficient pair plus evaluation budget')
  futures=[pool.submit(train,seed,family,mode) for family,mode in [('r1','full'),('r2','sagittal')]]
  pair=[f.result() for f in futures]
  # Never benchmark/evaluate while either member is still training.
  for item in pair:
   name=item['name']
   if item.get('training_complete'):
    try:
     out=p/'candidate_evaluations'/name
     evaluate(root/'results/research_20260911/delivery/candidate_models','roller',p/'runs'/name/'final.onnx',p/'development_cases',out,p/'baseline_candidate/summary.json',workers=8,control_config=root/'results/research_20260911/final_candidate/control.json',base_project=p/'runtime_r1')
     cmp=json.loads((out/'comparison.json').read_text())
     item.update(completed=True,comparison=str(out/'comparison.json'),eligible=cmp['eligible_for_quality_review'],passes=sum(c['brake_passes_after'] for c in cmp['cases'].values()),baseline_passes=51,regressions=len(cmp['regressions']))
    except Exception:item.update(completed=False,error=traceback.format_exc())
   item['elapsed_s']=time.time()-item['started_unix'];rows.append(item);atomic_json(p/'extension_progress.json',rows);print('EXTENSION_RESULT '+json.dumps(item),flush=True)
  if not all(x.get('completed') for x in pair):raise SystemExit(1)
atomic_json(p/'extension_completed.json',rows)
