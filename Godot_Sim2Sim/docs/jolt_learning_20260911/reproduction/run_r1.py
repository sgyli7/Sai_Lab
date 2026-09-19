"""Sequential matched-architecture task-state ablation; no automatic promotion."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from sim2sim.standalone.candidate import evaluate
from sim2sim.research.queue import atomic_json
from sim2sim.research.budget import remaining
root=Path.cwd();session=root/'results/jolt_learning_20260911'
recipe=json.loads((session/'r1_recipe.json').read_text())
# R0 already owns training resources; do not compete with its subsequent runs.
wait_started=time.monotonic()
while not (session/recipe['wait_for']).exists():
 if time.monotonic()-wait_started>3600:raise RuntimeError('R0 did not finish; R1 never started')
 time.sleep(5)
completed=[]
for seed in recipe['training_seeds']:
 for mode in recipe['actor_modes']:
  if remaining(session,reserve=5400)<1500:raise RuntimeError('Budget reserved for evaluation; no new trial')
  name=f'r1_{mode}_s{seed}';started=time.time()
  result=dict(name=name,seed=seed,mode=mode,started_unix=started)
  command=[sys.executable,'-m','sim2sim.research.train','--name',name,*recipe['fixed_training_arguments'],'--seed',str(seed)]
  if mode=='masked':command.append('--mask-task-state')
  result['command']=command
  try:
   with (session/(name+'.log')).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=1350,check=True)
   run=session/'runs'/name;done=json.loads((run/'completed.json').read_text())
   if done['samples']!=recipe['new_samples_per_completed_run'] or not done['final_parity']['passed']:raise RuntimeError('Incomplete sample budget or export parity failure')
   result['training']=done;out=session/'candidate_evaluations'/name
   evaluate(root/'results/research_20260911/delivery/candidate_models','roller',run/'final.onnx',session/'development_cases',out,session/'baseline_candidate/summary.json',workers=8,control_config=root/'results/research_20260911/final_candidate/control.json',base_project=session/'runtime_r1')
   cmp=json.loads((out/'comparison.json').read_text())
   result.update(completed=True,comparison=str(out/'comparison.json'),eligible=cmp['eligible_for_quality_review'],passes=sum(v['brake_passes_after'] for v in cmp['cases'].values()),baseline_passes=sum(v['brake_passes_before'] for v in cmp['cases'].values()),regressions=len(cmp['regressions']))
  except Exception:result.update(completed=False,error=traceback.format_exc())
  result['elapsed_s']=time.time()-started;completed.append(result)
  atomic_json(session/'r1_progress.json',completed);print('R1_RESULT '+json.dumps(result),flush=True)
  if not result['completed']:raise SystemExit(1)
atomic_json(session/'r1_completed.json',completed)
