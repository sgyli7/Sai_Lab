"""Predeclared R0 comparison. No automatic promotion or unbounded retries."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from sim2sim.standalone.candidate import evaluate
from sim2sim.research.queue import atomic_json
from sim2sim.research.budget import remaining
root=Path.cwd(); session=root/'results/jolt_learning_20260911'
recipe=json.loads((session/'r0_recipe.json').read_text()); completed=[]
for seed in recipe['training_seeds']:
    for objective in recipe['objectives']:
        if remaining(session,reserve=5400)<1500:
            atomic_json(session/'r0_budget_stop.json',dict(seed=seed,objective=objective));break
        name=f'r0_{objective}_s{seed}';log=session/(name+'.log')
        started=time.time();result=dict(name=name,seed=seed,objective=objective,started_unix=started)
        command=[sys.executable,'-m','sim2sim.research.train','--name',name,*recipe['fixed_training_arguments'],'--seed',str(seed),'--roller-objective',objective]
        result['command']=command
        try:
            with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=1350,check=True)
            run=session/'runs'/name;done=json.loads((run/'completed.json').read_text())
            if done['samples']!=recipe['new_samples_per_completed_run'] or not done['final_parity']['passed']:
                raise RuntimeError('Incomplete sample budget or export parity failure')
            result['training']=done
            out=session/'candidate_evaluations'/name
            with (session/(name+'_evaluation.log')).open('w') as f:
                # Native suite prints are kept in the supervisor log, training stays separate.
                evaluate(root/'results/research_20260911/delivery/candidate_models','roller',run/'final.onnx',session/'development_cases',out,session/'baseline_candidate/summary.json',workers=8,control_config=root/'results/research_20260911/final_candidate/control.json',base_project=session/'runtime_r0')
            cmp=json.loads((out/'comparison.json').read_text());result.update(completed=True,comparison=str(out/'comparison.json'),eligible=cmp['eligible_for_quality_review'],passes=sum(v['brake_passes_after'] for v in cmp['cases'].values()),baseline_passes=sum(v['brake_passes_before'] for v in cmp['cases'].values()),regressions=len(cmp['regressions']))
        except Exception:
            result.update(completed=False,error=traceback.format_exc())
        result['elapsed_s']=time.time()-started;completed.append(result)
        atomic_json(session/'r0_progress.json',completed);print('R0_RESULT '+json.dumps(result),flush=True)
        if not result['completed']:
            # Investigate the first infrastructure/numeric failure before consuming further runs.
            raise SystemExit(1)
atomic_json(session/'r0_completed.json',completed)
