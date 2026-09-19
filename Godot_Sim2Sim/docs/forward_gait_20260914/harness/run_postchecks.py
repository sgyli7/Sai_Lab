from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parent
base=json.loads((R/'training_base_args.json').read_text())
for label,iterations,steps,resume in [('warm_resume_start',5,64,False),('warm_resume_finish',6,64,True),('empty_actor_mask',1,1,False)]:
 args=base.copy()
 for key,value in [('--output',R/label),('--iterations',iterations),('--envs',64),('--steps',steps),('--seed',952499)]:args[args.index(key)+1]=str(value)
 if resume:
  i=args.index('--initialize-checkpoint');args[i:i+2]=['--resume',str(R/'warm_resume_start/latest.pt')]
 with (R/(label+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
 done=json.loads((R/label/'completed.json').read_text());assert done['parity_max_abs']<1e-5
 print(label,done,flush=True)
first=json.loads((R/'warm_resume_start/config.json').read_text());second=json.loads((R/'warm_resume_finish/config.json').read_text())
assert first['initialization']==second['initialization'] and first['initialization']
rows=[json.loads(x) for x in (R/'warm_resume_finish/metrics.jsonl').read_text().splitlines()]
assert [x['iteration'] for x in rows]==[6]
empty=json.loads((R/'empty_actor_mask/metrics.jsonl').read_text().splitlines()[0]);assert empty['residual']['sampled_actions']==0
(R/'postchecks.json').write_text(json.dumps(dict(passed=True,resume_iterations=[6],initialization_preserved=True,empty_mask_handled=True,not_optimization_candidates=True),indent=2)+'\n')
