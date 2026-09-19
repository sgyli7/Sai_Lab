from pathlib import Path
import json
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'baseline_all';out.mkdir(exist_ok=False)
control=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())
cases=write_cases(out/'cases',range(923000,923016),.3,paired=True,control=control)
s=run(cases,out/'suite',workers=4,executable='dist/MicroDuck-ARM64-20260912-joint-trial/MicroDuck.arm64')
accept=paired_acceptance(s,cases);atomic_json(out/'paired_acceptance.json',accept)
c=[e for e in s['episodes'] if not e['case'].endswith('_ordinary')]
result=dict(errors=s['errors'],candidate_count=len(c),candidate_pass=sum(e['task_metrics']['success'] for e in c),candidate_falls=sum(e['task_metrics']['fell'] for e in c),accepted=accept['accepted'],models=s['models'],comparisons=[{k:v for k,v in x.items() if k!='pairs'} for x in accept['comparisons']])
atomic_json(out/'completed.json',result);print(result,flush=True)
