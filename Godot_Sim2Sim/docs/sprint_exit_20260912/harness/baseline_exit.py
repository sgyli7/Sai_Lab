from pathlib import Path
import json
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'baseline_exits';out.mkdir(exist_ok=False)
control=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())
cases=write_cases(out/'cases',range(923000,923016),.3,selected=['left','right'],control=control)
s=run(cases,out/'suite',workers=4,executable='dist/MicroDuck-ARM64-20260912-joint-trial/MicroDuck.arm64')
failures=[dict(case=x['case'],seed=x['seed'],metrics=x['task_metrics'],trace=x['trace'],case_path=x['case_path']) for x in s['episodes'] if not x['task_metrics']['success']]
result=dict(count=len(cases),errors=s['errors'],passed=len(cases)-len(failures),failures=failures)
atomic_json(out/'completed.json',result);print(result,flush=True)
