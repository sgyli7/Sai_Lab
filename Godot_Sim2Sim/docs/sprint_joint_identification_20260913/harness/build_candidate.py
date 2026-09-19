from pathlib import Path
import json,subprocess
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.suite import runtime_inputs
from sim2sim.standalone.package import package
r=Path('results/sprint_joint_identification_20260913');f=json.loads((r/'candidate_freeze.json').read_text());source=r/'integrated_dev/suite/runtime';p=r/'export_project'
assert runtime_inputs(source)==f['runtime_inputs']
assert not p.exists()
subprocess.run(['cp','--reflink=auto','-a',str(source),str(p)],check=True,timeout=60)
result=package(Path('dist/MicroDuck-ARM64-20260913-sprint-reversal-trial'),project=p)
assert result['models']==f['models']
assert runtime_inputs(source)==f['runtime_inputs']
atomic_json(r/'package_completed.json',dict(completed=True,**result))
print(result,flush=True)
