"""Source-contract fast policy probe; physics remains the actual frozen Jolt game."""
from pathlib import Path
import copy,hashlib,json,shutil,subprocess
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.suite import run
from sim2sim.standalone.cases import randomized_poses
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'running_zero_shot';out.mkdir(exist_ok=False)
source=r/'running_source/policy.onnx';models=out/'models';shutil.copytree('results/sprint_joint_20260912/delivery/models',models)
actor=models/'Sprint_Godot.onnx';shutil.copy2(source,actor)
manifest=json.loads((r/'running_source/manifest.json').read_text());manifest['sim2sim']=dict(skill='sprint',use_stand_policy=False);manifest['sha256']=hashlib.sha256(actor.read_bytes()).hexdigest();atomic_json(actor.with_suffix('.manifest.json'),manifest)
project=out/'prepared';subprocess.run(['cp','--reflink=auto','-a','godot',str(project)],check=True,timeout=60)
control=dict(version='running_source_contract_probe',walk=dict(twist_limits=dict(vmax_x=.25,sprint_vmax_x=2.2,sprint_vmax_ang=.8)))
atomic_json(out/'control.json',control);prepare(models,project=project,sprint=actor,control_config=out/'control.json')
expected=[0,-.0873,-.4579,-.0049,.4530,.3491,.3491,0,0,0,.0873,.4579,.0049,-.4530]
d=json.loads((project/'runtime_assets/deployment.json').read_text());assert max(abs(a-b) for a,b in zip(expected,d['robots']['walk']['home']))<1e-6
cases=[]
for speed in [.3,2.2]:
 for seed in [923000,923001,923002,923003]:
  name=f'fast_source_{speed}_{seed}';case=dict(case=name,skill='walking',mode='walk',seconds=7,seed=seed,randomized_start=True,initial_poses=dict(walk=randomized_poses('walk',seed)),ordinary_control=False,protocol='walking_sprint_v1',pair_id=name,scoring=dict(start=0,end=7),segments=[dict(at=0,held=[]),dict(at=1,held=['sprint','fwd']),dict(at=4,held=[])],sprint_intervals=[[1,4]],control_config=copy.deepcopy(control))
  case['control_config']['walk']['twist_limits']['sprint_vmax_x']=speed
  p=out/(name+'.json');atomic_json(p,case);cases.append(p)
s=run(cases,out/'suite',workers=4,project=project)
rows=[dict(case=x['case'],seed=x['seed'],metrics=x['task_metrics'],trace=x['trace']) for x in s['episodes']]
atomic_json(out/'completed.json',dict(errors=s['errors'],models=s['models'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=rows))
print(rows,flush=True)
