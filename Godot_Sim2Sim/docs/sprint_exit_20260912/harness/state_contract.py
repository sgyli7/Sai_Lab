from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np
from sim2sim.policy_state import wrap_anchor,BRAKE_STATE_V1
from sim2sim.research.models import NativeAnchor
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.suite import run

r=Path('results/sprint_exit_20260912');out=r/'state_contract';out.mkdir(exist_ok=False)
source=Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')
anchor=wrap_anchor(source,r/'state_source/anchor.onnx')
shutil.copy2(source.with_suffix('.manifest.json'),anchor.with_suffix('.manifest.json'))
models=out/'models';shutil.copytree('results/research_20260911/delivery/candidate_models',models)
for name in ['Walk_Godot.onnx','Sprint_Godot.onnx']:
 shutil.copy2(anchor,models/name);shutil.copy2(anchor.with_suffix('.manifest.json'),(models/name).with_suffix('.manifest.json'))
project=out/'prepared';subprocess.run(['cp','--reflink=auto','-a','godot',str(project)],check=True,timeout=60)
prepare(models,project=project,sprint=models/'Sprint_Godot.onnx',control_config='results/sprint_joint_20260912/delivery/control.json')
cases=[r/'minimize/remove_forward_prefix.json',*sorted((r/'baseline_exits/cases').glob('*.json'))]
s=run(cases,out/'suite',workers=4,project=project)
actual=NativeAnchor(anchor);original=NativeAnchor(source)
max_anchor=0.
for episode in s['episodes']:
 rows=json.loads(Path(episode['trace']).read_text())['rows']
 x=np.asarray([row['obs'] for row in rows],np.float32);legacy=x.copy();legacy[:,58:61]=0
 max_anchor=max(max_anchor,float(np.abs(actual(x)-original(legacy)).max()))
runtime=out/'suite/runtime'
with (out/'selftest.log').open('w') as log:
 subprocess.run(['godot','--headless','--path',str(runtime),'res://standalone/main.tscn','--','--self-test'],stdout=log,stderr=subprocess.STDOUT,timeout=35,check=True)
result=dict(errors=s['errors'],count=len(s['episodes']),passed=sum(e['task_metrics']['success'] for e in s['episodes']),anchor_max_abs=max_anchor,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),wrapped_sha256=actual.sha256)
atomic_json(out/'completed.json',result);print(result,flush=True)
assert not s['errors'] and max_anchor==0.
