import json,subprocess,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sim2sim.research.world import World
from sim2sim.research.models import NativeAnchor
from sim2sim.research.sprint_tasks import commands
from sim2sim.research.sprint_composition import SprintComposition
from sim2sim.standalone.replay import raw_state
from sim2sim.motion_control import MotionControl
root=Path('results/sprint_exit_20260912');out=root/'state_shadow';out.mkdir(exist_ok=False)
project=root/'state_contract/suite/runtime';case=out/'case.json';trace=(out/'trace.json').resolve()
config=json.loads((Path('results/sprint_joint_20260912/delivery/control.json')).read_text());config['walk']['twist_limits'].update(sprint_vmax_x=.3,sprint_vmax_ang=.8)
from sim2sim.standalone.sprint import templates
program=templates()['right'];program['control_config']=config
for segment in program['segments']:segment['order']=sorted(segment['held'])
case.write_text(json.dumps(program))
with (out/'native.log').open('w') as log:
 subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(project),'res://standalone/main.tscn','--','--replay='+str(case.resolve()),'--trace='+str(trace)],stdout=log,stderr=subprocess.STDOUT,timeout=35,check=True)
data=json.loads(trace.read_text());deployment=json.loads((project/'runtime_assets/deployment.json').read_text());robot=deployment['robots']['walk']
world=object.__new__(World);world.motion=MotionControl(config['walk']);world._command_stamp=None;world.sprint_composed=True
world.command_tape,world.sprint_selection=commands('sprint_030_nativeright',seconds=14,include_selection=True,twist_limits=config['walk']['twist_limits'])
world.home=np.array(robot['home'],np.float32);world.state_input='planar_com_velocity_height_v1';world.task_state=None;world.yaw_memory=None
learner=NativeAnchor(root/'state_source/anchor.onnx');controller=SprintComposition('results/sprint_20260912/runs/s05_native_handoff/final.onnx',learn_all=True)
assert data['summary']['models']['sprint']==learner.sha256
assert data['summary']['models']['walking']==learner.sha256
errors={key:0. for key in ['observation','command','action']};mismatches=[]
for index,row in enumerate(data['rows']):
 world.t=row['t'];world.state=raw_state(row['raw'],robot);world.last=np.array(row['last_action'],np.float32)
 obs=world.obs();proposed=learner(obs[None]);action=controller.actions([world],proposed)[0]
 if world.sprint_active()!=(row['skill']=='sprint'):mismatches.append(index)
 for key,a,b in [('observation',obs,row['obs']),('command',world.command(),row['command']),('action',action,row['action'])]:
  errors[key]=max(errors[key],float(np.max(np.abs(np.asarray(a)-b))))
result=dict(check='training_composition_on_real_native_trace',rows=len(data['rows']),maximum=errors,phase_mismatches=mismatches,controller_steps=controller.counts,models=data['summary']['models'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/sim2sim/research/world.py'),Path('src/sim2sim/research/sprint_tasks.py'),Path('src/sim2sim/research/sprint_composition.py')]},passed=not mismatches and max(errors.values())<1e-5)
(out/'result.json').write_text(json.dumps(result,indent=2));print(result);assert result['passed']
