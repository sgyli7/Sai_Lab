"""Prepare the existing nine-model MicroDuck bundle for the workshop native player."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

from sim2sim.paths import sim2sim_root, load_robot_json
from sim2sim.play import capture_home_poses, ensure_godot_scene, ROLLER_LIMITS
from sim2sim.policy import OnnxPolicy
from sim2sim.research.tasks import TASKS


def prepare_ui_font(destination):
    source=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    license_path=Path('/usr/share/doc/fonts-noto-cjk/copyright')
    if not source.is_file() or not license_path.is_file():
        raise FileNotFoundError('Install the Ubuntu fonts-noto-cjk build dependency before preparing the player')
    directory=destination/'fonts';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'ui-font.bin'
    shutil.copyfile(source,target)
    shutil.copyfile(license_path,directory/'LICENSE.txt')
    return dict(path='res://runtime_assets/fonts/ui-font.bin',face_index=2,
        family='Noto Sans CJK SC',sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        license='res://runtime_assets/fonts/LICENSE.txt',source=str(source))


def prepare(models, *, fixtures=None, fixture_count=128, real_traces=(), control_config=None, project=None, sprint=None):
    root = sim2sim_root()
    destination = (Path(project) if project else root/'godot')/'runtime_assets'
    (destination/'policies').mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, runtime='onnxruntime', runtime_version='1.29.0',
                  physics_hz=200, decimation=4, policies={}, robots={})
    sources = {skill:Path(models)/task.previous for skill,task in TASKS.items()}
    if sprint is not None:sources['sprint']=Path(sprint)
    for skill, source in sources.items():
        manifest_path = source.with_suffix('.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        policy = OnnxPolicy(source)
        policy.check_dims(14)
        if policy.obs_dim != 61:
            raise ValueError("Workshop preparation currently accepts the verified 61D nine-model bundle")
        policy.state_input = getattr(policy, "state_input", "")
        policy.task_input = getattr(policy, "task_input", "")
        if skill=='sprint' and (policy.obs_dim!=61 or policy.time_input_s or policy.heading_input or policy.yaw_memory_input or policy.task_input):
            raise ValueError('Sprint requires the walking observation contract')
        if policy.state_input and (skill not in ('walking','sprint','roller') or policy.time_input_s or policy.heading_input or policy.yaw_memory_input):
            raise ValueError('State input requires a walking, sprint or native roller actor')
        if policy.task_input and (skill!='roller' or not policy.state_input):
            raise ValueError('Task input requires the roller velocity-state actor')
        target = destination/'policies'/('Sprint_Godot.onnx' if skill=='sprint' else source.name)
        shutil.copyfile(source, target)
        shutil.copyfile(manifest_path, target.with_suffix('.manifest.json'))
        result['policies'][skill] = dict(path='res://runtime_assets/policies/'+target.name,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), manifest=manifest,
            time_input_s=policy.time_input_s, heading_input=policy.heading_input,state_input=policy.state_input,
            task_input=policy.task_input,obs_dim=policy.obs_dim)
    for mode, name in [('walk','microduck_ball_stand_fix'),('roller','microduck_roller')]:
        cfg = load_robot_json(root/'robots'/f'{name}.json')
        ensure_godot_scene(cfg)
        spec_path = Path(cfg['godot_spec'])
        spec = json.loads(spec_path.read_text())
        base = next(b for b in spec['bodies'] if b['name']==cfg.get('base_body','trunk_base'))
        result['robots'][mode] = dict(
            scene='res://'+spec_path.with_name('robot.tscn').relative_to(root/'godot').as_posix(),
            spec='res://'+spec_path.relative_to(root/'godot').as_posix(),
            home=np.asarray(cfg['home'],dtype=np.float32).tolist(),
            action_scale=cfg['action_scale'], timestep=cfg.get('timestep',.005),
            decimation=cfg.get('decimation',4), current_limit_a=cfg.get('current_limit_a',1.75),
            base_body=cfg.get('base_body','trunk_base'), ipos=base['ipos'], iquat=base['iquat_wxyz'],
            poses=capture_home_poses(cfg),
            spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest())
    result['roller_limits'] = vars(ROLLER_LIMITS)
    result['control_config']={} if control_config is None else json.loads(Path(control_config).read_text())
    result['ui_font']=prepare_ui_font(destination)
    (destination/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
    if fixture_count:
        rng=np.random.default_rng(20260911)
        real={skill:[] for skill in result['policies']}
        real_sources={skill:{} for skill in result['policies']}
        for trace in real_traces:
            trace_path=Path(trace).resolve();trace_bytes=trace_path.read_bytes()
            trace_hash=hashlib.sha256(trace_bytes).hexdigest()
            payload=json.loads(trace_bytes)
            hashes=payload['summary']['models']
            for row in payload['rows']:
                skill=row['skill']
                if skill in real and hashes.get(skill)==result['policies'][skill]['sha256']:
                    real[skill].append(row['obs'])
                    real_sources[skill][str(trace_path)]=trace_hash
        cases=[]
        for skill, item in result['policies'].items():
            source=destination/'policies'/Path(item['path']).name
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            policy=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
            width=item['obs_dim']
            observations=rng.normal(0,.5,(fixture_count,width)).astype(np.float32)
            observations[0]=0;observations[:,3:6]=[0,0,-1]
            if item['time_input_s']:
                observations[:,48]=np.linspace(0,1,fixture_count)
            boundaries=[]
            commands=[-.5,-.05,-1e-6,0.,1e-6,.05,.3,.6]
            if item['time_input_s']:
                commands=[0.,1e-7,.5,1.-1e-7,1.]
            headings=[-np.pi,-np.pi+1e-7,-np.pi/2,0.,np.pi/2,np.pi-1e-7,np.pi] if item['heading_input'] else [None]
            for command in commands:
                for angle in headings:
                    boundary=np.zeros(width,np.float32);boundary[5]=-1.;boundary[48]=command
                    if angle is not None:boundary[49:51]=[np.sin(angle),np.cos(angle)]
                    boundaries.append(boundary)
            observations=np.concatenate([observations,np.array(boundaries,np.float32)])
            real_count=0
            if real[skill]:
                selected=np.linspace(0,len(real[skill])-1,min(64,len(real[skill])),dtype=int)
                observations=np.concatenate([observations,np.array(real[skill],np.float32)[selected]])
                real_count=len(selected)
            outputs=[policy.run(None, {policy.get_inputs()[0].name:x[None]})[0][0].tolist() for x in observations]
            cases.append(dict(skill=skill,path=item['path'],sha256=item['sha256'],
                              observations=observations.tolist(),actions=outputs,
                              random_count=fixture_count,boundary_count=len(boundaries),real_count=real_count,
                              real_trace_sha256=real_sources[skill],real_input_selection='matching model SHA only'))
        payload=json.dumps(dict(schema_version=1,cases=cases))
        (destination/'self_test.json').write_text(payload)
        if fixtures:
            path=Path(fixtures);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(payload)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--sprint',type=Path,help='Explicit optional trained walking sprint actor with manifest')
    p.add_argument('--fixtures',type=Path)
    p.add_argument('--fixture-count',type=int,default=128)
    p.add_argument('--real-traces',type=Path,nargs='*',default=[])
    p.add_argument('--control-config',type=Path)
    p.add_argument('--project',type=Path,help='Prepare an isolated copy of the Godot project')
    a=p.parse_args();r=prepare(a.models,fixtures=a.fixtures,fixture_count=a.fixture_count,real_traces=a.real_traces,control_config=a.control_config,project=a.project,sprint=a.sprint)
    if a.sprint is None and a.control_config is None:
        from sim2sim.default_sprint import apply_default_sprint
        project = a.project or sim2sim_root()/"godot"
        apply_default_sprint(project)
        r = json.loads((project/"runtime_assets/deployment.json").read_text())
    print(json.dumps(dict(policies=len(r['policies']),robots=list(r['robots']))))


if __name__=='__main__':main()
