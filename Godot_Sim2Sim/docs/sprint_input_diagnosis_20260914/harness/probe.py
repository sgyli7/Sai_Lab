"""Actual desktop workshop entry: paired native input and measured displacement."""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import time

ROOT = Path('/home/ethan/Projects/MicroDuck/sim2sim')
R = ROOT / 'results/sprint_input_diagnosis_20260914'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--equal-commands', action='store_true')
    parser.add_argument('--project',type=Path)
    args = parser.parse_args()
    out = R / args.label
    out.mkdir(exist_ok=False)
    project = args.project or R / 'runtime'
    if not project.exists():
        shutil.copytree(ROOT / 'results/workshop-hub/runtime', project)
    reports = {}
    deployment_path=project/'runtime_assets/deployment.json'
    original=deployment_path.read_text()
    cases=[('forward', ['W'],None), ('sprint', ['Shift', 'W'],None), ('backward', ['S'],None)]
    if args.equal_commands:cases += [('walk_forward_030',['W'],.3),('walk_backward_025',['S'],-.25),('sprint_forward_045',['Shift','W'],.45)]
    for name, keys, override in cases:
        deployment=json.loads(original)
        if override is not None:deployment['control_config']['walk']['twist_limits']['sprint_vmax_x' if 'Shift' in keys else 'vmax_x' if override>0 else 'vmin_x']=override
        deployment_path.write_text(json.dumps(deployment))
        case = out / name
        case.mkdir()
        events = [dict(at=.5, key=k, pressed=True, location=1 if k=='Shift' else 0) for k in keys]
        options = dict(robot='microduck', task='drive', drive_speed=.5, record=False,
                       output=str(case), plan=dict(seconds=4.5, events=events))
        (project/'hub/options.json').write_text(json.dumps(options))
        cmd = ['godot','--path',str(project),'res://hub/main.tscn','--headless','--fixed-fps','200']
        started = time.monotonic()
        with (case/'godot.log').open('w') as stream:
            child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'SIM2SIM_TRACE_INPUT':'1', 'SIM2SIM_VISUAL_STYLE':'legacy', 'MD_WORKSHOP_COLLISIONS':'1','MD_MODE':'hub'})
            try:
                child.wait(timeout=45)
                assert child.returncode == 0, (name, child.returncode)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=3)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
        data=json.loads((case/'native-0.json').read_text())
        rows=[v for v in data['rows'] if 1.5 <= v['episode_t'] <= 4.4]
        assert len(rows)>100, name
        a,b=rows[0],rows[-1]
        dx=b['body']['base_pos'][0]-a['body']['base_pos'][0]
        dy=b['body']['base_pos'][1]-a['body']['base_pos'][1]
        dt=b['episode_t']-a['episode_t']
        contacts=[]
        for row in rows:
            for body in row['raw'].get('body_states',[]):
                if body['name']=='ball': continue
                for c in body.get('contacts',[]):
                    if c['body'] not in ['Floor','Ground']:contacts.append([row['episode_t'],body['name'],c['body']])
        reports[name]=dict(skills=sorted(set(v['skill'] for v in rows)), held=sorted(set(k for v in rows for k in v['held'])),
            request_x=statistics.mean(v['requested_command'][0] for v in rows),
            command_x=statistics.mean(v['command'][0] for v in rows),
            velocity_x=dx/dt, lateral_velocity=dy/dt, distance_x=dx,
            first_fall=data['first_fall'],other_contacts=contacts[:10],wall_seconds=time.monotonic()-started)
    deployment_path.write_text(original)
    f,s,b=[reports[k] for k in ['forward','sprint','backward']]
    checks=dict(sprint_model_selected=s['skills']==['sprint'] and f['skills']==['walking'],
        sprint_command_faster=s['request_x']>f['request_x']*1.1,
        sprint_measurably_faster=s['velocity_x']>f['velocity_x']*1.1,
        backward_not_faster=abs(b['velocity_x'])<=f['velocity_x']*1.05)
    result=dict(cases=reports,checks=checks,passed=all(checks.values()),scope='Current desktop prepared workshop, identical seed/start, native InputEventKey delivery; steady 1.5–4.4 seconds. No model/control modifications.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
