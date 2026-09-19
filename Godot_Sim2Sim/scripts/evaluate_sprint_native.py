"""Freeze and evaluate an experimental actor without touching the desktop player."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import onnxruntime as ort

from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run


def install_actor(project, source, control):
    assets=project/'runtime_assets';path=assets/'deployment.json';deployment=json.loads(path.read_text())
    old=deployment['policies']['sprint'];target=project/old['path'].removeprefix('res://')
    if target.is_symlink():raise ValueError('Experimental model destination must be owned')
    shutil.copyfile(source,target);sha=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest=json.loads(source.with_suffix('.manifest.json').read_text())
    manifest.update(sha256=sha,status='experimental_forward_gait_not_promoted')
    manifest.setdefault('sim2sim',{}).update(skill='sprint',use_stand_policy=False)
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    deployment['policies']['sprint']={**old,'sha256':sha,'manifest':manifest};deployment['control_config']=control
    path.write_text(json.dumps(deployment,indent=2)+'\n')
    fixture_path=assets/'self_test.json';fixture=json.loads(fixture_path.read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    actor=ort.InferenceSession(str(target),opts,providers=['CPUExecutionProvider'])
    for item in fixture['cases']:
        if item['skill']!='sprint':continue
        item['sha256']=sha
        item['actions']=[actor.run(None,{actor.get_inputs()[0].name:np.array(x,np.float32)[None]})[0][0].tolist() for x in item['observations']]
        item.update(real_count=0,real_trace_sha256={},observation_provenance='Frozen fixture inputs reused, expected actions recalculated for this experimental actor')
    fixture_path.write_text(json.dumps(fixture)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--speed',type=float,required=True);p.add_argument('--seed-start',type=int,default=929100);p.add_argument('--seeds',type=int,default=4);p.add_argument('--paired',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    project=a.output/'prepared';subprocess.run(['cp','--reflink=auto','-a',str(a.project),str(project)],check=True,timeout=60)
    control=json.loads(a.control.read_text());control['walk']['twist_limits']['sprint_vmax_x']=a.speed
    control['walk']['twist_limits']['sprint_vmax_ang']=.8
    install_actor(project,a.actor,control)
    check=subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(project),'res://standalone/main.tscn','--','--self-test'],capture_output=True,text=True,timeout=40)
    (a.output/'self_test.log').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('Native inference self-test failed')
    lines=[line.split('STANDALONE_SELF_TEST ',1)[1] for line in check.stdout.splitlines() if 'STANDALONE_SELF_TEST ' in line]
    if len(lines)!=1 or not json.loads(lines[0])['passed']:raise RuntimeError('Missing native self-test pass')
    (a.output/'self_test.json').write_text(json.dumps(json.loads(lines[0]),indent=2)+'\n')
    cases=write_cases(a.output/'cases',range(a.seed_start,a.seed_start+a.seeds),a.speed,paired=a.paired,control=control)
    summary=run(cases,a.output/'suite',workers=2,project=project)
    episodes=summary['episodes'];candidate=[e for e in episodes if '_ordinary' not in e['case']]
    result=dict(completed=True,errors=summary['errors'],count=len(candidate),passes=sum(e.get('task_metrics',{}).get('success',False) for e in candidate),falls=sum(e.get('task_metrics',{}).get('fell',False) for e in candidate),long_speed=[e['task_metrics']['sustained_mean_vx'] for e in candidate if e['case']=='sprint_long'],models=summary['models'],failed=[dict(case=e['case'],seed=e.get('seed'),metrics=e.get('task_metrics'),error=e.get('error')) for e in candidate if not e.get('task_metrics',{}).get('success',False)],ordinary_pass=sum(e.get('task_metrics',{}).get('success',False) for e in episodes if '_ordinary' in e['case']))
    (a.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['passes','count','falls','errors','long_speed']})
    if summary['errors']:raise RuntimeError('Native cases failed to execute; preserve failed attempts')


if __name__=='__main__':main()
