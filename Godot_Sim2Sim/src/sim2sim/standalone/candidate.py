"""Stage a model candidate and compare identical development cases without promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from sim2sim.paths import sim2sim_root
from sim2sim.policy import OnnxPolicy
from sim2sim.research.queue import atomic_json
from sim2sim.research.tasks import TASKS
from .prepare import prepare
from .score import score
from .suite import run


def checksum(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(incumbent,overrides,output):
    if set(overrides)-set(TASKS):raise ValueError('Unknown skill override')
    incumbent=Path(incumbent).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);records={}
    for skill,task in TASKS.items():
        original=incumbent/task.previous
        source=Path(overrides.get(skill,original)).resolve()
        actor=OnnxPolicy(source);actor.check_dims(14)
        target=output/task.previous;shutil.copyfile(source,target)
        manifest=json.loads(original.with_suffix('.manifest.json').read_text())
        if skill in overrides:
            manifest['obs_len']=actor.obs_dim
            manifest['action_len']=actor.act_dim
            if actor.task_input:
                manifest['observation_extension']=actor.task_input
            else:
                manifest.pop('observation_extension',None)
            manifest['research']=dict(skill=skill,source=str(source),sha256=checksum(source),
                parent_sha256=checksum(original),status='unpromoted_candidate')
            manifest['training']=dict(source=str(source),parent_sha256=checksum(original),role='targeted_experiment')
            manifest['eval']=dict(status='pending paired native evaluation')
            manifest['description']='Unpromoted targeted optimization candidate: '+skill
        atomic_json(target.with_suffix('.manifest.json'),manifest)
        records[skill]=dict(source=str(source),sha256=checksum(target),changed=checksum(target)!=checksum(original))
    atomic_json(output/'bank.json',dict(status='unpromoted',models=records))
    return records


def rescore(summary_path,output):
    summary_path=Path(summary_path).resolve();source=json.loads(summary_path.read_text())
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);rows=[]
    for episode in source['episodes']:
        if not episode['completed']:raise ValueError('Cannot rescore an incomplete episode')
        case=episode.get('case_path') or next(arg.removeprefix('--replay=') for arg in episode['command'] if arg.startswith('--replay='))
        row=score(episode['trace'],case);row.update(completed=True,case_sha256=checksum(case),
            command=episode['command'],prior_task_protocol=episode['task_protocol'])
        atomic_json(output/(Path(case).stem+'.json'),row);rows.append(row)
    result=dict(models=source['models'],source_summary=str(summary_path),source_sha256=checksum(summary_path),
        reason='Recompute original trajectories after a declared telemetry/scoring correction; no rerun or parameter tuning',
        errors=0,episodes=rows)
    atomic_json(output/'summary.json',result);return result


def compare(baseline_path,candidate_path,output):
    baseline=json.loads(Path(baseline_path).read_text());candidate=json.loads(Path(candidate_path).read_text())
    lookup={(row['case'],row.get('seed')):row for row in baseline['episodes']}
    if not candidate['episodes']:raise ValueError('Candidate comparison has no episodes')
    pairs=[];regressions=[];groups={}
    for new in candidate['episodes']:
        key=(new['case'],new.get('seed'));old=lookup.get(key)
        if old is None:raise ValueError('Missing paired baseline: '+str(key))
        if not old['completed'] or not new['completed']:raise ValueError('An incomplete case cannot enter comparison')
        if old['case_sha256']!=new['case_sha256']:raise ValueError('Paired command/initial-pose inputs differ')
        if old['task_protocol']!=new['task_protocol']:raise ValueError('Rescore both sides with the same task protocol')
        if old['task_metrics']['success'] and not new['task_metrics']['success']:
            regressions.append(dict(case=key,reason='task_success_lost'))
        if old.get('brake_success') and not new.get('brake_success'):
            regressions.append(dict(case=key,reason='brake_success_lost'))
        if new['skill']!='roulade' and not old['task_metrics']['fell'] and new['task_metrics']['fell']:
            regressions.append(dict(case=key,reason='new_fall'))
        pairs.append((old,new));groups.setdefault(new['case'],[]).append((old,new))
    comparison={}
    for name,group in groups.items():
        metrics=set.intersection(*[set(old['task_metrics'])&set(new['task_metrics']) for old,new in group])
        numeric={key for key in metrics if all(isinstance(r['task_metrics'][key],(float,int,bool)) for pair in group for r in pair)}
        def means(side):
            return {key:float(np.mean([pair[side]['task_metrics'][key] for pair in group])) for key in sorted(numeric)}
        speed_key='moving_mean_speed' if all('moving_mean_speed' in r for pair in group for r in pair) else 'moving_mean_vx'
        old_speeds=[p[0][speed_key] for p in group if p[0][speed_key] is not None]
        new_speeds=[p[1][speed_key] for p in group if p[1][speed_key] is not None]
        ratio=float(np.mean(new_speeds)/np.mean(old_speeds)) if old_speeds and np.mean(old_speeds)>0 else None
        if group[0][0]['skill']=='walking' and ratio is not None and ratio<.95:
            regressions.append(dict(case=name,reason='walking_speed_below_95_percent',ratio=ratio))
        comparison[name]=dict(pairs=len(group),before=means(0),after=means(1),moving_speed_ratio=ratio,speed_metric=speed_key,
            brake_passes_before=sum(p[0].get('brake_success',False) for p in group),
            brake_passes_after=sum(p[1].get('brake_success',False) for p in group))
    result=dict(baseline=str(Path(baseline_path).resolve()),candidate=str(Path(candidate_path).resolve()),
        baseline_models=baseline['models'],candidate_models=candidate['models'],paired_episodes=len(pairs),
        regressions=regressions,eligible_for_quality_review=not regressions,
        decision='manual evidence review required; no automatic promotion',cases=comparison)
    atomic_json(Path(output),result);return result


def evaluate(incumbent,skill,model,cases,output,baseline=None,workers=4,control_config=None,base_project=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bank=output/'models';stage(incumbent,{skill:model} if model else {},bank)
    project=output/'prepared_project'
    source_project=Path(base_project) if base_project else sim2sim_root()/'godot'
    subprocess.run(['cp','--reflink=auto','-a',str(source_project),str(project)],timeout=60,check=True)
    prepare(bank,fixture_count=128,project=project,control_config=control_config)
    selected=[case for case in sorted(Path(cases).glob('*.json')) if json.loads(case.read_text())['skill']==skill]
    if not selected:raise ValueError('No skill cases selected')
    result=run(selected,output/'suite',workers=workers,project=project)
    if result['errors']:raise RuntimeError('Candidate replay errors; inspect suite evidence')
    if baseline:compare(baseline,output/'suite/summary.json',output/'comparison.json')
    return dict(summary=str(output/'suite/summary.json'),models=result['models'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    r=sub.add_parser('rescore');r.add_argument('summary');r.add_argument('--out',required=True)
    c=sub.add_parser('compare');c.add_argument('baseline');c.add_argument('candidate');c.add_argument('--out',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--incumbent',required=True);e.add_argument('--skill',choices=TASKS,required=True)
    e.add_argument('--model',required=True);e.add_argument('--cases',required=True);e.add_argument('--out',required=True)
    e.add_argument('--baseline');e.add_argument('--workers',type=int,default=4)
    c=sub.add_parser('control');c.add_argument('--incumbent',required=True);c.add_argument('--skill',choices=TASKS,required=True)
    c.add_argument('--control-config',required=True);c.add_argument('--cases',required=True);c.add_argument('--out',required=True)
    c.add_argument('--baseline');c.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.operation=='rescore':result=rescore(a.summary,a.out);print(json.dumps(dict(episodes=len(result['episodes']),errors=result['errors'])))
    elif a.operation=='compare':result=compare(a.baseline,a.candidate,a.out);print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    elif a.operation=='control':print(json.dumps(evaluate(a.incumbent,a.skill,None,a.cases,a.out,a.baseline,a.workers,a.control_config)))
    else:print(json.dumps(evaluate(a.incumbent,a.skill,a.model,a.cases,a.out,a.baseline,a.workers)))
