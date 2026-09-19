"""Evaluate velocity-dependent brake candidates through the complete native player.

This search includes the real crouch policy and all three repeated brake events,
which cannot be represented by a single-actor command tape.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from sim2sim.standalone.candidate import evaluate
from sim2sim.standalone.suite import runtime_inputs
from sim2sim.paths import sim2sim_root
from .brake_velocity_pose import export
from .budget import require_supervision
from .queue import atomic_json


def episode_cost(row):
    task=row['task_metrics']
    costs=[]
    for brake in row['brakes']:
        confirmation=brake['stop_confirmed_s']
        lateness=max(0., (6. if confirmation is None else confirmation)-2.)
        reverse=max(0.,-brake['minimum_forward_02s']-.03)
        distance=max(0.,brake['braking_distance']-brake['distance_limit'])
        costs.append(100.*brake['fell']+30.*(not brake['success'])+4.*lateness
            +10.*reverse+10.*distance+10.*brake['final_speed']**2)
    # All repeated stops must work. The mean also guides secondary failures.
    return max(costs)+.25*float(np.mean(costs))+10.*max(0.,.085-task['final_z'])/.03


def run(experiment, output, minutes=25., generations=12, population=16, workers=4):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    spec=json.loads(Path(experiment).read_text())
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=Path(spec['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_sha256']:
        raise ValueError('Source checksum mismatch')
    project=output/'source_project'
    subprocess.run(['cp','--reflink=auto','-a',str(sim2sim_root()/'godot'),str(project)],timeout=60,check=True)
    atomic_json(output/'source_project.json',runtime_inputs(project))
    bounds=np.asarray(spec['bounds'],float);mean=np.asarray(spec['initial'],float)
    adapter=spec.get('adapter','velocity_pose')
    dimensions=(5,) if adapter in ('onset_pose','pitch_rate','height_feedback','near_stop') else (10,15)
    if adapter not in ('velocity_pose','onset_pose','pitch_rate','height_feedback','near_stop'):raise ValueError('Unknown brake adapter')
    if (bounds.shape not in tuple((n,) for n in dimensions) or mean.shape!=bounds.shape or not np.isfinite(bounds).all()
            or not np.isfinite(mean).all() or np.any(bounds<=0) or np.any(abs(mean)>bounds)):
        raise ValueError('Invalid velocity residual search range')
    cases=output/'cases';cases.mkdir()
    for path in sorted(Path(spec['cases']).glob('*.json')):
        case=json.loads(path.read_text())
        if case['case'] in spec['selected_cases'] and case['seed'] in spec['seeds']:
            shutil.copyfile(path,cases/path.name)
    count=len(list(cases.glob('*.json')))
    if count!=len(spec['selected_cases'])*len(spec['seeds']):raise ValueError('Incomplete native search cases')
    adapter_source=Path(__file__).with_name({'velocity_pose':'brake_velocity_pose.py',
        'onset_pose':'brake_onset_pose.py','pitch_rate':'brake_pitch_rate.py',
        'height_feedback':'brake_height_feedback.py','near_stop':'brake_near_stop.py'}[adapter])
    spec.update(minutes=minutes,generations=generations,population=population,workers=workers,
        objective_version='native_brake_timing_v1',case_count=count,
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),adapter_source,Path(evaluate.__code__.co_filename)]})
    atomic_json(output/'experiment.json',spec)
    started=time.monotonic();rng=np.random.default_rng(915005);sigma=bounds*.16
    frozen_prefix=int(spec.get('frozen_prefix',0))
    if not 0<=frozen_prefix<len(mean):raise ValueError('Invalid frozen parameter prefix')
    fixed=mean[:frozen_prefix].copy()
    def trial(parameters,label):
        model=output/(label+'.onnx')
        if adapter=='near_stop':
            from .brake_near_stop import export as export_near
            parity=export_near(source,parameters,model)
        elif adapter=='height_feedback':
            from .brake_height_feedback import export as export_height
            parity=export_height(source,parameters,model)
        elif adapter=='pitch_rate':
            from .brake_pitch_rate import export as export_rate
            parity=export_rate(source,parameters,model)
        elif adapter=='onset_pose':
            from .brake_onset_pose import export as export_onset
            parity=export_onset(source,parameters,model)
        else:
            parity=export(source,parameters,model,spec.get('pitch_mode','centered'))
        directory=output/label
        evaluate(spec['incumbent'],'roller',model,cases,directory,workers=workers,base_project=project)
        rows=json.loads((directory/'suite/summary.json').read_text())['episodes']
        result=dict(parameters=np.asarray(parameters).tolist(),model=str(model),parity=parity,
            objective=float(np.mean([episode_cost(row) for row in rows])),
            passes=sum(row['brake_success'] for row in rows),
            falls=sum(any(brake['fell'] for brake in row['brakes']) for row in rows),
            episodes=len(rows),evidence=str(directory/'suite/summary.json'))
        atomic_json(directory/'search_result.json',result)
        print(json.dumps(dict(trial=label,**{k:result[k] for k in ['objective','passes','falls']})),flush=True)
        return result
    best=trial(mean,'initial');atomic_json(output/'initial.json',best)
    cost=(time.monotonic()-started)*population*1.2
    history=[]
    for generation in range(generations):
        if time.monotonic()-started+cost+20>minutes*60:break
        begin=time.monotonic()
        values=np.clip(rng.normal(mean,sigma,(population,len(bounds))),-bounds,bounds)
        values[:,:frozen_prefix]=fixed
        # Reuse the saved, immutable result of the unchanged elite.
        candidates=[best]
        for i,parameters in enumerate(values[1:],1):
            candidates.append(trial(parameters,f'g{generation:02d}_{i:03d}'))
        cost=(time.monotonic()-begin)*1.2
        if spec.get('rank_zero_falls_first',False):
            candidates.sort(key=lambda row:(row['falls']>0,row['falls'],row['objective']))
        else:
            candidates.sort(key=lambda row:row['objective'])
        best=candidates[0]
        elite=np.array([row['parameters'] for row in candidates[:max(3,population//5)]])
        mean=.3*mean+.7*elite.mean(0)
        mean[:frozen_prefix]=fixed
        sigma=np.maximum(bounds*.015,.3*sigma+.7*elite.std(0))
        record=dict(generation=generation,best=best,candidates=candidates,mean=mean.tolist(),sigma=sigma.tolist(),elapsed=time.monotonic()-started)
        atomic_json(output/f'generation_{generation:02d}.json',record);history.append(record)
    shutil.copyfile(best['model'],output/'candidate.onnx')
    result=dict(completed=True,best=best,generations=len(history),elapsed=time.monotonic()-started,
        candidate=str(output/'candidate.onnx'),decision='unpromoted; requires all development seeds and frozen holdout')
    atomic_json(output/'completed.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment');parser.add_argument('--out',required=True)
    parser.add_argument('--minutes',type=float,default=25.);parser.add_argument('--generations',type=int,default=12)
    parser.add_argument('--population',type=int,default=16);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();run(args.experiment,args.out,args.minutes,args.generations,args.population,args.workers)
