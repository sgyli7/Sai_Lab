"""One-factor controller experiments with immutable cases and explicit comparisons."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from sim2sim.paths import sim2sim_root
from sim2sim.research.budget import require_supervision
from sim2sim.research.queue import atomic_json
from sim2sim.research.tasks import TASKS
from .cases import standard_cases
from .suite import run


HYPOTHESES={
    'roller_decel':dict(skill='roller',mode='roller',key='decel',values=[20.,10.,5.,2.,1.],
        hypothesis='S command reversal is too abrupt; a slower command deceleration avoids tipping while retaining cruise.'),
    'roller_brake_strength':dict(skill='roller',mode='roller',key='vmin_x',values=[-.5,-.4,-.3,-.2,-.1,-.05],
        hypothesis='S requests excessive continuous braking; reducing only negative throttle prevents loss of balance without reducing W cruise.'),
    'roller_brake_pulse':dict(skill='roller',mode='roller',key='brake_pulse_s',scope='motion',values=[None,.15,.25,.35,.45,.55,.65],
        hypothesis='S continues a braking posture after useful deceleration; a bounded brake pulse followed by neutral avoids tipping and stops without reversing.'),
    'roller_heading_hold':dict(skill='roller',mode='roller',key='heading_hold',scope='motion',values=[False,True],
        hypothesis='A fixed heading reference prevents drift during push/coast/brake and improves stable braking using the original actor heading-error input.'),
    'walk_heading_hold':dict(skill='walking',mode='walk',key='walk_heading_gain',scope='motion',values=[0.,.5,1.,2.,4.],
        hypothesis='A bounded yaw-rate correction while walking straight or idle reduces heading drift; deliberate turning commands and requested forward speed remain unchanged.'),
}


def experiment(name,output,workers=4):
    root=sim2sim_root();output=Path(output).resolve()
    require_supervision(Path(__import__('os').environ['SIM2SIM_RESEARCH_DIR']))
    output.mkdir(parents=True,exist_ok=False);cases=output/'cases';cases.mkdir()
    definition=HYPOTHESES[name]
    templates={k:v for k,v in standard_cases().items() if v['skill']==definition['skill']}
    started=time.time()
    record=dict(name=name,definition=definition,started_unix=started,
        model_source_sha256=hashlib.sha256((root/'godot/runtime_assets/policies'/TASKS[definition['skill']].previous).read_bytes()).hexdigest(),
        control_sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [root/'godot/standalone/play_brain.gd',root/'godot/standalone/driver.gd',root/'src/sim2sim/play_input.py',
             root/'godot/standalone/motion_control.gd',root/'src/sim2sim/motion_control.py']},
        selection='development only; no automatic promotion')
    atomic_json(output/'experiment.json',record)
    for value in definition['values']:
        for key,template in templates.items():
            case=copy.deepcopy(template);case.update(case=key,seed=None,randomized_start=False)
            settings={definition['key']:value} if value is not None else {}
            if definition.get('scope')!='motion':settings={'twist_limits':settings}
            case['control_config']={'version':name+'_v1',definition['mode']:settings}
            label='baseline' if value is None else f'{value:g}'
            atomic_json(cases/f'{key}_{label}.json',case)
    results=run(sorted(cases.glob('*.json')),output/'suite',workers=workers)
    comparisons=[]
    for value in definition['values']:
        def parameter(row):
            settings=row.get('control_config',{}).get(definition['mode'],{})
            if definition.get('scope')!='motion':settings=settings.get('twist_limits',{})
            return settings.get(definition['key'])
        rows=[row for row in results['episodes'] if parameter(row)==value]
        active=[row for row in rows if '_release_' not in row['case'] and '_space_' not in row['case']]
        baseline=next((row for row in rows if row['case']==('roller_brake_3s' if definition['skill']=='roller' else 'walking_straight')), {})
        comparisons.append(dict(value=value,completed=len(rows),active_brake_passes=sum(row.get('brake_success',False) for row in active),
            active_brake_cases=sum('brakes' in row for row in active),falls=sum(row['task_metrics']['fell'] for row in active),
            task_successes=sum(row['task_metrics']['success'] for row in rows),
            original_case=baseline.get('brakes',baseline.get('task_metrics')),cruise_mean_vx=baseline.get('moving_mean_vx')))
    record.update(completed=results['errors']==0,errors=results['errors'],elapsed_s=time.time()-started,comparisons=comparisons)
    atomic_json(output/'completed.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name',choices=HYPOTHESES)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();result=experiment(a.name,a.out,a.workers)
    print(json.dumps(result))
    if not result['completed']:raise SystemExit(1)
