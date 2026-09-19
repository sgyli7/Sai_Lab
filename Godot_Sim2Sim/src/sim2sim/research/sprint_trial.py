"""Freeze each sprint actor and compare explicit controls in native Jolt replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from sim2sim.paths import sim2sim_root
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.standalone.suite import run
from .budget import require_supervision
from .tasks import SESSION
from .queue import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--actors',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--speed',type=float,default=.35);p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--seed-start',type=int,default=919000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--paired',action='store_true')
    p.add_argument('--cases',nargs='*',default=['shift_first','left','right','w_first'])
    a=p.parse_args();require_supervision(SESSION)
    a.out.mkdir(parents=True,exist_ok=False)
    actors=json.loads(a.actors.read_text());controls=json.loads(a.controls.read_text());results=[]
    for name,source in actors.items():
        directory=a.out/name;directory.mkdir();project=directory/'prepared'
        subprocess.run(['cp','--reflink=auto','-a',str(sim2sim_root()/'godot'),str(project)],check=True,timeout=60)
        actor=directory/'Sprint_Godot.onnx';shutil.copy2(source,actor)
        manifest=Path(source).with_suffix('.manifest.json')
        data=json.loads(manifest.read_text()) if manifest.exists() else {}
        data.update(sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
                    status='experimental_sprint_requires_acceptance')
        data.setdefault('sim2sim',{}).update(skill='sprint',use_stand_policy=False)
        actor.with_suffix('.manifest.json').write_text(json.dumps(data,indent=2)+'\n')
        prepare(a.models,project=project,sprint=actor)
        cases=[]
        for label,control in controls.items():
            paths=write_cases(directory/'cases'/label,range(a.seed_start,a.seed_start+a.seeds),
                              a.speed,paired=a.paired,selected=a.cases,control=control)
            for path in paths:
                case=json.loads(path.read_text());case['case']+='_'+label;case['pair_id']+='_'+label
                renamed=path.with_name(path.stem+'_'+label+'.json')
                atomic_json(renamed,case);path.unlink();cases.append(renamed)
        result=run(cases,directory/'suite',workers=a.workers,project=project)
        if a.paired:
            atomic_json(directory/'paired_acceptance.json',paired_acceptance(result,cases))
        rows=result['episodes']
        results.append(dict(actor=name,sha256=data['sha256'],errors=result['errors'],
            episodes=[dict(case=r['case'],seed=r.get('seed'),error=r.get('error'),
                metrics=r.get('task_metrics'),trace=r.get('trace')) for r in rows]))
        atomic_json(a.out/'progress.json',results)
    atomic_json(a.out/'completed.json',dict(experiments=results,promoted=False))


if __name__=='__main__':main()
