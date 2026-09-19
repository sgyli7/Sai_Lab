"""Bounded CEM of raw PD position targets; evaluates unchanged physical gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .brake_pose_controller import prediction, export
from .brake_pose_search import rollout, action_offset
from .budget import require_supervision
from .models import NativeAnchor
from .queue import atomic_json


class TargetActor:
    def __init__(self,anchor,parameters,alpha,feedback,joint):
        self.anchor=anchor;self.target=action_offset(parameters)
        self.alpha,self.feedback,self.joint=alpha,feedback,joint

    def __call__(self,observations):
        return prediction(self.anchor,observations,self.target,self.alpha,self.feedback,False,self.joint)


def run(experiment,output,minutes=12.,generations=12,population=20,workers=4):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    spec=json.loads(Path(experiment).read_text());output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    rng=np.random.default_rng(915003)
    mean=np.array(spec['initial'],float);radius=np.array(spec['radius'],float)
    low,high=mean-radius,mean+radius;sigma=radius*.35
    anchors=[NativeAnchor(spec['source']) for _ in range(workers)]
    if any(a.sha256!=spec['source_sha256'] for a in anchors):raise ValueError('Source checksum mismatch')
    spec.update(minutes=minutes,generations=generations,population=population,workers=workers,
        source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('brake_pose_controller.py'),Path(__file__).with_name('brake_pose_search.py')]})
    atomic_json(output/'experiment.json',spec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def evaluate(parameters,label):
            jobs=[(i,p,case,seed) for i,p in enumerate(parameters) for case in spec['cases'] for seed in spec['seeds']]
            rows=[]
            for start in range(0,len(jobs),workers):
                futures=[pool.submit(rollout,None,TargetActor(anchors[lane],p,spec['alpha'],spec['feedback'],spec['feedback_joint']),
                    np.zeros(5),seed,output,f'{label}_{i:03d}_{case}',case)
                    for lane,(i,p,case,seed) in enumerate(jobs[start:start+workers])]
                rows.extend(f.result() for f in futures)
            candidates=[]
            for i,p in enumerate(parameters):
                trials=[r for r in rows if r['trial'].startswith(f'{label}_{i:03d}_')]
                candidates.append(dict(parameters=np.asarray(p).tolist(),trials=trials,
                    objective=float(np.mean([r['objective'] for r in trials])),
                    passes=sum(r['metrics']['success'] for r in trials),falls=sum(r['metrics']['fell'] for r in trials)))
            return sorted(candidates,key=lambda r:r['objective'])
        best=evaluate([mean],'initial')[0];atomic_json(output/'initial.json',best)
        cost=3*population*len(spec['cases'])*len(spec['seeds'])/workers
        history=[]
        for generation in range(generations):
            if time.monotonic()-started+cost+20>minutes*60:break
            begin=time.monotonic();parameters=np.clip(rng.normal(mean,sigma,(population,5)),low,high)
            parameters[0]=best['parameters'];candidates=evaluate(parameters,f'g{generation:02d}')
            cost=(time.monotonic()-begin)*1.2
            if candidates[0]['objective']<best['objective']:best=candidates[0]
            elite=np.array([r['parameters'] for r in candidates[:max(3,population//5)]])
            mean=.3*mean+.7*elite.mean(0);sigma=np.maximum(radius*.025,.3*sigma+.7*elite.std(0))
            row=dict(generation=generation,best=best,candidates=candidates,mean=mean.tolist(),sigma=sigma.tolist(),elapsed=time.monotonic()-started)
            atomic_json(output/f'generation_{generation:02d}.json',row);history.append(row)
            print(json.dumps(dict(generation=generation,passes=best['passes'],falls=best['falls'],objective=best['objective'],elapsed=row['elapsed'])),flush=True)
        parity=export(spec['source'],output/'candidate.onnx',best['parameters'],spec['alpha'],spec['feedback'],False,spec['feedback_joint'])
        result=dict(completed=True,best=best,generations=len(history),parity=parity,elapsed=time.monotonic()-started,decision='unpromoted; full native development suite required')
        atomic_json(output/'completed.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');p.add_argument('--out',required=True)
    p.add_argument('--minutes',type=float,default=12.);p.add_argument('--generations',type=int,default=12)
    p.add_argument('--population',type=int,default=20);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.experiment,a.out,a.minutes,a.generations,a.population,a.workers)
