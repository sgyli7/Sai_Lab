"""Evaluate a bounded IMU pitch-feedback residual on a frozen braking pose."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from .brake_pose_search import action_offset,export,rollout
from .budget import require_supervision
from .models import NativeAnchor
from .queue import atomic_json


def run(source,initial,output,workers=4):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();previous=json.loads(Path(initial).read_text())
    parameters=np.array(previous['best']['parameters'])
    grid=list(itertools.product([0.,.15,.3,.6,-.15,-.3],[0.,.02,.05]))
    cases=['roller_brake_1s','roller_brake_3s','roller_turn_brake'];seeds=[915000,915001]
    definition=dict(hypothesis='A bounded ankle correction from existing pitch gravity and gyro observations can bring the stopped torso upright without weakening normal cruise.',
        source=str(Path(source).resolve()),source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        initial=str(Path(initial).resolve()),initial_sha256=hashlib.sha256(Path(initial).read_bytes()).hexdigest(),
        parameters=parameters.tolist(),grid=grid,cases=cases,seeds=seeds,bound_rad=.12,
        implementation='analytic brake-only policy residual; not a learned network improvement',
        physics='unchanged; fresh Jolt process for every trial',holdout_used=False)
    atomic_json(output/'experiment.json',definition)
    anchors=[NativeAnchor(source) for _ in range(workers)]
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,feedback in enumerate(grid):
            jobs=list(itertools.product(cases,seeds));rows=[]
            for begin in range(0,len(jobs),workers):
                futures=[pool.submit(rollout,None,anchors[lane],parameters,seed,output,
                                     f'grid_{index:02d}_{case}',case,feedback)
                         for lane,(case,seed) in enumerate(jobs[begin:begin+workers])]
                rows.extend(future.result() for future in futures)
            item=dict(feedback=feedback,trials=rows,passes=sum(r['metrics']['success'] for r in rows),
                      falls=sum(r['metrics']['fell'] for r in rows),objective=float(np.mean([r['objective'] for r in rows])))
            atomic_json(output/f'grid_{index:02d}.json',item);records.append(item)
            print(json.dumps({key:value for key,value in item.items() if key!='trials'}),flush=True)
    best=min(records,key=lambda item:(item['falls']>0,-item['passes'],item['objective']))
    parity=export(source,action_offset(parameters),output/'candidate.onnx',best['feedback'])
    result=dict(completed=True,best=best,baseline=records[0],candidates=records,parity=parity,
                elapsed=time.monotonic()-started,decision='unpromoted; full native paired development suite required')
    atomic_json(output/'completed.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--initial',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();result=run(args.source,args.initial,args.out,args.workers)
    print(json.dumps({key:value for key,value in result.items() if key not in ['candidates','best','baseline']}))
