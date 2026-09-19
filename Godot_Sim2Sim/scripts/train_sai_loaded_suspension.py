"""Bounded MuJoCo curriculum optimization; every candidate sees all terrain families."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sai_loaded_mujoco import run

CASES=[('flat',47,.1),('rough',47,.1),('washboard',71,.1),('potholes',71,.05),('bumps',109,.25),
       ('cross',109,.1),('slope',47,.1),('up20',47,.1),('down20',71,.1),('up40',109,.1),('down40',47,.1),('mixed20',109,.1)]
LOW=np.array([12.,.5,0.,0.,0.,.03]);HIGH=np.array([65.,3.5,1.15,1800.,180.,.5])
KEYS=['cargo_accel_rms','cargo_accel_p95','cargo_jerk_rms','deck_accel_rms','cargo_accel_peak']
WEIGHTS=np.array([.35,.25,.15,.15,.10])


def episode(job):
    kind,seed,mass,p,out=job
    return run(kind,seed,mass,p,out)


def assess(reports,baseline):
    violations=[];ratios=[]
    for r,b in zip(reports,baseline):
        m,bm=r['metrics'],b['metrics'];kind=r['kind'];stairs=kind.startswith(('up','down','mixed'))
        for label,bad in [('unfinished',not m['completed']),('cargo_lost',m['cargo_lost']),('fell',m['min_upright']<.8),
                          ('speed',m['speed']<max(.085 if stairs else .4,bm['speed']*.90)),
                          ('height',abs(m['height_error_mean'])>.025 or m['height_error_p95']>max(.035,bm['height_error_p95']*1.1)),
                          ('duration',m['duration']>bm['duration']*1.2),
                          ('cargo_contact',m['cargo_supported_fraction']<min(.90,bm['cargo_supported_fraction']-.02))]:
            if bad:violations.append(kind+':'+label)
        ratios.append(float(np.sum(WEIGHTS*np.array([m[k]/max(1e-8,bm[k]) for k in KEYS]))))
    # Worst-family pressure prevents low average noise from hiding a bad stair case.
    loss=float(np.mean(ratios)+.35*np.max(ratios))
    if violations:loss+=10*len(violations)
    return loss,violations,ratios


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--generations',type=int,default=6)
    p.add_argument('--population',type=int,default=10);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();rng=np.random.default_rng(15092026)
    manifest=dict(cases=CASES,low=LOW.tolist(),high=HIGH.tolist(),keys=KEYS,weights=WEIGHTS.tolist(),generations=a.generations,population=a.population,
                  final_heldout_seeds=[1201,1301,1409],method='bounded cross-entropy optimization of stance impedance in full MuJoCo with free cargo; neural actors frozen',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('sai_loaded_mujoco.py'),Path(__file__).parents[1]/'src/sim2sim/sai_compliance.py']})
    (a.out/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n');history=[];best=None
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        baseline=list(pool.map(episode,[(k,s,m,None,a.out/'baseline'/k) for k,s,m in CASES]))
        (a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');print('BASELINE_COMPLETE',flush=True)
        mean=(np.array([30.,1.5,.7,400.,60.,.2])-LOW)/(HIGH-LOW);spread=np.full(6,.22)
        for generation in range(a.generations):
            if generation==0:
                params=[[30.,1.5,0.,0.,0.,.2],[30.,1.5,1.,0.,0.,.2],[30.,1.5,1.,600.,100.,.2],[18.,1.2,.8,400.,70.,.15]]
            else:params=[best['parameters']]
            while len(params)<a.population:
                point=np.clip(rng.normal(mean,spread),0,1);params.append((LOW+point*(HIGH-LOW)).tolist())
            batch=[]
            for point in params:
                index=len(history);trial=a.out/'trials'/f'{index:03d}'
                reports=list(pool.map(episode,[(k,s,m,point,trial/k) for k,s,m in CASES]))
                loss,bad,ratios=assess(reports,baseline)
                row=dict(index=index,generation=generation,parameters=point,loss=loss,violations=bad,ratios=ratios,metrics=[r['metrics'] for r in reports])
                history.append(row);batch.append(row)
                with (a.out/'trials.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ('index','generation','parameters','loss','violations')}),flush=True)
                best=min(history,key=lambda r:r['loss']);(a.out/'best-so-far.json').write_text(json.dumps(best,indent=2)+'\n')
            elite=sorted(batch,key=lambda r:r['loss'])[:max(3,a.population//3)]
            points=(np.array([r['parameters'] for r in elite])-LOW)/(HIGH-LOW)
            mean=points.mean(axis=0);spread=np.maximum(points.std(axis=0),.06)
    result=dict(schema_version=1,id='sai-loaded-suspension-v2',parameters=best['parameters'],status='candidate_unvalidated',
                training_loss=best['loss'],training_violations=best['violations'],trials=len(history),elapsed_seconds=time.monotonic()-started,protocol=manifest)
    (a.out/'candidate.json').write_text(json.dumps(result,indent=2)+'\n');print('TRAINING_COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':main()
