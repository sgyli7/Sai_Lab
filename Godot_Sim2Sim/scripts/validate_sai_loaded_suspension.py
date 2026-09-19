"""Untouched loaded MuJoCo terrain/initial-state matrix, frozen before evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from sai_loaded_mujoco import run

KINDS=['flat','rough','washboard','potholes','bumps','cross','slope','up20','down20','up40','down40','mixed20']


def job(args):
    config,parameters,out=args
    return run(parameters=parameters,out=out,**config)


def cases(final=False,round_id=1):
    cases=[]
    for i,seed in enumerate([2003,2203,2401] if final else [1201,1301,1409]):
        for kind in KINDS+(["mixed40"] if final else []):
            cases.append(dict(kind=kind,seed=seed,mass=[.05,.1,.25][i],yaw=([-.06,.03,.07] if final else [-.08,0.,.08])[i],start_x=([-.02,.01,.04] if final else [-.03,.02,.05])[i],
                              tread=([.175,.185,.195] if final else [.17,.18,.19])[i],terrain_scale=([.95,1.05,1.15] if final else [.9,1.,1.1])[i]))
    for kind in ['rough','washboard','up20','down20','up40','down40']:
        cases.append(dict(kind=kind,seed=2609 if final else 1601,mass=.1,clamped=True))
    if final and round_id == 2:
        for i, case in enumerate(cases):
            group=i//13
            case['seed']=[3001,3203,3407,3607][min(group,3)]
            if not case.get('clamped'):
                rng=np.random.default_rng(case['seed']*100+i)
                case.update(yaw=float(rng.uniform(-.08,.08)),start_x=float(rng.uniform(-.03,.05)),
                            tread=float(rng.uniform(.17,.195)),terrain_scale=float(rng.uniform(.9,1.15)))
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--final',action='store_true');p.add_argument('--round',type=int,default=1);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);profile=json.loads(a.profile.read_text());parameters=profile['parameters'];matrix=cases(a.final,a.round)
    (a.out/'frozen-matrix.json').write_text(json.dumps(dict(profile=profile,cases=matrix),indent=2)+'\n')
    import hashlib,shutil
    source_root=Path(__file__).resolve().parents[1]
    provenance={}
    files=[*source_root.glob('src/sim2sim/sai_*.py'),*source_root.glob('scripts/*sai*loaded*.py'),*source_root.glob('godot/sai/*.gd'),source_root/'scripts/sai_loaded_probe.gd']
    for source in files:
        relative=source.relative_to(source_root);target=a.out/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        provenance[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (a.out/'source-sha256.json').write_text(json.dumps(provenance,indent=2)+'\n')
    jobs=[]
    for index,case in enumerate(matrix):
        for name,params in [('baseline',None),('candidate',parameters)]:jobs.append((case,params,a.out/f'{index:02d}-{case["kind"]}-{name}'))
    reports=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for args,r in zip(jobs,pool.map(job,jobs)):
            name=args[2].name;r['case_id']=name;reports.append(r)
            (a.out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
            print(json.dumps(dict(case=name,metrics=r['metrics'])),flush=True)
    pairs=[];checks={}
    for index,case in enumerate(matrix):
        b,c=reports[2*index]['metrics'],reports[2*index+1]['metrics'];name=f'{index:02d}-{case["kind"]}';stairs=case['kind'].startswith(('up','down','mixed'))
        checks[name+'_completed']=c['completed'] and not c['cargo_lost'] and c['min_upright']>.8
        checks[name+'_speed']=c['speed']>=max(.08 if stairs else .4,min(.16 if stairs else .5,b['speed'])*.9)
        checks[name+'_height']=abs(c['height_error_mean'])<.025
        if case.get('clamped'):checks[name+'_clamp']=c['bilateral_clamp_fraction']>.9
        ratios={k:c[k]/max(1e-8,b[k]) for k in ['cargo_accel_rms','cargo_accel_p95','cargo_accel_peak','cargo_jerk_rms','deck_accel_rms']}
        pairs.append(dict(case=case,ratios=ratios,baseline=b,candidate=c))
    for group in ['continuous','stairs']:
        pp=[p for p in pairs if p['case']['kind'].startswith(('up','down','mixed'))==(group=='stairs')]
        for metric in ['cargo_accel_rms','cargo_accel_p95','cargo_jerk_rms']:
            checks[group+'_'+metric]=np.mean([p['ratios'][metric] for p in pp])<.95
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},pairs=pairs)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v])),flush=True)

if __name__=='__main__':main()
