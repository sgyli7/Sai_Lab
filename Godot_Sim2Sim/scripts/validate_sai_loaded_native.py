"""Independent-engine loaded checks. Never used for fitting MuJoCo parameters."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sai_loaded_native import run

def job(args):return run(**args)

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    profile=json.loads(a.profile.read_text());matrix=[]
    for kind in ['rough','washboard','potholes','cross','up20','down20','up40','down40','mixed20','mixed40']:
        matrix.append(dict(kind=kind,seed=2801,mass=.1,clamped=False))
    for kind in ['rough','washboard','up20','down20','up40','down40']:
        matrix.append(dict(kind=kind,seed=2801,mass=.1,clamped=True))
    (a.out/'protocol.json').write_text(json.dumps(dict(cases=matrix,profile=profile,requirements='Every candidate completes without falling/cargo loss, rough speed >= .4 m/s, stairs >= .08 m/s; mean paired RMS and P95 improve in continuous and stairs groups. Raw peaks separately disclosed, no claim that each instantaneous peak decreases.'),indent=2)+'\n')
    import hashlib,shutil
    source_root=Path(__file__).resolve().parents[1]
    provenance={}
    files=[*source_root.glob('src/sim2sim/sai_*.py'),*source_root.glob('scripts/*sai*loaded*.py'),*source_root.glob('godot/sai/*.gd'),source_root/'scripts/sai_loaded_probe.gd']
    for source in files:
        relative=source.relative_to(source_root);target=a.out/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        provenance[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (a.out/'source-sha256.json').write_text(json.dumps(provenance,indent=2)+'\n')
    jobs=[]
    for i,case in enumerate(matrix):
        for name,params in [('baseline',None),('candidate',profile['parameters'])]:jobs.append(dict(case,out=a.out/f'{i:02d}-{case["kind"]}-{name}',parameters=params))
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for args,r in zip(jobs,pool.map(job,jobs)):
            r['case_id']=args['out'].name;reports.append(r);(a.out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    checks={};pairs=[]
    for i,case in enumerate(matrix):
        b,c=reports[i*2]['metrics'],reports[i*2+1]['metrics'];stairs=case['kind'].startswith(('up','down','mixed'));name=f'{i:02d}-{case["kind"]}'
        checks[name+'_completed']=c['completed'] and not c['cargo_lost'] and c['min_upright']>.8
        checks[name+'_speed']=c['speed']>=max(.08 if stairs else .4,min(.16 if stairs else .5,b['speed'])*.9)
        checks[name+'_height']=abs(c['height_error_mean'])<.025
        if case.get('clamped'):checks[name+'_clamp']=c['bilateral_clamp_fraction']>.9
        pairs.append(dict(case=case,baseline=b,candidate=c,ratios={k:c[k]/max(1e-8,b[k]) for k in ['cargo_accel_rms','cargo_accel_p95','cargo_accel_peak','cargo_jerk_rms','deck_accel_rms']}))
    for group in ['continuous','stairs']:
        pp=[x for x in pairs if x['case']['kind'].startswith(('up','down','mixed'))==(group=='stairs')]
        for metric in ['cargo_accel_rms','cargo_accel_p95']:checks[group+'_'+metric]=float(np.mean([x['ratios'][metric] for x in pp]))<1.
    result=dict(passed=all(checks.values()),checks=checks,pairs=pairs);(a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v])),flush=True)
if __name__=='__main__':main()
