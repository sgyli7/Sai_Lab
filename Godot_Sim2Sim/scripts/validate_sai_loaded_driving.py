"""Loaded driver regression checks outside the forward terrain curriculum."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from sai_loaded_mujoco import run

def job(a):
    c,p,out=a
    return run(parameters=p,out=out,**c)

def main():
    root=Path('results/sai-cargo-suspension-20260915');out=root/'accept-driving-v2';out.mkdir(exist_ok=False)
    p=json.loads((root/'descent-selection/candidate.json').read_text())['parameters']
    cases=[dict(kind='rough',drive_speed=.25),dict(kind='rough',drive_speed=-.25,start_x=2.),dict(kind='rough',drive_speed=.3,turn_rate=.35),dict(kind='rough',drive_speed=.3,turn_rate=-.35),dict(kind='flat',drive_speed=.25,crouch=1.),dict(kind='rough',drive_speed=.25,crouch=1.)]
    for c in cases:c.update(seed=3907,mass=.1,clamped=True)
    cases += [dict(kind='rough',seed=seed,mass=.1,clamped=True,drive_speed=speed,turn_rate=yaw) for seed,speed,yaw in [(4001,.35,.25),(4001,.35,-.25),(4201,.3,.45),(4201,.3,-.45)]]
    (out/'protocol.json').write_text(json.dumps(dict(cases=cases,parameters=p,checks='No fall or cargo loss, completion; speed retains >=90% successful baseline and >=70% requested; crouch height compared to crouch baseline, not standing reference.'),indent=2)+'\n')
    jobs=[(c,par,out/f'{i}-{label}') for i,c in enumerate(cases) for label,par in [('baseline',None),('candidate',p)]]
    with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(job,jobs))
    checks={}
    for i,c in enumerate(cases):
        b,r=reports[2*i]['metrics'],reports[2*i+1]['metrics']
        checks[f'{i}_complete']=r['completed'] and not r['cargo_lost'] and r['min_upright']>.8
        checks[f'{i}_speed']=r['speed']>=max(abs(c['drive_speed'])*.7,b['speed']*.9)
        checks[f'{i}_height']=abs(r['height_error_mean']-b['height_error_mean'])<.015
        checks[f'{i}_clamp']=r['bilateral_clamp_fraction']>.9
    result=dict(passed=all(checks.values()),checks=checks,reports=reports);(out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v])),flush=True)
if __name__=='__main__':main()
