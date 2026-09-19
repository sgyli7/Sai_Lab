"""Turning support transfer check after the MuJoCo single-parameter probe."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sai_loaded_native import run

def job(a):return run(**a)

def main():
    root=Path('results/sai-cargo-suspension-20260915');out=root/'accept-native-turns';out.mkdir(exist_ok=False)
    p=json.loads((root/'descent-selection/candidate.json').read_text())['parameters'];cases=[(.3,.35),(.3,-.35),(.35,.25),(.3,-.45)]
    jobs=[dict(out=out/f'{i}-{name}',kind='rough',seed=4409,mass=.1,clamped=True,parameters=parameters,drive_speed=speed,turn_rate=yaw) for i,(speed,yaw) in enumerate(cases) for name,parameters in [('baseline',None),('candidate',p)]]
    with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(job,jobs))
    checks={}
    for i,(speed,yaw) in enumerate(cases):
        b,c=reports[2*i]['metrics'],reports[2*i+1]['metrics']
        checks[f'{i}_complete']=c['completed'] and not c['cargo_lost'] and c['min_upright']>.8
        checks[f'{i}_speed']=c['speed']>=max(.7*speed,.9*b['speed'])
        checks[f'{i}_clamp']=c['bilateral_clamp_fraction']>.9
    result=dict(passed=all(checks.values()),checks=checks,reports=reports);(out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v])),flush=True)
if __name__=='__main__':main()
