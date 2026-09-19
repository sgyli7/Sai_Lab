"""Independent pass/fail gates; reward/loss does not determine deployment."""
import argparse
import json
from pathlib import Path
import numpy as np


def accept(directory):
    checks={};totals={}
    for sim in ['cpu','native']:
        rows=json.loads((directory/f'heldout-{sim}/summary.json').read_text())
        checks[sim+'_18_completed']=len(rows)==18 and all(r['metrics']['samples']==351 for r in rows)
        for seed in [211,307,401,503,607,709]:
            group={r['mode']:r['metrics'] for r in rows if r['seed']==seed}
            c,b=group['suspension'],group['gate']
            checks[f'{sim}_{seed}_rolling_speed']=c['speed']>=.4 and c['stair_fraction']==0.
            checks[f'{sim}_{seed}_upright']=c['min_upright']>.9
            if seed in [211,307,401]:
                checks[f'{sim}_{seed}_suspension_benefit']=c['angular_rms']<b['angular_rms']*.9 and c['gap_rms']<b['gap_rms']*.7 and abs(c['speed']-b['speed'])<.03
            if seed==709:
                checks[sim+'_flat_unchanged']=abs(c['speed']-b['speed'])<.001 and abs(c['angular_rms']-b['angular_rms'])<.001
        totals[sim]={mode:{k:float(np.mean([r['metrics'][k] for r in rows if r['mode']==mode and r['kind']=='rough']))
            for k in ['speed','angular_rms','vertical_accel_rms','gap_rms','four_contact_fraction','mean_contacts']}
            for mode in ['original','gate','suspension']}
    stair=json.loads((directory/'stairs/release-validation/acceptance.json').read_text())
    checks['original_stairs_acceptance']=stair['passed'] and len(stair['cases'])==4
    maneuvers=json.loads((directory/'maneuvers/summary.json').read_text())
    checks['six_maneuvers_completed']=len(maneuvers)==6
    for report in maneuvers:
        name=report['maneuver'];m=report['metrics']
        checks[name+'_continuous_and_upright']=m['stair_fraction']==0. and m['min_upright']>.9 and abs(m['speed'])>=.4
        raw=json.loads((directory/f'maneuvers/{name}/raw.json').read_text())['samples']
        stopped=[r for r in raw if r['time']>=6.6]
        checks[name+'_stopped']=bool(stopped) and max(abs(r['policy_observation'][3]) for r in stopped)<.04
        if name=='turn':
            moving=[r for r in raw if 2<=r['time']<6]
            checks['requested_turn_delivered']=np.mean([r['policy_observation'][8] for r in moving])>.2
        if name=='crouch_cycle':
            checks['crouch_then_recovered']=max(r['effective_crouch'] for r in raw)>.99 and raw[-1]['effective_crouch']<.01
    station=json.loads((directory/'station/comparison.json').read_text())
    c=station['candidate']
    checks['science_station_original_scene']=c['speed']>=.4 and c['stair_fraction']==0. and c['min_upright']>.9 and c['mean_contacts']>=3.5
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},rough_averages=totals,science_station=station)
    (directory/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=accept(a.directory)
    print(json.dumps(dict(passed=r['passed'],checks=len(r['checks']),failed=[k for k,v in r['checks'].items() if not v]),indent=2))
    raise SystemExit(not r['passed'])

if __name__=='__main__':main()
