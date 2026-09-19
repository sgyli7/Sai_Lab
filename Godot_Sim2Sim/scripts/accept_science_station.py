"""Exercise actual keys and Jolt bodies across both scenes; retain compact evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/science-station/acceptance')
    parser.add_argument('--share-resources',action='store_true')
    parser.add_argument('--runtime-dir',type=Path,help='Isolated runtime; leave a live desktop session untouched')
    parser.add_argument('--cases',nargs='*',help='Optional subset, for retrying a corrected case')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[(f'{route}-{robot}',robot,route,'science_station')
           for route in ['loop','slope','tower','lab','north'] for robot in ['microduck','roller','sai']]
    cases += [('switches','microduck','switches','science_station'),
              ('grab','sai','grab','science_station'),('cancel','sai','cancel','science_station'),
              ('play','microduck','play','science_station'),('workshop','microduck','controls','workshop')]
    cases += [(name,'sai',name,'science_station') for name in
              ['grab-service-bottle','grab-samples','grab-samples-bottle','grab-berth','grab-berth-bin']]
    records=[]
    for name,robot,plan,scene in cases:
        if args.cases and name not in args.cases:continue
        destination=args.output/name
        plan_path=ROOT/('docs/workshop-hub-20260912/plans/controls.json' if scene=='workshop'
                       else f'docs/science-station/plans/{plan}.json')
        command=[sys.executable,str(ROOT/'scripts/run_science_check.py'),'--scene',scene,'--robot',robot,
                 '--headless','--fast-check','--plan',str(plan_path),'--output',str(destination)]
        if args.share_resources:command+=['--share-resources']
        if args.runtime_dir:command+=['--runtime-dir',str(args.runtime_dir.resolve())]
        print(f'CHECK {name}',flush=True)
        process=subprocess.run(command,cwd=ROOT)
        record={'case':name,'exit':process.returncode,'passed':False,'source':str(destination)}
        path=destination/'hub.json'
        if path.exists() and process.returncode==0:
            h=json.loads(path.read_text())
            checks={'scene':h['scene']==scene}
            if plan in ['loop','slope','tower','lab','north']:
                checks['route']=h['route'].get('passed',False)
                checks['no_native_fall']=all(json.loads(p.read_text())['first_fall'] is None for p in destination.glob('native-*.json'))
                record['route']=h['route']
            elif plan=='switches':
                checks['switch_order']=[e['to'] for e in h['events']]==['microduck','sai','roller','microduck','microduck']
                checks['same_scenery']=len({e['atelier'] for e in h['events']})==1
                checks['props_preserved']=all(e['props_before']==e['props_after'] for e in h['events'])
                checks['safe_spawn']=all(math.hypot(e['spawn'][0]+5,e['spawn'][2]-1)<.08 for e in h['events'])
                anchors=[(.3,7),(.66,7.08),(-4.7,1),(-4.34,1.08),(5.3,5),(5.66,5.08)]
                checks['props_reset']=all(math.hypot(p['position'][0]-a[0],p['position'][2]-a[1])<.02 for p,a in zip(h['props'],anchors))
            elif plan.startswith('grab'):
                sessions=h['grab_sessions']
                checks['delivered']=any(d.get('success',False) and d.get('captured',False) and d.get('released',False)
                                        for s in sessions for d in s['deliveries'])
                record['grab']=sessions
            elif plan=='cancel':
                checks['cancelled']=any(e['event']=='cancel' for s in h['grab_sessions'] for e in s['history'])
                checks['not_delivered']=all(not s['deliveries'] for s in h['grab_sessions'])
            elif plan=='play':
                native=json.loads((destination/'native-0.json').read_text())
                checks['kick_command']=any(r['skill']=='kick_right' for r in native['rows'])
                peak=max(math.sqrt(sum(v*v for v in s['prop_motion'][0]['velocity'])) for s in h['samples'] if 3<s['time']<7)
                checks['physical_prop_motion']=peak>.15
                checks['walked']=math.hypot(h['position'][0],h['position'][2]-7)>.5
                record['kicked_bin_peak_speed_m_s']=peak
            elif scene=='workshop':
                result=destination/'accepted.json'
                check=subprocess.run([sys.executable,str(ROOT/'scripts/accept_workshop.py'),str(destination),'--out',str(result)],cwd=ROOT)
                checks['existing_regression']=check.returncode==0
            record['checks']=checks;record['passed']=all(checks.values())
            record['simulation_seconds']=h['seconds']
        records.append(record)
        (args.output/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(f"RESULT {name}: {'PASS' if record['passed'] else 'FAIL'}",flush=True)
    all_records=[json.loads((args.output/(name+'.json')).read_text()) for name,_,_,_ in cases if (args.output/(name+'.json')).exists()]
    summary={'passed':len(all_records)==len(cases) and all(r['passed'] for r in all_records),'cases':all_records,
             'method':'Native Godot/Jolt, normal input events, unpaced headless rendering; original physics timestep and policies'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if all(r['passed'] for r in records) else 1

if __name__=='__main__':raise SystemExit(main())
