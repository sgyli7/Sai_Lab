"""Fail-closed paired sprint acceptance; never chooses training parameters."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def paired_acceptance(summary, case_paths, minimum_gain=1.15):
    expected={str(Path(p).resolve()):json.loads(Path(p).read_text()) for p in case_paths}
    errors=[];pairs=defaultdict(dict);seen=set()
    for episode in summary['episodes']:
        path=str(Path(episode['case_path']).resolve())
        if path not in expected or path in seen:
            errors.append('Unexpected or duplicate episode: '+path);continue
        seen.add(path);case=expected[path]
        if not episode.get('completed') or episode.get('error'):
            errors.append('Incomplete episode: '+path);continue
        key=(case['pair_id'],case['seed']);ordinary=bool(case['ordinary_control'])
        if ordinary in pairs[key]:errors.append('Duplicate pair member: '+str(key))
        pairs[key][ordinary]=(case,episode)
    if seen!=set(expected):errors.append('Missing expected episodes')
    groups=defaultdict(list);hard_pass=True
    for key,members in pairs.items():
        if set(members)!={False,True}:
            errors.append('Incomplete pair: '+str(key));continue
        candidate,c=members[False];ordinary,o=members[True]
        # The only keyboard intervention is removal of the sprint modifier.
        stripped=[dict(s,held=[k for k in s['held'] if k!='sprint']) for s in candidate['segments']]
        comparable=(stripped==ordinary['segments'] and
                    all(candidate[k]==ordinary[k] for k in
                        ['initial_poses','control_config','seconds','sprint_intervals','protocol']) and
                    c['models']==o['models'] and c['runtime_id']==o['runtime_id'])
        if not comparable:errors.append('Mismatched paired contract: '+str(key))
        cm=c['task_metrics'];om=o['task_metrics']
        cv=float(cm['sustained_mean_vx']);ov=float(om['sustained_mean_vx'])
        if not all(math.isfinite(x) for x in [cv,ov]) or ov<=0:
            errors.append('Invalid paired speed: '+str(key));continue
        hard_pass &= bool(cm['success'] and not om['fell'] and om['actor_selection_passed'])
        groups[key[0]].append(dict(seed=key[1],candidate_vx=cv,ordinary_vx=ov,
                                   candidate_pass=cm['success'],ordinary_fell=om['fell']))
    comparisons=[]
    for name,rows in sorted(groups.items()):
        cv=sum(r['candidate_vx'] for r in rows)/len(rows)
        ov=sum(r['ordinary_vx'] for r in rows)/len(rows)
        comparisons.append(dict(case=name,count=len(rows),candidate_vx=cv,ordinary_vx=ov,
                                ratio=cv/ov,speed_pass=cv>=minimum_gain*ov,pairs=rows))
    accepted=bool(expected and comparisons and not errors and hard_pass and
                  all(r['speed_pass'] for r in comparisons))
    return dict(protocol='walking_sprint_paired_v1',minimum_speed_ratio=minimum_gain,
                expected_episodes=len(expected),errors=errors,physical_pass=hard_pass,
                comparisons=comparisons,accepted=accepted)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path)
    p.add_argument('cases',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=paired_acceptance(json.loads(a.summary.read_text()),sorted(a.cases.rglob('*.json')))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
