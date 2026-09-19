"""Measure real neck response and whole-robot COM in native intervention traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def measure(path, mass):
    trace = json.loads(Path(path).read_text())
    rows = [r for r in trace['rows'] if 4. <= r['episode_t'] <= 20.]
    if not rows:
        raise ValueError('Missing steady window')
    head_names = ['neck', 'neck_pitch', 'yaw_roll_motion', 'jaw_soft']
    total = sum(mass.values())
    head_mass = sum(mass[k] for k in head_names)
    t = np.array([r['episode_t'] for r in rows])
    roots = np.array([r['body']['base_pos'] for r in rows])
    com, head_relative, support_relative, trunk_relative, com_trunk_relative, pitch, contacts = [], [], [], [], [], [], []
    for r in rows:
        bodies = {b['name']: b for b in r['raw']['body_states']}
        if not mass.keys() <= bodies.keys():
            raise ValueError('Whole-robot COM requires every massive robot body')
        pos = {k: np.array(v['pos']) for k, v in bodies.items()}
        center = sum(mass[k] * pos[k] for k in mass) / total
        head = sum(mass[k] * pos[k] for k in head_names) / head_mass
        w, x, y, z = r['body']['base_quat']
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        head_relative.append((head-pos['trunk_base']) @ heading)
        midpoint = (pos['ankle_left']+pos['ankle_right'])/2
        support_relative.append((center-midpoint) @ heading)
        trunk_relative.append((pos['trunk_base']-midpoint) @ heading)
        com_trunk_relative.append((center-pos['trunk_base']) @ heading)
        com.append(center)
        pitch.append(np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))))
        contacts.append([bodies[k]['ground_contact'] for k in ['ankle_left','ankle_right']])
    w, x, y, z = rows[0]['body']['base_quat']
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
    duration = t[-1]-t[0]
    if duration <= 0:
        raise ValueError('Window must span time')
    q = np.array([r['raw']['q'] for r in rows])
    output = np.array([r['experiment_policy_action'] for r in rows])
    targets = np.array([r['ctrl'] for r in rows])
    contact = np.array(contacts, bool)
    return dict(trunk_net_mps=float((roots[-1]-roots[0]) @ heading/duration),
                com_net_mps=float((com[-1]-com[0]) @ heading/duration),
                neck_angle_deg=float(np.rad2deg(q[:,5]).mean()),
                head_angle_deg=float(np.rad2deg(q[:,6]).mean()),
                raw_policy_neck_action=float(output[:,5].mean()),
                neck_target_deg=float(np.rad2deg(targets[:,5]).mean()),
                head_neck_relative_trunk_mm=float(np.mean(head_relative)*1000),
                com_relative_ankle_midpoint_mm=float(np.mean(support_relative)*1000),
                trunk_relative_ankle_midpoint_mm=float(np.mean(trunk_relative)*1000),
                com_relative_trunk_mm=float(np.mean(com_trunk_relative)*1000),
                trunk_pitch_mean_deg=float(np.mean(pitch)),
                trunk_pitch_std_deg=float(np.std(pitch)),
                both_feet_contact_fraction=float(contact.all(-1).mean()),
                no_feet_contact_fraction=float((~contact.any(-1)).mean()),
                valid_steady_window=not any(r['fell'] or r['tilt']>60. for r in trace['rows'] if r['episode_t']<=20.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    root = args.session.resolve()
    spec = json.loads((root/'robot_spec.json').read_text())
    mass = {b['name']: b['mass'] for b in spec['bodies'] if b['name'] not in ['world','ball']}
    completed = json.loads((root/'evaluation_completed.json').read_text())
    result = {}
    successes = {}
    for row in completed:
        label = row['label']
        suite = json.loads((root/label/'suite/summary.json').read_text())
        successes[label] = {(e['case'],e['seed']) for e in suite['episodes'] if e['task_metrics']['success']}
        long = [dict(seed=e['seed'],**measure(e['trace'],mass)) for e in suite['episodes'] if e['case']=='sprint_long']
        result[label] = dict(**row, long=long,
                             mean_long_speed=float(np.mean(row['long_speed'])),
                             mean_net_speed=float(np.mean([x['trunk_net_mps'] for x in long])),
                             mean_com_speed=float(np.mean([x['com_net_mps'] for x in long])),
                             valid_steady_windows=sum(x['valid_steady_window'] for x in long))
    for label, row in result.items():
        base_label = f"v{round(row['speed']*100):02d}_neck00"
        base = result[base_label]
        row['lost_baseline_successes'] = sorted(successes[base_label]-successes[label])
        row['net_speed_ratio_to_baseline'] = row['mean_net_speed']/base['mean_net_speed']
        valid = row['valid_steady_windows']==len(row['long']) and base['valid_steady_windows']==len(base['long'])
        row['posture_changes'] = ({key:float(np.mean([x[key] for x in row['long']])-np.mean([x[key] for x in base['long']]))
                                   for key in ['neck_angle_deg','neck_target_deg','head_angle_deg','raw_policy_neck_action','head_neck_relative_trunk_mm','com_relative_ankle_midpoint_mm','trunk_relative_ankle_midpoint_mm','com_relative_trunk_mm','trunk_pitch_mean_deg']}
                                  if valid else None)
        row['screen_signal'] = bool(row['neck_degrees'] != 0 and row['falls']==0 and not row['lost_baseline_successes'] and
                                     ((row['speed']==.3 and row['passes']==32 and row['net_speed_ratio_to_baseline']>=1.05) or
                                      (row['speed']==.45 and row['passes']>=base['passes']+4 and row['net_speed_ratio_to_baseline']>=1.)))
    output = dict(results=result, any_screen_signal=any(x['screen_signal'] for x in result.values()),
                  default_promoted=False, definitions='Long cases 4–20s, all seeds including failed windows. Net displacement projected along heading at4s. Mass-weighted native COM poses exclude ball/world. Ankle midpoint is not center of pressure; contacts are not support forces or exact sole slip. Mean posture differences are descriptive, invalid steady windows flagged. Screen signal is not deployment acceptance.')
    (root/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:{key:v[key] for key in ['passes','falls','mean_net_speed','net_speed_ratio_to_baseline','screen_signal','posture_changes']} for k,v in result.items()},indent=2))


if __name__ == '__main__':
    main()
