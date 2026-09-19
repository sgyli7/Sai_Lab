"""Independent continuous-flight checks from actual Godot/Jolt telemetry."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('artifacts'));p.add_argument('--prefix',default='godot-stairs-');p.add_argument('--out',type=Path,required=True);p.add_argument('--require-pass',action='store_true');a=p.parse_args()
rows=[];engines=set();physics_rates=set()
for f in sorted(a.directory.glob(a.prefix+'*.json')):
 raw=json.loads(f.read_text())
 if 'suite' in raw and 'cases' in raw:continue  # Previous aggregate, not a raw run.
 ss=raw['samples'];last=ss[-1];engines.add(raw['engine']);physics_rates.add(raw['physics_hz'])
 xyz=np.array([s['base_position'] for s in ss]);up=np.array([s['upright'] for s in ss]);wheel=np.array(last['wheel_positions'])
 settled=[s for s in ss if s['time']>last['time']-1]
 final_ground=0 if raw['descending'] else 4*raw['riser']
 checks=dict(all_wheels_cleared=wheel[:,0].min()>.45+3*raw.get('tread',.18)+.05,three_second_stop=raw['cleared_at']>=0 and last['time']-raw['cleared_at']>=2.999,
    upright_on_final_level=up[-1]>.9,remained_in_lane=abs(xyz[:,1]).max()<.3,
    height_on_final_level=abs(xyz[-1,2]-(final_ground+.2192-.035*last.get('effective_crouch',0.)))<.02,
    four_wheels_supported=np.mean([s['wheels_supported']==4 for s in settled])>.7,no_fall=up.min()>.6,
    real_forward_key=any(e['key']==87 and e['pressed'] for e in raw['input_events']),
    terrain_triggered_stair_policy=any(s['controller_stage']=='stairs' for s in ss),
    full_articulation=raw['body_count']==26 and raw['hinges']==23 and raw['sliders']==2)
 rows.append(dict(riser=raw['riser'],descending=raw['descending'],duration=last['time'],final_xyz=xyz[-1].tolist(),min_upright=float(up.min()),
    stair_profile=last.get('stair_profile','stairs-dev40'),effective_crouch=last.get('effective_crouch',0.),
    tread=raw.get('tread',.18),initial_yaw=raw.get('initial_yaw',0.),
    maximum_lateral_m=float(abs(xyz[:,1]).max()),checks={k:bool(v) for k,v in checks.items()},passed=bool(all(checks.values()))))
result=dict(suite='godot-continuous-stairs-v1',engine=next(iter(engines)) if len(engines)==1 else sorted(engines),physics_hz=next(iter(physics_rates)) if len(physics_rates)==1 else sorted(physics_rates),controller_hz=50,
    sensor='Ground-only raycasts, not camera-based VLA',cases=rows,passed=bool(rows) and all(r['passed'] for r in rows))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if a.require_pass and not result['passed']:raise SystemExit(1)
