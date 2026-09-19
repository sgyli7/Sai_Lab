"""Check real native movement, keyboard ownership and the preserved physical step."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('result',type=Path)
a=p.parse_args()
d=json.loads(a.result.read_text())
s=d['samples']
def at(time):
    return min(s,key=lambda row:abs(row['time']-time))
c=[row['carrier'] for row in s]
checks={
    'completed_60_simulation_seconds':d['seconds']>=60,
    '154_bodies':all(row['bodies']==154 for row in c),
    'no_connector_failures':all(row['jel_failures']==0 and row['cargo_failures']==0 for row in c),
    'forward_motion':at(25)['carrier']['position'][0]-at(1)['carrier']['position'][0]>.5,
    'reverse_motion':at(49)['carrier']['speed_m_s']<-.03,
    'steering_request':any(row['command']==[.2,.003] for row in c),
    'measured_turn':abs(at(25)['carrier']['yaw']-at(10)['carrier']['yaw'])>.0001,
    'released_command':at(30)['carrier']['requested']==[0.,0.],
    'sai_selection_stops_carrier':not at(52)['carrier']['selected'] and at(52)['carrier']['requested']==[0.,0.],
    'space_stops_command':at(56)['carrier']['requested']==[0.,0.],
    'no_sai_input_leak':all(row['held']==[0.,0.,0.] for row in s if row['carrier']['selected']),
    'stable':min(row['upright'] for row in c)>.99,
    'time_rates_exercised':{.1,1.,3.}.issubset({row['time_scale'] for row in s}),
    'unchanged_integration_step':max(abs(row['physics_delta']-.0005) for row in s)<1e-10,
}
print(json.dumps(dict(passed=all(checks.values()),checks=checks,
    final_position=c[-1]['position'],final_speed_m_s=c[-1]['speed_m_s']),indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
