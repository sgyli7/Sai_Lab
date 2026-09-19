"""Report frame delivery and achieved time rate separately; require both to pass."""
import argparse
import json
from pathlib import Path
from statistics import median

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('result',type=Path)
a=p.parse_args()
d=json.loads(a.result.read_text())
s=[v for v in d['samples'] if v['time']>=1.]
# Only measure the final uninterrupted requested-rate segment.
rate=float(s[-1].get('time_scale',1.))
for index in range(len(s)-2,-1,-1):
    if float(s[index].get('time_scale',1.))!=rate:
        s=s[index+1:];break
if len(s)<2:raise SystemExit('Not enough steady-rate samples')
fps=median(v['fps'] for v in s)
ratio=(s[-1]['time']-s[0]['time'])/((s[-1]['wall_usec']-s[0]['wall_usec'])/1e6)
frame_pass=fps>=24
time_pass=ratio>=rate*.95
print(json.dumps(dict(passed=frame_pass and time_pass,frame_delivery_passed=frame_pass,
    requested_rate=rate,median_fps=fps,simulation_to_wall_ratio=ratio,requested_time_rate_passed=time_pass),indent=2))
raise SystemExit(0 if frame_pass and time_pass else 1)
