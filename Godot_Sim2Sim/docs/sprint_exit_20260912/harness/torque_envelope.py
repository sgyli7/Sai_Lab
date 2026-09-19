from pathlib import Path
import json,math
import numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912')
p=json.loads((r/'bam_source/bam/params/xl330/m6.json').read_text());kt=p['kt'];res=p['R'];limit=kt*1.75
result={}
for label in ['donor','s05']:
 x=np.load(r/'torque_comparison'/(label+'.npz'));dq=x['qd'];error=x['target']-x['q'];voltages={}
 for vin in [6.5,7.5,8.2]:
  duty=np.clip(np.clip(error*200*4096/(2*math.pi)/(256*885),(kt*dq-res*1.75)/vin,(kt*dq+res*1.75)/vin),-1,1)
  desired=kt*(vin*duty-kt*dq)/res-p['friction_viscous']*dq
  required_drive=desired+.053*dq
  shortfall=np.maximum(np.abs(required_drive)-limit,0.)
  voltages[str(vin)]=dict(infeasible_fraction=float(np.mean(shortfall>1e-6)),p95_shortfall=float(np.quantile(shortfall,.95)),max_shortfall=float(shortfall.max()))
 result[label]=voltages
atomic_json(r/'torque_comparison/envelope.json',dict(scope='Desired BAM electrical/viscous torque outside attainable frozen Jolt drive-minus-viscous envelope, excluding both dry friction models',equation='required_Jolt_drive = BAM_electrical_viscous_torque + 0.053 * qdot; feasible iff abs(drive) <= kt * 1.75',results=result))
print(result,flush=True)
