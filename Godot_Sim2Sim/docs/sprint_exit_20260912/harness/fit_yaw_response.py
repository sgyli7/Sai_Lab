"""Identify nominal turn-command response from already frozen development traces."""
from pathlib import Path
import json,math
import numpy as np
from scipy.optimize import least_squares
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'yaw_identification';out.mkdir(exist_ok=False)
episodes=json.loads((r/'baseline_exits/suite/summary.json').read_text())['episodes'];responses=[];inputs=[];initial=[]
for e in episodes:
 rows=json.loads(Path(e['trace']).read_text())['rows'];sign=1 if e['case']=='sprint_left' else -1
 yaw=np.unwrap([math.atan2((m:=quat_wxyz_to_mat(row['body']['base_quat']))[1,0],m[0,0]) for row in rows])
 rate=np.gradient(yaw,.02)*sign
 responses.append(rate[150:225]);inputs.append(np.array([row['command'][2]*sign for row in rows[150:225]]));initial.append(rate[140:150].mean())
actual=np.mean(responses,axis=0);command=np.mean(inputs,axis=0);v0=np.mean(initial)
def predict(parameters):
 gain,tau=parameters;a=math.exp(-.02/tau);v=v0;values=[]
 for u in command:
  values.append(v);v=a*v+(1-a)*gain*u
 return np.array(values)
fit=least_squares(lambda x:predict(x)-actual,[.8,.2],bounds=([.05,.02],[3.,2.]))
gain,tau=fit.x;pred=predict(fit.x);r2=1-np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2)
kd=max(0.,(2*math.sqrt(tau*gain*6.)-1.)/gain)
result=dict(scope='Averaged measured left/right onset in 32 development episodes; one-pole approximation, not a validated whole-body dynamics model',gain=float(gain),tau_seconds=float(tau),r2=float(r2),pd_heading_kp=6.,critical_damping_kd=float(kd),eligible_for_single_control_probe=bool(r2>=.7 and kd>0),conditions='No gain sweep; only test derived damping if fit explains at least 70% onset variance',sources=[e['trace'] for e in episodes])
atomic_json(out/'completed.json',result);np.savez(out/'fit.npz',actual=actual,command=command,predicted=pred)
print({k:v for k,v in result.items() if k!='sources'},flush=True)
