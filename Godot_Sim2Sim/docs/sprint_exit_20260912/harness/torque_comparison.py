"""Same-state electrical/viscous comparison, never a replacement simulator.

Read fixed BAM source listed in bam_source/sources.json. Re-express only its
scalar voltage equations with nominal kp/kd. Do not import downloaded code.
Constraint friction, load/supply histories, delay, contacts and rotor mapping
are deliberately not inferred from incomplete Jolt telemetry.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'torque_comparison';out.mkdir(exist_ok=False)
p=json.loads((r/'bam_source/bam/params/xl330/m6.json').read_text())
kt=p['kt'];resistance=p['R'];gain=4096/(2*math.pi)/(256*885);current=1.75
datasets={'donor':list((r/'running_zero_shot/suite').glob('fast_source_*/attempt_01/trace.json')),
          's05':[r/'state_shadow/trace.json']}
records={}
def stats(x):return dict(mean_abs=float(np.abs(x).mean()),rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.quantile(np.abs(x),.95)),max_abs=float(np.abs(x).max()))
for label,paths in datasets.items():
 rows=[];sources=[]
 for path in paths:
  payload=path.read_bytes();d=json.loads(payload)
  first=d['summary'].get('first_fall');selected=[row for row in d['rows'] if row['skill']=='sprint' and not row['fell'] and (first is None or row['t']<first)]
  rows.extend(selected);sources.append(dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),upright_sprint_rows=len(selected)))
 q=np.asarray([row['raw']['q'] for row in rows]);dq=np.asarray([row['raw']['qd'] for row in rows]);target=np.asarray([row['ctrl'] for row in rows]);error=target-q
 # Hold the old dry-friction term out of both sides. BAM applies its dry
 # friction through a different, stateful constraint formulation.
 drive=np.clip(.55*error,-kt*current,kt*current)
 jolt=drive-.053*dq
 voltages={}
 for vin in [6.5,7.5,8.2]:
  requested=error*200*gain;center=kt*dq/vin;span=resistance*current/vin
  limited=np.clip(requested,center-span,center+span);duty=np.clip(limited,-1,1)
  bam=kt*(vin*duty-kt*dq)/resistance-p['friction_viscous']*dq
  difference=bam-jolt
  small=(np.abs(error)<.5)&(np.abs(dq)<5.)
  voltages[str(vin)]=dict(difference=stats(difference),small_signal_difference=stats(difference[small]),
   fraction_difference_over_0_1=float(np.mean(np.abs(difference)>.1)),current_limiter_fraction=float(np.mean(np.abs(limited-requested)>1e-9)),pwm_saturation_fraction=float(np.mean(np.abs(limited)>1)),sign_disagreement_above_0_02=float(np.mean((bam*jolt<0)&(np.abs(bam)>.02)&(np.abs(jolt)>.02))))
 records[label]=dict(sources=sources,rows=len(rows),joint_samples=int(q.size),position_error=stats(error),joint_speed=stats(dq),jolt_pre_dry_friction_torque=stats(jolt),jolt_drive_clamp_fraction=float(np.mean(np.abs(.55*error)>kt*current)),voltage_cases=voltages)
 np.savez_compressed(out/(label+'.npz'),q=q,qd=dq,target=target,jolt_pre_dry_friction_torque=jolt)
result=dict(scope='Counterfactual same-state motor plus viscous terms at 50 Hz observation snapshots, not full BAM or actual measured net torques',limitations=['No delay, battery sag, domain randomization, dry/load/constraint friction, contacts or rotor-map equivalence','Only pre-first-fall sprint rows; donor states do not establish the cause of falling','No physics parameters changed and no MuJoCo rollout claim'],bam_revision='62bd8ce12154340be97e06f7f41a0ca8f116d967',nominal_small_signal=dict(kp=200*gain*7.5*kt/resistance,damping=kt*kt/resistance+p['friction_viscous']),records=records)
atomic_json(out/'completed.json',result)
print({name:record['voltage_cases']['7.5'] for name,record in records.items()},flush=True)
