"""Attribute same-frame chain geometry errors, without simulating corrections."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from sim2sim.paths import load_robot_json
from sim2sim.research.joint_residuals import JointResidualModel
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913')

def main():
 if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R.resolve():raise RuntimeError('Supervision required')
 capture=json.loads((R/'capture/completed.json').read_text());assert capture['completed']
 cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'));spec=json.loads(Path(cfg['godot_spec']).read_text());model=JointResidualModel(spec)
 protocol=dict(scope='Offline counterfactual kinematics, not physical trajectories or policy improvements',seeds=[e['seed'] for e in capture['results']],q_error_cap_rad=1e-5,both_reconstruction_cap_m=1e-10,prior_rigid_manifold_cap_m=1e-7,phases=dict(precontact=[0,.04],landing=[.04,.30],standing=[.3,1.],forward=[1.,3.],left=[3.,6.],reverse=[6.,9.]),modes=['rigid','translation','swing','both'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/sim2sim/research/joint_residuals.py')]})
 atomic_json(R/'analysis_protocol.json',protocol)
 translations=[];swings=[];qerrors=[];vectors={k:[] for k in protocol['modes']};reconstruction_errors=[]
 names=['ankle_left','ankle_right']
 for entry in capture['results']:
  rows=json.loads((Path(entry['directory'])/'trace.json').read_text())['rows']
  tr=[];sw=[];qe=[];vec={k:[] for k in vectors};errs=[]
  for row in rows:
   raw=row['raw'];poses=raw['diagnostic_joint_poses'];body=model.body_poses(poses);measured=model.measure(body,raw['q'])
   tr.append(measured['translation_parent']);sw.append(measured['swing_parent']);qe.append(measured['q_error'])
   actual={p['name']:np.asarray(p['pos']) for p in poses}
   for mode,t,s in [('rigid',False,False),('translation',True,False),('swing',False,True),('both',True,True)]:
    pred=model.reconstruct(body,raw['q'],measured,translation=t,swing=s)
    vec[mode].append(np.array([actual[n]-pred[n][0] for n in names]))
    if mode=='both':errs.append(max(np.linalg.norm(actual[n]-pred[n][0]) for n in actual))
  translations.append(tr);swings.append(sw);qerrors.append(qe);reconstruction_errors.append(errs)
  for k in vectors:vectors[k].append(vec[k])
  print('analyzed',entry['seed'],flush=True)
 # Time, seed, joint/body, xyz. Retain development split for future calibration.
 tr=np.swapaxes(np.array(translations),0,1);sw=np.swapaxes(np.array(swings),0,1);qe=np.swapaxes(np.array(qerrors),0,1)
 vectors={k:np.swapaxes(np.array(v),0,1) for k,v in vectors.items()}
 qmax=float(np.abs(qe).max());rmax=float(np.max(reconstruction_errors))
 prior=np.load('results/sprint_contact_calibration_20260913/manifold_residuals.npz')['position_vector'][:450]
 parity=float(np.abs(vectors['rigid']-prior).max())
 checks=dict(q_max_abs_rad=qmax,q_passed=qmax<=1e-5,both_reconstruction_max_m=rmax,both_passed=rmax<=1e-10,prior_manifold_max_m=parity,prior_manifold_passed=parity<=1e-7)
 atomic_json(R/'analysis_canary.json',checks)
 assert all(checks[k] for k in ['q_passed','both_passed','prior_manifold_passed']),checks
 np.savez_compressed(R/'joint_residuals.npz',translation_parent=tr,swing_parent=sw,q_error=qe,**{'foot_'+k:v for k,v in vectors.items()})
 metrics={};time=np.arange(450)*.02
 for phase,(lo,hi) in protocol['phases'].items():
  mask=(time>=lo-1e-8)&(time<hi-1e-8)
  perjoint=[]
  for i,h in enumerate(model.hinges):
   tv=tr[mask,:,i];sv=sw[mask,:,i]
   tnorm=np.linalg.norm(tv,axis=-1);snorm=np.linalg.norm(sv,axis=-1)
   perjoint.append(dict(name=h.name,translation_median_m=float(np.median(tnorm)),translation_p95_m=float(np.quantile(tnorm,.95)),translation_max_m=float(tnorm.max()),translation_mean_parent_m=tv.mean(axis=(0,1)).tolist(),swing_median_deg=float(np.rad2deg(np.median(snorm))),swing_p95_deg=float(np.rad2deg(np.quantile(snorm,.95))),swing_max_deg=float(np.rad2deg(snorm.max()))))
  modes={k:dict(mse_m2=float(np.mean(v[mask]**2)),median_m=float(np.median(np.linalg.norm(v[mask],axis=-1))),p95_m=float(np.quantile(np.linalg.norm(v[mask],axis=-1),.95))) for k,v in vectors.items()}
  for k in modes:modes[k]['mse_ratio_to_rigid']=modes[k]['mse_m2']/modes['rigid']['mse_m2']
  metrics[phase]=dict(joints=perjoint,foot_reconstruction=modes)
 atomic_json(R/'analysis_completed.json',dict(completed=True,checks=checks,metrics=metrics,joint_order=[h.name for h in model.hinges],scope=protocol['scope']))
 print(json.dumps(dict(checks=checks,ratios={p:{k:round(v['mse_ratio_to_rigid'],6) for k,v in x['foot_reconstruction'].items()} for p,x in metrics.items()})),flush=True)

if __name__=='__main__':main()
