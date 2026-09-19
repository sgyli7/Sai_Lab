"""One preregistered, train-seed-only fit of passive per-hinge stiffness."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import mujoco
from sim2sim.coords import mat_to_quat_wxyz
from sim2sim.paths import load_robot_json
from sim2sim.research.joint_compliance import add_joint_compliance_direct
from sim2sim.research.joint_residuals import JointResidualModel
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913')

def main():
 if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R.resolve():raise RuntimeError('Supervision required')
 atomic_json(R/'fit_protocol.json',dict(source='Previous frozen direct20k_hard_contact prediction',training_seeds=list(range(927000,927008)),validation_seeds=list(range(927008,927016)),fit_window_s=[.04,1.],rule='Per hinge nonnegative scalar least squares of native anchor translation against proxy translation; stiffness=20000/scale, bounded [5000,80000], damping stays 200. All three parent axes pooled. No validation-based scale or subset choice.',preflight_gate='At least 20% reduction in train aggregate anchor-vector MSE against unchanged profile; otherwise no new physics run. Per-hinge negative alignment is a model limitation, not repaired with a signed spring.',physical_gate='Existing whole response/manifold gate unchanged in a single fitted-profile experiment; a geometric fit is not itself eligibility.',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'));source=json.loads(Path(cfg['godot_spec']).read_text());jmodel=JointResidualModel(source)
 spec=mujoco.MjSpec.from_file(cfg['mjcf']);add_joint_compliance_direct(spec,cfg['godot_spec'],20000.,200.)
 m=spec.compile();d=mujoco.MjData(m)
 proxy=np.load('results/sprint_contact_calibration_20260913/compliance_direct_probe/direct20k_hard_contact.npz')
 qpos=proxy['qpos'];joints=proxy['joints'];native=np.load(R/'joint_residuals.npz')['translation_parent'][:101]
 pred=np.empty_like(native);qcheck=0.
 for i in range(101):
  for seed in range(16):
   d.qpos[:]=qpos[i,seed];mujoco.mj_kinematics(m,d)
   poses=[dict(name=m.body(b).name,pos=d.xipos[b].copy(),quat=mat_to_quat_wxyz(d.ximat[b].reshape(3,3))) for b in range(1,m.nbody)]
   result=jmodel.measure(jmodel.body_poses(poses),joints[i,seed]);pred[i,seed]=result['translation_parent'];qcheck=max(qcheck,float(np.abs(result['q_error']).max()))
 assert qcheck<1e-10,qcheck
 time=np.arange(101)*.02;mask=(time>=.04-1e-8)&(time<1-1e-8)
 scale=np.sum(pred[mask,:8]*native[mask,:8],axis=(0,1,3))/np.maximum(np.sum(pred[mask,:8]**2,axis=(0,1,3)),1e-24)
 bounded=np.clip(scale,.25,4);profile={h.name:[float(20000/s),200.] for h,s in zip(jmodel.hinges,bounded)}
 fits={}
 for split,ids in [('train',slice(0,8)),('validation',slice(8,16))]:
  fits[split]={}
  for phase,(lo,hi) in [('landing',(.04,.3)),('standing',(.3,1.)),('moving',(1.,1.3)),('fit_window',(.04,1.))]:
   phase_mask=(time>=lo-1e-8)&(time<hi-1e-8)
   a=pred[phase_mask,ids];b=native[phase_mask,ids]
   old=float(np.mean((a-b)**2));new=float(np.mean((a*bounded[None,None,:,None]-b)**2))
   fits[split][phase]=dict(mse_original=old,mse_scaled=new,ratio=new/old)
 perjoint=[]
 for i,h in enumerate(jmodel.hinges):
  a=pred[mask,:8,i];b=native[mask,:8,i]
  cosine=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
  perjoint.append(dict(name=h.name,scale_raw=float(scale[i]),scale_bounded=float(bounded[i]),stiffness=profile[h.name][0],cosine=cosine,proxy_rms_m=float(np.sqrt(np.mean(a*a))),native_rms_m=float(np.sqrt(np.mean(b*b)))))
 eligible=fits['train']['fit_window']['ratio']<=.8 and bool(np.all(scale>0))
 atomic_json(R/'fit_completed.json',dict(completed=True,profile=profile,metrics=fits,per_joint=perjoint,eligible_for_single_physical_probe=eligible,scope='Linear prediction scaling is only a fit preflight; new physics and unchanged full gates still required'))
 np.savez_compressed(R/'fit_arrays.npz',proxy_translation=pred,native_translation=native,raw_scale=scale,used_scale=bounded)
 print(json.dumps(dict(eligible=eligible,metrics=fits,per_joint=perjoint)),flush=True)

if __name__=='__main__':main()
