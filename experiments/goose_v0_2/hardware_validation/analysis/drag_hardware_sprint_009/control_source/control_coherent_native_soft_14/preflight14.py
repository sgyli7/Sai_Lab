import os,sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';sys.dont_write_bytecode=True
from pathlib import Path
import json,gzip,hashlib,numpy as np,mujoco as mj
R=Path(__file__).parent;sys.path.insert(0,str(R));A=R.parent/'coherent_native_soft_13_precision_online1000/attempt_001';O=R/'readbacks14';O.mkdir(exist_ok=True)
def blocked(*a,**k):raise AssertionError('NO PHYSICAL STEP candidate14 preparation')
mj.mj_step=blocked
from teacher_mit_profile import TeacherMITDrive
from native_dispatch_witness import witness
from native_error_certificate import certificate
E=TeacherMITDrive(R/'scene');m=E.model;save=json.loads((A/'true_terminal_poststate.json').read_text());failed=save['metadata']['drive_FF_refresh']['actual_blended_native_witness'];rows=[];fixtures=[]
with gzip.open(A/'substeps.jsonl.gz','rt') as f:
 for line in f:
  x=json.loads(line)
  if any(abs(x['pre_time_s']-t)<1e-8 for t in [0.,.8,5.8]):fixtures.append((str(round(x['pre_time_s'],3)),x['pre_qpos'],x['pre_qvel'],x['actual_torque18']))
  if len(fixtures)==3:break
fixtures.append(('failure6.290_candidate_not_dispatched',save['qpos'],save['qvel'],failed['total_tau18']))
for name,q,v,tau in fixtures:
 w=witness(E,q,v,tau);c=w['numerical_certificate'];mappinglaw=max(w[k] for k in ['mapping_identity_max','equality_acceleration_max','contact_KKT_max','contact_circle_excess_max','dry_friction_mode_max']);passed=w['all_finite'] and w['source_legal'] and mappinglaw<=1e-6 and c['valid'];row={'case':name,'old_1e6_would_fail':w['old_1e6_force_balance_would_fail'],'raw_force_balance':w['force_balance_max'],'mapping_and_otherlaw_max':mappinglaw,'new_numerical_certificate_pass':c['valid'],'new_dispatch_witness_pass':bool(passed),'max_5ms_point_m':c.get('max_point_5ms_m'),'max_5ms_translation_m':c.get('max_translation_5ms_m'),'max_5ms_rotation_rad':c.get('max_rotation_5ms_rad')};rows.append(row);(O/(name+'.json')).write_text(json.dumps(w,indent=2));assert passed,row;print(row,flush=True)
 if name.startswith('failure'):
  d=mj.MjData(m);d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=tau;mj.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mj.mj_fullM(m,d,M);r=M@d.qacc-d.qfrc_smooth-d.qfrc_constraint;bad=certificate(m,d,M,r*1000);badM=M.copy();badM[0,0]=-abs(badM[0,0]);nonSPD=certificate(m,d,badM,r);nanr=r.copy();nanr[0]=np.nan;nonfinite=certificate(m,d,M,nanr);assert not bad['valid'] and not nonSPD['valid'] and not nonfinite['valid'];(O/'negative_mathematical_inputs.json').write_text(json.dumps({'scaled_residual_1000x':bad,'nonSPD':nonSPD,'nonfinite_r':nonfinite},indent=2));np.savez_compressed(O/'same_failure_certificate_arrays.npz',qpos=q,qvel=v,total_tau=tau,M=M,r=r,qacc=d.qacc)
(R/'preflight14_readback.json').write_text(json.dumps({'physical_steps':0,'LP_calls':0,'rows':rows,'negative_inputs_rejected':True,'intrinsic_newton_error_only_not_full_trajectory_guarantee':True,'original_forcebalance_guard_fail_preserved':True},indent=2))
