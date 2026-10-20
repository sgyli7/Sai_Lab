import os,sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';sys.dont_write_bytecode=True
from pathlib import Path
import json,gzip,time,hashlib,importlib.util,numpy as np,mujoco as mj
R=Path(__file__).parent;sys.path.insert(0,str(R));A=R.parent/'coherent_native_soft_12_dependencyfix_online1000/attempt_002';O=R/'readbacks13';O.mkdir(exist_ok=True)
def blocked(*a,**k):raise RuntimeError('PHYSICAL_INTEGRATION_FORBIDDEN_PREPARATION')
mj.mj_step=blocked
from teacher_mit_profile import TeacherMITDrive
from native_coherent_motion import NativeCoherentMotion
from teacher_drive_segment import compute_drive_tick,sample_packet
E=TeacherMITDrive(R/'scene');E.torque_mask=np.arange(18)>=6;ctx=json.loads((A/'actual_capture_to_motion_context.json').read_text());E.data.qpos[:]=ctx['actual_qpos'];E.data.qvel[:]=ctx['actual_qvel'];E.original.target[:]=ctx['original_env_target'];E.command_details=ctx['held_motor_command'];mj.mj_forward(E.model,E.data);E.reference_output_dir=O/'generated_reference';motion=NativeCoherentMotion(E,ctx,ctx['prefix_epoch_slip_R_L']);rows={};previous={};wanted=[5.8,6.2,6.265,6.4,6.7];last=None
with gzip.open(A/'substeps.jsonl.gz','rt') as f:
 for line in f:
  x=json.loads(line)
  for t in wanted:
   if abs(x['pre_time_s']-t)<1e-7:rows[t]=x;previous[t]=last
  last=x
commands=json.loads((A/'commands50hz.json').read_text());checks=[]
for t in [5.8,6.2,6.4,6.7]:
 x=rows[t];p=previous[t];E.data.qpos[:]=x['pre_qpos'];E.data.qvel[:]=x['pre_qvel'];E.data.ctrl[:]=x['actual_torque18'];E.data.time=t;E.original.target[:]=p['actual_slew_target18'];E.command_details=p['motor_profile_details'];mj.mj_forward(E.model,E.data);qsave=E.data.qpos.copy();vsave=E.data.qvel.copy();_,meta=motion.prepare(max(0.,x['pre_time_s']-ctx['motion_start_time_s']));old=min(commands,key=lambda z:abs(z['time_s']-t))['metadata'];pk=meta['trajectory_packet'];op=old['trajectory_packet'];err=max(float(max(abs(np.array(n['bernstein_q18']).ravel()-np.array(o['bernstein_q18']).ravel()))) for n,o in zip(pk['segments'],op['segments']));qrerr=float(max(abs(np.array(meta['motor_command']['q_ref18_rad'])-np.array(old['motor_command']['q_ref18_rad']))));checks.append(dict(time_s=t,packet_nonFF_coefficients_error=err,qref_error=qrerr,front_q_only_FF0=True,LP_result_is_None=meta['soft_phase'] is None,qv_unchanged=np.array_equal(qsave,E.data.qpos) and np.array_equal(vsave,E.data.qvel)));assert err<1e-10 and qrerr<1e-10 and meta['soft_phase'] is None,checks[-1]
results=[]
for t in [5.8,6.265,6.7]:
 x=rows[t];p=previous[t];E.data.qpos[:]=x['pre_qpos'];E.data.qvel[:]=x['pre_qvel'];E.data.ctrl[:]=x['actual_torque18'];E.data.time=t;E.original.target[:]=p['actual_slew_target18'];E.command_details=p['motor_profile_details'];mj.mj_forward(E.model,E.data);old=max((z for z in commands if z['time_s']<=t+1e-8),key=lambda z:z['time_s']);packet=old['metadata']['trajectory_packet'];motion.current_drive_meta={};qsave=E.data.qpos.copy();vsave=E.data.qvel.copy();res=motion.refresh_drive_FF(packet,max(0.,x['pre_time_s']-ctx['motion_start_time_s']),E.target.copy());_,tau,detail=compute_drive_tick(E,packet,max(0.,x['pre_time_s']-ctx['motion_start_time_s']),E.target,E.data.qpos,E.data.qvel,res['FF18']);err=float(max(abs(tau-np.array(res['preview_total18']))));res['same_drive_source_shadow_error']=err;res['qv_unchanged']=bool(np.array_equal(qsave,E.data.qpos) and np.array_equal(vsave,E.data.qvel));(O/f'drive_{t:.3f}.json').write_text(json.dumps(res,indent=2));results.append(dict(time_s=t,abort_reason=res['abort_reason'],LP_status=res['soft_phase']['status'],LP_wall_s=res['LP_wall_s'],entry_blend=res['entry_C2_blend'],same_drive_source_shadow_error=err,source_legal=res['actual_blended_native_witness']['source_legal'],hard_max=max(res['actual_blended_native_witness'][k] for k in ['force_balance_max','equality_acceleration_max','contact_KKT_max','contact_circle_excess_max','dry_friction_mode_max']),qv_unchanged=res['qv_unchanged']));assert err<1e-9 and res['qv_unchanged'],results[-1];print('DRIVE',results[-1],flush=True)
# Original bounded decision function, synthetic result only: exactly two
# accepted drive holds then rejection. Does not call solver or integrate.
from soft_phase_model import dispatch_or_validated_witness
count=total=0;fall=[]
for _ in range(3):
 z=dispatch_or_validated_witness({'success':False,'status':'synthetic_only_NotSet','validated_witness':{'valid':True}},np.zeros(18),count,total);fall.append(z);count=z['consecutive'];total=z['total']
assert [z['accepted'] for z in fall]==[True,True,False]
(R/'preflight13_readback.json').write_text(json.dumps(dict(physical_steps=0,nonFF50Hz_shadow=checks,drive200Hz_shadow=results,fallback_branch_readback=fall,fallback_time_bound_s=.010,no_new_actual_authorization=True),indent=2));print('PREFLIGHT13_DONE',flush=True)
