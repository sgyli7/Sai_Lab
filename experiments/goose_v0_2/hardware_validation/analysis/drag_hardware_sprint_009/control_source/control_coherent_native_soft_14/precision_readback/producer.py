import os,sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';sys.dont_write_bytecode=True
from pathlib import Path
import json,gzip,pickle,hashlib,time,numpy as np,mujoco as mj
B=Path(__file__).parent;R=B/'coherent_native_soft_13_online1000';A=R/'attempt_001';O=B/'coherent_native_soft_13_precision_singlepoint';O.mkdir(exist_ok=False);sys.path.insert(0,str(R))
def blocked(*a,**k):raise RuntimeError('mj_step forbidden precision audit')
mj.mj_step=blocked
from teacher_mit_profile import TeacherMITDrive
from drive_ff200 import refresh
import soft_phase_model as soft
from types import SimpleNamespace
saved=json.loads((A/'true_terminal_poststate.json').read_text());meta=saved['metadata'];failed=meta['drive_FF_refresh'];last=None
with gzip.open(A/'substeps.jsonl.gz','rt') as f:
 for line in f:last=json.loads(line)
ctx=json.loads((A/'actual_capture_to_motion_context.json').read_text());reference={k:np.array(meta[v]) for k,v in [('qpos','reference_qpos'),('qvel','reference_qvel'),('qacc','reference_qdd')]};original_linprog=soft.linprog;calls=[];results=[]
for label,tight in [('original',False),('tight_1e9',True)]:
 e=TeacherMITDrive(R/'scene');e.data.qpos[:]=saved['qpos'];e.data.qvel[:]=saved['qvel'];e.data.ctrl[:]=np.load(A/'true_terminal_poststate.npz')['ctrl'];e.data.time=saved['time_s'];e.original.target[:]=last['actual_slew_target18'];e.command_details=last['motor_profile_details'];mj.mj_forward(e.model,e.data)
 def call(c,**kwargs):
  matrix={k:v for k,v in kwargs.items() if k!='options'};matrix['c']=c;data=pickle.dumps(matrix,protocol=5);options=dict(kwargs['options'])
  if tight:options.update(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
  kwargs['options']=options;calls.append({'label':label,'matrix_sha256':hashlib.sha256(data).hexdigest(),'method':kwargs['method'],'options':options});np.savez_compressed(O/(label+'_matrix.npz'),c=c,A_eq=kwargs['A_eq'],b_eq=kwargs['b_eq'],A_ub=kwargs['A_ub'],b_ub=kwargs['b_ub']);return original_linprog(c,**kwargs)
 soft.linprog=call
 motion=SimpleNamespace(env=e,entry=SimpleNamespace(held_FF=np.array(ctx['held_motor_command']['tau_ff18_Nm'])),curve=SimpleNamespace(sample=lambda t:reference),last_ff=np.array(failed['previous_actual_FF18']),consecutive=0,fallback_total=1,current_drive_meta={})
 result=refresh(motion,meta['trajectory_packet'],failed['elapsed_s'],e.target.copy());(O/(label+'.json')).write_text(json.dumps(result,indent=2));results.append({'label':label,'abort':result['abort_reason'],'LP_status':result['soft_phase']['status'],'success':result['soft_phase']['success'],'hard_eq':result['soft_phase'].get('hard_equality_residual'),'hard_ineq':result['soft_phase'].get('hard_inequality_residual'),'circle':result['soft_phase'].get('maximum_contact_circle_excess'),'source_model_error':result['soft_phase'].get('source_applied_model_error'),'LP_wall_s':result['LP_wall_s'],'native_witness':{k:result['actual_blended_native_witness'][k] for k in ['force_balance_max','equality_acceleration_max','contact_KKT_max','contact_circle_excess_max','dry_friction_mode_max','source_legal']}});print(results[-1],flush=True)
soft.linprog=original_linprog
report={'physical_steps':0,'LP_calls':2,'saved_failed_hard_eq':failed['soft_phase']['hard_equality_residual'],'original_reproduced':abs(results[0]['hard_eq']-failed['soft_phase']['hard_equality_residual'])<1e-12,'same_exact_LP_matrix':calls[0]['matrix_sha256']==calls[1]['matrix_sha256'],'method':'highs unchanged (not forced IPM)','calls':calls,'results':results,'same_qv_target_packet_reference_caps_weights':True,'source_actual13_unchanged':True};(O/'comparison.json').write_text(json.dumps(report,indent=2));print('SUMMARY',report['original_reproduced'],report['same_exact_LP_matrix'],flush=True)
