import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';sys.dont_write_bytecode=True
from pathlib import Path
import json,gzip,time,hashlib,numpy as np,mujoco as mj
B=Path(__file__).parent;R=B/'coherent_native_soft_12_dependencyfix_online1000';A=R/'attempt_002';O=B/'fresh_drive_ff_audit12';O.mkdir(exist_ok=False);sys.path.insert(0,str(R))
def block(*a,**k):raise RuntimeError('mj_step FORBIDDEN offline audit')
mj.mj_step=block
from teacher_mit_profile import TeacherMITDrive
from teacher_drive_segment import compute_drive_tick
from coherent_plan_sampler import CoherentPlan
from capture_plan_entry import CapturePlanEntry
from motion_stop_reference import MotionStopReference
from soft_phase_model import solve,smooth
from native_dispatch_witness import witness
import complete_cycle_plan as plan
ctx=json.loads((A/'actual_capture_to_motion_context.json').read_text());start=ctx['motion_start_time_s'];commands=json.loads((A/'commands50hz.json').read_text());wanted=[6.265,6.290,6.355,6.433];fixtures={};last=None
with gzip.open(A/'substeps.jsonl.gz','rt') as f:
 for line in f:
  x=json.loads(line)
  for t in wanted:
   if abs(x['pre_time_s']-t)<1e-7:fixtures[t]=(x,last)
  last=x
E=TeacherMITDrive(R/'scene');m=E.model;curve=MotionStopReference(CapturePlanEntry(CoherentPlan(m,A/'generated_reference/observed_capture_reference.npz',E.qidx),ctx,ctx['prefix_epoch_slip_R_L']));names=[m.actuator(i).name for i in range(18)];results=[]
def physical(total,q,v):
 d=mj.MjData(m);d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=total;mj.mj_forward(m,d);N=np.zeros(2)
 for i,c in enumerate(d.contact):
  f=np.zeros(6);mj.mj_contactForce(m,d,i,f);F=c.frame.reshape(3,3).T@f[:3]
  for k,s in enumerate(['right','left']):
   g=m.geom(s+'_actual_sole').id
   if g in (c.geom1,c.geom2) and any(m.geom(z).name in ['ground','sole_ground_box'] for z in [c.geom1,c.geom2]):N[k]+=max(0.,F[2]*(1 if c.geom2==g else -1))
 return d,N
for t in wanted:
 x,prev=fixtures[t];elapsed=t-start;c=max((cc for cc in commands if cc['time_s']<=t+1e-9),key=lambda z:z['time_s']);meta=c['metadata'];packet=meta['trajectory_packet'];ffold=np.array(x['motor_profile_details']['tau_ff18_Nm']);q=np.array(x['pre_qpos']);v=np.array(x['pre_qvel']);oldtarget=np.array(prev['actual_slew_target18']);target,oldtotal,drv=compute_drive_tick(E,packet,elapsed,oldtarget,q,v,ffold);pd=np.array(drv['PD18']);pdsource=pd.copy();pdsource[:6]=E.limit_torque(pd,v[E.drive_vidx])[:6];d,N=physical(oldtotal,q,v);ref=curve.sample(elapsed);force=plan.support_force_objective(elapsed,float(sum(N)));before=time.perf_counter();res=solve(E,d,ref['qacc'],pdsource,ffold,force['normal_soft_target_N'],100*force['weight_blend'],root_reference=(ref['qpos'],ref['qvel'],ref['qacc']),unload_side=force['unload_side']);wall=time.perf_counter()-before
 out={'time_s':t,'elapsed_s':elapsed,'held_FF_age_s':t-c['time_s'],'LP_calls_this_state':1,'LP_wall_s':wall,'LP_success':res.get('success'),'LP_status':res.get('status'),'old_dispatch_target_error':float(max(abs(target-np.array(x['actual_slew_target18'])))),'old_dispatch_tau_error':float(max(abs(oldtotal-np.array(x['actual_torque18'])))),'baseline_native_qacc_error_vs_saved':float(max(abs(d.qacc-np.array(x['pre_qacc'])))),'PD18':pd.tolist(),'held_FF18':ffold.tolist(),'old_request18':(pd+ffold).tolist(),'old_applied18':oldtotal.tolist(),'old_native_root_qacc6':d.qacc[:6].tolist(),'old_native_foot_N_R_L':N.tolist(),'force_objective':force,'same_segment_no_front_IK_recompute':True,'reference_qdd':ref['qacc'].tolist(),'reference_root_goal':res.get('root_task'),'LP_result':res,'entry_blend':float(smooth(elapsed/.5))}
 # Same geometric point acceleration: J*qdd + Jdot*qvel; no integration.
 Js=[];bias=[]
 for side in ['right','left']:
  g=m.geom(side+'_actual_sole').id;J=np.zeros((3,m.nv));mj.mj_jacGeom(m,d,J,None,g);pair=[]
  for h in [-1e-5,1e-5]:
   z=mj.MjData(m);z.qpos[:]=q;mj.mj_integratePos(m,z.qpos,v,h);mj.mj_forward(m,z);JJ=np.zeros((3,m.nv));mj.mj_jacGeom(m,z,JJ,None,g);pair.append(JJ)
  Js.append(J);bias.append((pair[1]-pair[0])@v/2e-5)
 out['old_native_sole_geom_acceleration_R_L']=[(J@d.qacc+b).tolist() for J,b in zip(Js,bias)]
 if res.get('success'):
  rawff=np.array(res['FF18']);z=out['entry_blend'];blendff=(1-z)*np.array(ctx['held_motor_command']['tau_ff18_Nm'])+z*rawff;blendff[:6]=0.
  for label,ff in [('fresh_raw',rawff),('fresh_same_entry_blend',blendff)]:
   _,tau,dd=compute_drive_tick(E,packet,elapsed,oldtarget,q,v,ff);shadow,fn=physical(tau,q,v);ww=witness(E,q,v,tau);out[label]={'FF18':ff.tolist(),'deltaFF18':(ff-ffold).tolist(),'request18':dd['requested_total18'],'applied18':tau.tolist(),'delta_applied18':(tau-oldtotal).tolist(),'native_root_qacc6':shadow.qacc[:6].tolist(),'native_foot_N_R_L':fn.tolist(),'native_sole_geom_acceleration_R_L':[(J@shadow.qacc+b).tolist() for J,b in zip(Js,bias)],'native_vs_raw_LP_qacc_max':float(max(abs(shadow.qacc-np.array(res['predicted_full_qacc'])))),'native_witness':ww}
 out['axis_names']=names;(O/f'state_{t:.3f}.json').write_text(json.dumps(out,indent=2));results.append(out);print('STATE',t,'LP',res.get('status'),'wall',wall,'oldtauerr',out['old_dispatch_tau_error'],flush=True)
(O/'summary.json').write_text(json.dumps({'zero_steps':True,'LP_calls':4,'states':[dict(time_s=x['time_s'],success=x['LP_success'],status=x['LP_status'],wall_s=x['LP_wall_s'],held_FF_age_s=x['held_FF_age_s'],old_dispatch_tau_error=x['old_dispatch_tau_error'],old_native_root_qacc6=x['old_native_root_qacc6'],new_native_root_qacc6=x.get('fresh_same_entry_blend',{}).get('native_root_qacc6'),old_footN=x['old_native_foot_N_R_L'],new_footN=x.get('fresh_same_entry_blend',{}).get('native_foot_N_R_L')) for x in results]},indent=2))
