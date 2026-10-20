"""Full original guard/serialization path; hard-block physical source_step.
Instrumentation stays in memory, not the candidate runner. Cold and archived
hold are separate scratch fixtures, never initialization for actual restart.
"""
from pathlib import Path
import sys,json,gzip,hashlib,numpy as np,mujoco as mj,datetime
sys.dont_write_bytecode=True
R=Path(__file__).parent;sys.path.insert(0,str(R));source=(R/'run_support.py').read_text();prefix=source.split('mj.mj_step=logged;error=None;bridge_completed=False')[0];A=R.parent/'physical08_active_capture_10_online1000/attempt_001';ctx=json.loads((A/'capture_handoff_context.json').read_text());sm=json.loads((A/'summary.json').read_text());last=json.loads((A/'commands50hz.json').read_text())[-1]['metadata'];oldstep=mj.mj_step;results=[]
class ReachedBlockedPhysicalStep(Exception):pass
for name in ['motion_0','motion_0p9','motion_STOP']:
 calls=[0]
 def blocked(*args,**kwargs):calls[0]+=1;raise ReachedBlockedPhysicalStep('physical integration deliberately blocked after complete guard+JSON')
 mj.mj_step=blocked
 # Do not execute the actualauthorization path: this in-memory prelude has
 # explicit zero-step instrumentation and only creates separate dryrun outputs.
 pre=prefix.replace("authorization_path=Path(sys.argv[3]) if len(sys.argv)>3 else R/'ACTUAL_AUTHORIZATION_NOT_GRANTED.json';assert authorization_path.is_file(),'Owner actual authorization required';authorization=json.loads(authorization_path.read_text());assert authorization.get('actual_authorized') is True and authorization.get('attempt_limit')==1;deadline=datetime.datetime.fromisoformat(authorization['deadline_UTC']);", "authorization={'frozen_input_manifest_sha256':'ZERO_STEP_INSTRUMENTATION'};deadline=datetime.datetime(2026,10,20,1,0,tzinfo=datetime.timezone.utc);")
 pre=pre.replace("S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False)",f"S=R/'dryrun13driver04_{name}';S.mkdir(exist_ok=False)")
 pre=pre.replace("assert sha(R/'frozen_admission_manifest.json')==authorization['frozen_input_manifest_sha256'];",'')
 sys.argv=[str(R/'run_support.py'),'1',str(R/'capture_prefix_source/baseline_capture/causal_program.json')];ns={'__file__':str(R/'run_support.py'),'__name__':'zero_step_guard_fixture'};exec(compile(pre,str(R/'run_support.py'),'exec'),ns);e=ns['env'];d=ns['d'];teacher=ns['teacher'];_,meta=teacher.action(e,0.,d.geom_xpos[ns['rod']].copy())
 if name!='cold_open':
  d.qpos[:]=ctx['actual_qpos'];d.qvel[:]=ctx['actual_qvel'];d.time=ctx['clock_s'];e.original.target[:]=ctx['original_env_target'];e.command_details=ctx['held_motor_command'];ns['slip'][:]=sm['foot_material_slip_integral_m'];ns['slip_total_since_cold'][:]=sm['foot_material_slip_integral_m'];meta=last;cmd=dict(q_ref18_rad=ctx['original_env_target'],qd_ref18_rad_s=ctx['held_motor_command']['qd_ref18_rad_s'],tau_ff18_Nm=ctx['held_motor_command']['tau_ff18_Nm']);mj.mj_forward(e.model,d)
 else:cmd=meta['motor_command']
 if name.startswith('motion_'):
  elapsed={'motion_0':0.,'motion_0p9':.9,'motion_STOP':4.90625}[name];rb=json.loads((R/'readbacks'/('state_00.json' if elapsed==0 else 'state_03.json')).read_text());cc=dict(ctx,observed_root_world=d.qpos[:3].tolist(),observed_cart_world=d.xpos[ns['cart']].tolist(),observed_cargo_world=d.xpos[ns['cargo']].tolist());meta=dict(meta,phase='final_DS_hold_no_return' if name=='motion_STOP' else 'coherent_planned_leg_native_soft',motion_start_time_s=float(d.time)-elapsed,capture_context=cc,motion_reference_heading=[1.,0.,0.],trajectory_packet=rb['packet'],actual_dispatched_native_witness=rb['actual_blended_native_witness'],stance=['left','right'],readback_only_metadata_fixture=True);meta['COM_reference_world']=d.subtree_com[e.model.body('torso').id].tolist()


 from native_coherent_motion import NativeCoherentMotion
 from runner_segment_dispatch import dispatch_segment_20ms
 actualctx=json.loads((R.parent/'coherent_native_soft_12_dependencyfix_online1000/attempt_002/actual_capture_to_motion_context.json').read_text());e.reference_output_dir=ns['S']/'reference_offline';motion=NativeCoherentMotion(e,actualctx,sm['foot_material_slip_integral_m']);elapsed={'motion_0':0.,'motion_0p9':.9,'motion_STOP':4.90625}[name]
 if name=='motion_STOP':
  d.qpos[:]=actualctx['actual_qpos'];d.qvel[:]=0.;shift=motion.alignment.R.apply([-0.525,0.,0.])
  for j in range(e.model.njnt):
   if int(e.model.jnt_type[j])==int(mj.mjtJoint.mjJNT_FREE):d.qpos[e.model.jnt_qposadr[j]:e.model.jnt_qposadr[j]+3]+=shift
  e.original.target[:]=actualctx['original_env_target'];e.command_details=actualctx['held_motor_command'];d.time=actualctx['clock_s']+elapsed;mj.mj_forward(e.model,d)
 if name=='motion_0p9':
  zzprev=None
  with gzip.open(R.parent/'coherent_native_soft_12_dependencyfix_online1000/attempt_002/substeps.jsonl.gz','rt') as zf:
   for ln in zf:
    zz=json.loads(ln)
    if abs(zz['pre_time_s']-6.7)<1e-8:
     d.qpos[:]=zz['pre_qpos'];d.qvel[:]=zz['pre_qvel'];d.ctrl[:]=zz['actual_torque18'];d.time=actualctx['clock_s']+elapsed;e.original.target[:]=zzprev['actual_slew_target18'];e.command_details=zzprev['motor_profile_details'];mj.mj_forward(e.model,d);break
    zzprev=zz
 _,meta=motion.prepare(elapsed);assert meta.get('teacher_abort_reason') is None,meta.get('teacher_abort_reason');meta.update(capture_context=actualctx,motion_start_time_s=actualctx['clock_s'],motion_reference_heading=[1.,0.,0.]);cmd=meta['segment_motor_command'];motionstart=float(d.time)-elapsed
 ns['lasttarget']=e.target.copy();ns['shadow_target']=e.target.copy();ns['currentmeta']=meta;q=d.qpos.copy();v=d.qvel.copy();t=float(d.time);mj.mj_step=ns['logged'];reached=False;error=None
 try:dispatch_segment_20ms(e,cmd['packet'],cmd['tau_ff18_Nm'],motionstart,ns['logged'])
 except ReachedBlockedPhysicalStep:reached=True
 except BaseException as ex:
  import traceback;traceback.print_exc();error=repr(ex)
 finally:ns['log'].close();mj.mj_step=blocked
 out=dict(case=name,complete_guard_and_json_reached_physical_step_stub=reached,stub_calls=calls[0],actual_physical_steps=0,actual_time_unchanged=float(d.time)==t,qpos_unchanged=np.array_equal(q,d.qpos),qvel_unchanged=np.array_equal(v,d.qvel),error=error,all_guard_imports_including_centroidal_helper_resolved=reached)
 results.append(out);print(out,flush=True)
 assert reached and calls[0]==1 and float(d.time)==t and np.array_equal(q,d.qpos) and np.array_equal(v,d.qvel),out
mj.mj_step=oldstep
out=dict(physics_steps=0,cases=results,source_runner_sha256=hashlib.sha256((R/'run_support.py').read_bytes()).hexdigest(),logging_helper_sha256=hashlib.sha256((R/'actual_centroidal_wrench.py').read_bytes()).hexdigest(),runner_control_guard_bytes_unchanged_except_attempt_index=True,archived_hold_used_only_for_OFFLINE_fixture_NOT_actual_restart=True)
(R/'driver_fullguard_zero13_readback.json').write_text(json.dumps(out,indent=2))
