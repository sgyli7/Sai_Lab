import os,sys,time,datetime,resource,signal
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
from pathlib import Path
import json,gzip,hashlib,importlib.util,traceback
import numpy as np,mujoco as mj
R=Path(__file__).parent;P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04');F=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/full_body_readiness');num=int(sys.argv[1]);gain=1.;program_path=Path(sys.argv[2]);used_steps=sum(json.loads(p.read_text()).get('actual_steps',0) for p in R.glob('attempt_*/summary.json'));assert num<=6 and datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,10,18,15,18,10,tzinfo=datetime.timezone.utc);S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ad=module('immutable_adapter',P/'runtime_adapter.py');prov=module('gait008_provider',R/'gait_target_provider.py');config=json.loads((F/'next_trial_input.json').read_text());env=ad.FrozenPhysics(R/'scene');m,d=env.model,env.data;assert m.neq==3;assert set(m.equality(i).name for i in range(m.neq))=={'beak_distal_pin','left_yaw_input_rotor_belt_relation','right_yaw_input_rotor_belt_relation'}
gait_config=json.loads(program_path.read_text());first=config['times_and_targets'][0].copy();
if gait_config.get('cold_start_from_first_frame'):
 first['q_target18_rad']=gait_config['frames'][0]['q_target18_rad'];first['expected_root_xyz_offline_reference_only']=(np.array(first['expected_root_xyz_offline_reference_only'])+np.array(gait_config['frames'][0]['expected_root_delta_xyz'])).tolist()
root=np.r_[first['expected_root_xyz_offline_reference_only'],1,0,0,0];env.reset(dict(zip(env.names,first['q_target18_rad'])),root_pose=root);assert np.array_equal(env.target,d.qpos[env.qidx]);rod=m.geom('cart_bite_bar').id;cart=m.body('cart_base').id;cargo=m.body('probe_cargo').id;cart0=d.xpos[cart].copy();cargo0=d.xpos[cargo].copy();teacher=prov.AlternatingBackwardTeacher(program_path,d.geom_xpos[rod].copy(),d.qpos[env.qidx].copy());assert used_steps+int(teacher.duration/.005)<=48000;qinitial=d.qpos.copy();initialroot=d.qpos[:3].copy();H={str(p):sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in str(p)};assert len(H)==53;dump(S/'source_before.json',H);head=m.body('head_roll').id;torso=m.body('torso').id;feet=[m.geom(n).id for n in ['right_actual_sole','left_actual_sole']];ground=m.geom('ground').id;tcp_local=np.array([.13750001972253326,0,-.03849996489687674]);initialfeet=d.geom_xpos[feet].copy();rawcontacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact];bad=[c for c in rawcontacts if c['dist_m']<-.0005];assert not bad,bad
meta={'attempt':num,'mode':'LOADED_ALTERNATING_BACKWARD_STEPS','source_model_sha256':sha(P/'robot.xml'),'frozen_provider_sha256':sha(F/'target_provider.py'),'frozen_input_sha256':sha(F/'next_trial_input.json'),'static_bias':True,'order008_sha256':sha(Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/experiments/goose_v0_2/cart_drag_loaded_diagnostic_008/order.json')),'qualification_5mm_unchanged_diagnostic_gross_abort20mm':True,'declared_bias_multiplier':gain,'gait_program_sha256':sha(program_path),'gait_provider_sha256':sha(R/'gait_target_provider.py'),'bias_variant':'gait_provider_declared_tau','bias_proposal_sha256':sha(Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_whole_body_007/l1_bias_proposal.json')),'root_and_feet_fixture':'NONE','equalities':[m.equality(i).name for i in range(m.neq)],'initial_root':root.tolist(),'initial_public_q18':d.qpos[env.qidx].tolist(),'initial_target18':env.target.tolist(),'initial_contacts':rawcontacts,'hold_time_window_pre_s':[teacher.warmup+teacher.walk_time,teacher.duration],'gates':{'root_back_hold_min_m':.015,'hold_duration_s_min':2.,'foot_material_slip_limit_m':.005,'tilt_max_deg':25.,'dual_support_each_normal_N_min':2.,'settling_s':1.,'numerical_hold_max_closure_or_contact_pen_m':.0005},'jointtarget_only':True,'stance_epoch_guard':'reset only actual N<=2N+soleclearance>1mm for50ms followed by actual landing; no planned-phase reset','hardstop_latency':'terminal prestate checked before next5ms integration; terminal_abort_prestate.json preserves trigger; no remaining50Hz substeps','TCP_error_policy':'actual>5mm for200ms explicitly measured diagnostic; no silent success'};dump(S/'declaration.json',meta);np.savez(S/'cold_state.npz',qpos=d.qpos,qvel=d.qvel);mj.mj_saveModel(m,str(S/'actual_scene.mjb'));print('EVENT admitted',num,flush=True)
class PhysicalAbort(Exception):pass
source_step=mj.mj_step;rows=[];commands=[];qtraj=[d.qpos.copy()];vtraj=[d.qvel.copy()];ts=[0.];slip=np.zeros(2);failure=None;qualification_failure=None;threads=1;currentmeta={};stance_active=[True,True];stance_epochs=[0,0];stance_failures=[];actual_air_streak=np.zeros(2);actual_air_seen=[False,False];headerr_streak=0.;supportloss=0.;lasttarget=env.target.copy();log=gzip.open(S/'substeps.jsonl.gz','wt',compresslevel=2)
def logged(model,data,nstep=1):
 global failure,qualification_failure,threads,headerr_streak,supportloss,lasttarget
 assert nstep==1 and not d.qfrc_applied.any() and not d.xfrc_applied.any();preq=d.qpos.copy();prev=d.qvel.copy();pret=float(d.time);mj.mj_forward(m,d);requested=env.kp*(env.target-d.qpos[env.qidx])-env.kd*d.qvel[env.vidx];expected=env.limit_torque(requested,d.qvel[env.drive_vidx]);assert np.max(abs(d.ctrl-expected))<1e-9;assert np.max(abs(env.target-lasttarget)-env.slew_speed*.005)<1e-8;lasttarget=env.target.copy();jawN=np.zeros(2);rodF=np.zeros(3);nonrodF=np.zeros(3);mouth_nonrod=[];normal=np.zeros(2);footforce=np.zeros((2,3));footcop=np.zeros((2,3));speedN=np.zeros(2);cs=[];maxpen=0.
 for ci,c in enumerate(d.contact):
  f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);a,b=int(c.geom1),int(c.geom2);F=c.frame.reshape(3,3).T@f[:3];rec={'geom1':m.geom(a).name,'geom2':m.geom(b).name,'pos_world':c.pos.tolist(),'frame_rows_world':c.frame.tolist(),'force_contact6':f.tolist(),'force_on_geom2_world':F.tolist(),'distance_m':float(c.dist),'friction5':c.friction.tolist()};maxpen=max(maxpen,-float(c.dist))
  if ground in (a,b):
   other=b if a==ground else a
   if other in feet:
    k=feet.index(other);Ffoot=F if b==other else -F;N=max(0.,Ffoot[2]);normal[k]+=N;footforce[k]+=Ffoot;footcop[k]+=N*c.pos;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[other]));v=jp@d.qvel;speedN[k]+=N*np.linalg.norm(v[:2]);rec['stance_material_point_velocity_world']=v.tolist()
  if rod in (a,b) and ('upper_bill_envelope' in (m.geom(a).name,m.geom(b).name) or 'lower_bill_envelope' in (m.geom(a).name,m.geom(b).name)):
   other=b if a==rod else a;k=0 if m.geom(other).name=='upper_bill_envelope' else 1;jawN[k]+=max(f[0],0);rodF+=(1 if b==rod else -1)*F
  namesab=(m.geom(a).name,m.geom(b).name);robot_a=0<int(m.geom_bodyid[a])<cart;robot_b=0<int(m.geom_bodyid[b])<cart;object_a=namesab[0].startswith('cart_') or namesab[0]=='probe_cargo_box';object_b=namesab[1].startswith('cart_') or namesab[1]=='probe_cargo_box';allowed=(rod in (a,b)) and any(n in namesab for n in ['upper_bill_envelope','lower_bill_envelope'])
  if ((robot_a and object_b) or (robot_b and object_a)) and not allowed and f[0]>.01:
   Fcart=F if object_b else -F;nonrodF+=Fcart;mouth_nonrod.append({'pair':namesab,'normal_N':float(f[0]),'force_on_cart_world_N':Fcart.tolist(),'all_robot_shell_guard':True})
  cs.append(rec)
 # True stance epoch changes require measured contact loss, not merely scheduled swing.
 for k in range(2):
  side='right' if k==0 else 'left';intended=side in currentmeta.get('stance',['left','right'])
  g=feet[k];mesh=int(m.geom_dataid[g]);vv=m.mesh_vert[int(m.mesh_vertadr[mesh]):int(m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh])];ww=vv@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];actual_clearance=float(ww[:,2].min())
  actual_air_streak[k]=actual_air_streak[k]+.005 if normal[k]<=2. and actual_clearance>.001 else 0.
  if actual_air_streak[k]>=.05:actual_air_seen[k]=True
  if normal[k]>2.:
   if actual_air_seen[k]:slip[k]=0.;stance_epochs[k]+=1;actual_air_seen[k]=False
   slip[k]+=.005*speedN[k]/normal[k] # includes planned swing that is still physically dragging
  stance_active[k]=intended
  if normal[k]>0:footcop[k]/=normal[k]
 eq=[]
 for i in range(m.neq):
  ids=np.where((d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id[:d.nefc]==i))[0];eq.append({'name':m.equality(i).name,'residual':d.efc_pos[ids].tolist(),'force':d.efc_force[ids].tolist()})
 Rtorso=d.xmat[torso].reshape(3,3);tilt=float(np.degrees(np.arccos(np.clip(Rtorso[2,2],-1,1))));tcp=d.xpos[head]+d.xmat[head].reshape(3,3)@tcp_local;power=float(np.maximum(d.actuator_force*d.qvel[env.drive_vidx],0).sum());assert power<=350+1e-8;rootback=float(initialroot[0]-d.qpos[0]);rootdown=float(initialroot[2]-d.qpos[2]);actualtcperr=float(np.linalg.norm(tcp-np.array(currentmeta['tcp_target_world'])));both=bool(min(normal)>2);intended_support_ok=all(normal[k]>2 for k,side in enumerate(['right','left']) if side in currentmeta.get('stance',['left','right']))
 wheel_reactions=[]
 for fi in np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF))[0]:
  dof=int(d.efc_id[fi]);jn=int(m.dof_jntid[dof]);name=m.joint(jn).name
  if name.startswith('cart_wheel_spin'):wheel_reactions.append({'joint':name,'torque_Nm':float(d.efc_force[fi]),'speed_rad_s':float(d.qvel[dof])})
 footbottom=[]
 for g in feet:
  mesh=int(m.geom_dataid[g]);vv=m.mesh_vert[int(m.mesh_vertadr[mesh]):int(m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh])];ww=vv@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];footbottom.append(float(ww[:,2].min()))
 row={'wheel_bearing_actual_reactions':wheel_reactions,'foot_geometry_min_z_m':footbottom,'pre_bodies':{name:{'xpos':d.xpos[m.body(name).id].tolist(),'xmat':d.xmat[m.body(name).id].tolist()} for name in ['head_roll','beak_hinge','cart_base','cart_roll','probe_cargo','torso']},'jaw_rod_normal_N':jawN.tolist(),'tow_on_rod_world_N':rodF.tolist(),'mouth_nonrod_contacts':mouth_nonrod,'nonrod_force_on_cart_world_N':nonrodF.tolist(),'cart_back_m':float(cart0[0]-d.xpos[cart,0]),'cargo_back_m':float(cargo0[0]-d.xpos[cargo,0]),'cart_pose':np.r_[d.xpos[cart],d.xquat[cart]].tolist(),'cargo_pose':np.r_[d.xpos[cargo],d.xquat[cargo]].tolist(),'rod_world':d.geom_xpos[rod].tolist(),'pre_time_s':pret,'pre_qpos':preq.tolist(),'pre_qvel':prev.tolist(),'root_pose':d.qpos[:7].tolist(),'root_velocity6':d.qvel[:6].tolist(),'root_back_m':rootback,'root_down_m':rootdown,'torso_pose':np.r_[d.xpos[torso],d.xquat[torso]].tolist(),'COM_world':d.subtree_com[torso].tolist(),'torso_tilt_deg':tilt,'actual_TCP_world':tcp.tolist(),'actual_TCP_error_m':actualtcperr,'command_metadata':currentmeta,'foot_normal_N':normal.tolist(),'foot_force_world_N':footforce.tolist(),'foot_CoP_world_m':footcop.tolist(),'foot_material_slip_integral_m':slip.tolist(),'foot_geom_center_excursion_m':np.linalg.norm((d.geom_xpos[feet]-initialfeet)[:,:2],axis=1).tolist(),'stance_intent':currentmeta.get('stance',['left','right']),'stance_epoch_ids':stance_epochs.copy(),'intended_support_ok':intended_support_ok,'dual_support':both,'contacts':cs,'max_penetration_m':maxpen,'equalities':eq,'requested_PD_tau18':requested.tolist(),'actual_torque18':d.actuator_force.tolist(),'actual_drive_speed18':d.qvel[env.drive_vidx].tolist(),'positive_mechanical_power_W':power,'actual_slew_target18':env.target.tolist(),'warnings':[int(w.number) for w in d.warning]}
 if pret>1. and abs(nonrodF[0])>max(2.,abs(rodF[0])):failure='nonrod_contact_is_main_X_transport_force'
 if tilt>25:failure='tilt_over25deg'
 if np.max(slip)>.005 and qualification_failure is None:qualification_failure={'reason':'stance_material_slip_over5mm','first_time_s':pret,'foot_material_slip_m':slip.tolist(),'root_back_m':rootback}
 if actualtcperr>.005:headerr_streak+=.005
 else:headerr_streak=0.
 if np.max(slip)>.020:failure='diagnostic_gross_material_slip_over20mm'
 if pret>=1 and not intended_support_ok:supportloss+=.005
 else:supportloss=0.
 if supportloss>.10:failure='dual_support_lost_over100ms_after_settle'
 if any(row['warnings']) or not np.isfinite(d.qpos).all():failure='nonfinite_or_mujoco_warning'
 if failure:
  dump(S/'terminal_abort_prestate.json',dict(row,abort_reason=failure,physics_step_executed=False));raise PhysicalAbort(failure)
 source_step(m,d,nstep=1);row.update(post_time_s=float(d.time),post_qpos=d.qpos.tolist(),post_qvel=d.qvel.tolist());log.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row);qtraj.append(d.qpos.copy());vtraj.append(d.qvel.copy());ts.append(float(d.time))
 if len(rows)%200==0:
  threads=max(threads,int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))));print('EVENT',num,round(d.time,2),'backmm',round(rootback*1000,2),'downmm',round(rootdown*1000,2),'Fn',normal.round(1).tolist(),'slipmm',(slip*1000).round(2).tolist(),'tilt',round(tilt,2),'cartmm',round(row['cart_back_m']*1000,2),'jawN',jawN.round(1).tolist(),'phase',currentmeta.get('phase'),flush=True)
mj.mj_step=logged;error=None
try:
 for tick in range(int(round(teacher.duration/.02))):
  assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,10,18,15,24,0,tzinfo=datetime.timezone.utc)
  action,currentmeta=teacher.action(env,float(d.time),d.geom_xpos[rod].copy())
  if gain!=1.:
   desired=np.array(currentmeta['requested_joint_targets18'])+(gain-1.)*tau/env.kp;action=(desired-env.neutral)/env.scale;currentmeta['requested_joint_targets18']=desired.tolist()
  currentmeta['bias_multiplier']=gain;commands.append({'time_s':float(d.time),'action18':action.tolist(),'metadata':currentmeta.copy()});env.step(action)
  if failure:break
except PhysicalAbort:pass
except BaseException as ex:error={'error':repr(ex),'traceback':traceback.format_exc()}
finally:
 mj.mj_step=source_step;log.close();np.savez_compressed(S/'trajectory.npz',qpos=np.array(qtraj),qvel=np.array(vtraj),time=np.array(ts));dump(S/'commands50hz.json',commands);hold=[r for r in rows if r['command_metadata'].get('phase')=='stop_hold'];hclosure=max([abs(x) for r in hold for e in r['equalities'] for x in e['residual']],default=0);hpen=max([r['max_penetration_m'] for r in hold],default=0);gate=bool(len(hold)>=400 and min(r['root_back_m'] for r in hold)>=.015 and all(r['dual_support'] for r in hold) and max(max(r['foot_material_slip_integral_m']) for r in hold)<=.005 and hclosure<=.0005 and hpen<=.0005 and failure is None and error is None and qualification_failure is None);out={'attempt':num,'mode':'LOADED_ALTERNATING_BACKWARD_STEPS','actual_steps':len(rows),'sim_duration_s':float(d.time),'failure':failure,'error':error,'final_cart_back_m':rows[-1]['cart_back_m'] if rows else None,'final_cargo_back_m':rows[-1]['cargo_back_m'] if rows else None,'max_cart_back_m':max([r['cart_back_m'] for r in rows],default=0),'jaw_capture_ever':any(min(r['jaw_rod_normal_N'])>.01 for r in rows),'final_jaw_rod_normal_N':rows[-1]['jaw_rod_normal_N'] if rows else None,'TCP_5mm_for200ms_recorded_diagnostic_only':headerr_streak>=.2,'gait_cycles_requested':teacher.cycles,'planned_back_m':teacher.cycles*teacher.cycle_back,'unloaded_gate_pass':False,'irreversible_qualification_failure':qualification_failure,'diagnostic_only_not_loaded_admission':True,'hold_steps':len(hold),'hold_min_root_back_m':min([r['root_back_m'] for r in hold],default=None),'hold_root_back_range_m':[min([r['root_back_m'] for r in hold],default=0),max([r['root_back_m'] for r in hold],default=0)],'hold_max_closure_m_or_rad':hclosure,'hold_max_contact_penetration_m':hpen,'final_root_back_m':rows[-1]['root_back_m'] if rows else None,'max_root_back_m':max([r['root_back_m'] for r in rows],default=0),'foot_material_slip_integral_m':slip.tolist(),'max_tilt_deg':max([r['torso_tilt_deg'] for r in rows],default=0),'max_positive_power_W':max([r['positive_mechanical_power_W'] for r in rows],default=0),'max_observed_threads':threads,'maxRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source53_unchanged':all(sha(Path(p))==h for p,h in H.items()),'GPU':0,'training':0};dump(S/'summary.json',out);print('STATUS',json.dumps(out),flush=True)
