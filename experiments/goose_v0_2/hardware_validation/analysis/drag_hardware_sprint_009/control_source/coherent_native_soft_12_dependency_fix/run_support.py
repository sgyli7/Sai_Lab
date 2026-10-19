import os,sys,time,datetime,resource,signal
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
from pathlib import Path
import json,gzip,hashlib,importlib.util,traceback
import numpy as np,mujoco as mj
R=Path(__file__).parent;P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261020_08_candidate');F=R;num=int(sys.argv[1]);gain=1.;program_path=Path(sys.argv[2]);authorization_path=Path(sys.argv[3]) if len(sys.argv)>3 else R/'ACTUAL_AUTHORIZATION_NOT_GRANTED.json';assert authorization_path.is_file(),'Owner actual authorization required';authorization=json.loads(authorization_path.read_text());assert authorization.get('actual_authorized') is True and authorization.get('attempt_limit')==1;deadline=datetime.datetime.fromisoformat(authorization['deadline_UTC']);used_steps=sum(json.loads(p.read_text()).get('actual_steps',0) for p in R.glob('attempt_*/summary.json'));assert num==2 and datetime.datetime.now(datetime.timezone.utc)<deadline;S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
frozen=json.loads((R/'frozen_admission_manifest.json').read_text());assert sha(R/'frozen_admission_manifest.json')==authorization['frozen_input_manifest_sha256'];assert all(sha(R/rel)==row['sha256'] for rel,row in frozen['files'].items())
ad=module('immutable_adapter',P/'runtime_adapter.py');prov=module('authorized09_provider',R/'dynamic_support_teacher.py');mit=module('separate_MIT009_profile',R/'teacher_mit_profile.py');env=mit.TeacherMITDrive(R/'scene');env.torque_mask=np.arange(18)>=6;m,d=env.model,env.data;env.reference_output_dir=S/'generated_reference';assert m.neq==3 and m.ngeom==51;assert set(m.equality(i).name for i in range(m.neq))=={'beak_distal_pin','left_yaw_input_rotor_belt_relation','right_yaw_input_rotor_belt_relation'}
cold=np.load(R/'capture_prefix_source/cold_state.npz');root=cold['qpos'][:7];env.reset(dict(zip(env.names,cold['qpos'][env.qidx])),root_pose=root);assert max(abs(d.qpos-cold['qpos']))<1e-12 and max(abs(d.qvel-cold['qvel']))<1e-12;assert np.array_equal(env.target,d.qpos[env.qidx]);
rod=m.geom('cart_bite_bar').id;cart=m.body('cart_base').id;cargo=m.body('probe_cargo').id;cart0=d.xpos[cart].copy();cargo0=d.xpos[cargo].copy();teacher=prov.AlternatingBackwardTeacher(program_path,d.geom_xpos[rod].copy(),d.qpos[env.qidx].copy());assert used_steps+int(teacher.duration/.001)<=15000;qinitial=d.qpos.copy();initialroot=d.qpos[:3].copy();H={str(p):sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in str(p)};source_manifest=json.loads((P/'manifest.json').read_text());assert len(source_manifest['files'])==79;assert len(H)==80;assert all(sha(P/x['path'])==x['sha256'] for x in source_manifest['files']);dump(S/'source_before.json',H);head=m.body('head_roll').id;torso=m.body('torso').id;feet=[m.geom(n).id for n in ['right_actual_sole','left_actual_sole']];ground=m.geom('ground').id;foot_ground_names=json.loads((R/'foot_ground_names.json').read_text());foot_ground_geoms={m.geom(n).id for n in foot_ground_names};tcp_local=np.array([.13750001972253326,0,-.03849996489687674]);initialfeet=d.geom_xpos[feet].copy();rawcontacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact];bad=[c for c in rawcontacts if c['dist_m']<-.0005];assert not bad,bad
meta={'attempt':num,'causal_cell':teacher.cell,'runner_sha256':sha(Path(__file__)),'diagnostic_pregrip_not_active_capture':False,'mode':'LOADED_ONLINE_TEACHER_PHYSICS1000_DRIVE200_COMMAND50','source_model_sha256':sha(P/'robot.xml'),'frozen_provider_sha256':sha(R/'dynamic_support_teacher.py'),'frozen_input_sha256':sha(R/'capture_prefix_source/cold_state.npz'),'static_bias':True,'order009_sha256':sha(Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/experiments/goose_v0_2/drag_hardware_sprint_009/order.json')),'qualification_5mm_unchanged_diagnostic_gross_abort20mm':True,'declared_bias_multiplier':gain,'gait_program_sha256':sha(program_path),'gait_provider_sha256':sha(R/'dynamic_support_teacher.py'),'bias_variant':'physical08_native_soft12_coherentlegs_segmentdrive_samecapture','static_bias_program_sha256':sha(R/'capture_prefix_source/baseline_capture/causal_program.json'),'root_and_feet_fixture':'NONE','equalities':[m.equality(i).name for i in range(m.neq)],'initial_root':root.tolist(),'initial_public_q18':d.qpos[env.qidx].tolist(),'initial_target18':env.target.tolist(),'initial_contacts':rawcontacts,'hold_time_window_pre_s':[teacher.warmup+teacher.walk_time,teacher.duration],'gates':{'root_back_hold_min_m':.015,'hold_duration_s_min':2.,'foot_material_slip_limit_m':.005,'tilt_max_deg':25.,'dual_support_each_normal_N_min':2.,'settling_s':1.,'numerical_hold_max_closure_or_contact_pen_m':.0005},'jointtarget_only':False,'original_motor_interfaces_q_qd_FF_only':True,'stance_epoch_guard':'reset only actual N<=2N+soleclearance>1mm for50ms followed by actual landing; no planned-phase reset','hardstop_latency':'terminal prestate checked before next1ms integration; terminal_abort_prestate.json preserves trigger; no remaining50Hz substeps','TCP_error_policy':'actual>5mm for200ms explicitly measured diagnostic; no silent success'};dump(S/'declaration.json',meta);np.savez(S/'cold_state.npz',qpos=d.qpos,qvel=d.qvel);mj.mj_saveModel(m,str(S/'actual_scene.mjb'));print('EVENT admitted',num,flush=True)
class PhysicalAbort(Exception):pass
shadow_target=env.target.copy();shadow_errors=[];height_error_streak=0.;grip_loss_streak=0.;source_step=mj.mj_step;rows=[];commands=[];qtraj=[d.qpos.copy()];vtraj=[d.qvel.copy()];ts=[0.];slip=np.zeros(2);slip_total_since_cold=np.zeros(2);failure=None;qualification_failure=None;threads=1;currentmeta={};stance_active=[True,True];stance_epochs=[0,0];stance_failures=[];actual_air_streak=np.zeros(2);actual_air_seen=[False,False];headerr_streak=0.;supportloss=0.;lasttarget=env.target.copy();log=gzip.open(S/'substeps.jsonl.gz','wt',compresslevel=2)
def logged_one(model,data,nstep=1,drivefirst=True):
 global failure,qualification_failure,threads,headerr_streak,supportloss,lasttarget,shadow_target,height_error_streak,grip_loss_streak
 assert nstep==1 and not d.qfrc_applied.any() and not d.xfrc_applied.any();preq=d.qpos.copy();prev=d.qvel.copy();pret=float(d.time);mj.mj_forward(m,d);requested=env.kp*(env.target-d.qpos[env.qidx])-env.kd*d.qvel[env.vidx];details=env.command_details;requested[env.torque_mask]=0. if details.get('profile')=='teacher_simulation_SI_torque_v3_12leg_configuration' else requested[env.torque_mask];extra=np.array(details.get('tau_ff18_Nm',[0.]*18))+env.kd*np.array(details.get('qd_ref18_rad_s',[0.]*18));expected=env.limit_torque(requested+extra,d.qvel[env.drive_vidx]);assert (not drivefirst) or np.max(abs(d.ctrl-expected))<1e-9;assert np.max(abs(d.ctrl-np.array(details.get('limited_total_tau18_Nm',d.ctrl))))<1e-9;
 if 'q_only_reference_request18' in currentmeta:
  shadow_target+=np.clip(np.array(currentmeta['q_only_reference_request18'])-shadow_target,-env.slew_speed*.005,env.slew_speed*.005)
  shadow_tau=env.limit_torque(env.kp*(shadow_target-d.qpos[env.qidx])-env.kd*d.qvel[env.vidx],d.qvel[env.drive_vidx]);shadow_error=float(np.max(abs(d.ctrl-shadow_tau)));shadow_errors.append(shadow_error);assert shadow_error<1e-6,('bumpless_total_tau_mismatch',shadow_error)
 else:shadow_target=env.target.copy();shadow_tau=d.ctrl.copy();shadow_error=0.
 assert np.max(abs(env.target-lasttarget)-env.slew_speed*.005)<1e-8;lasttarget=env.target.copy();jawN=np.zeros(2);rodF=np.zeros(3);nonrodF=np.zeros(3);mouth_nonrod=[];normal=np.zeros(2);footforce=np.zeros((2,3));footcop=np.zeros((2,3));speedN=np.zeros(2);cs=[];maxpen=0.
 for ci,c in enumerate(d.contact):
  f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);a,b=int(c.geom1),int(c.geom2);F=c.frame.reshape(3,3).T@f[:3];rec={'original_contact_index':ci,'native_efc_address':int(c.efc_address),'active_native_constraint':bool(c.efc_address>=0),'geom1_id':a,'geom2_id':b,'body1':m.body(int(m.geom_bodyid[a])).name,'body2':m.body(int(m.geom_bodyid[b])).name,'world_torque_on_geom2_at_contact_Nm':(c.frame.reshape(3,3).T@f[3:]).tolist(),'geom1':m.geom(a).name,'geom2':m.geom(b).name,'pos_world':c.pos.tolist(),'frame_rows_world':c.frame.tolist(),'force_contact6':f.tolist(),'force_on_geom2_world':F.tolist(),'distance_m':float(c.dist),'friction5':c.friction.tolist()};maxpen=max(maxpen,-float(c.dist))
  if any(x in (a,b) for x in foot_ground_geoms):
   other=b if a in foot_ground_geoms else a
   if other in feet:
    k=feet.index(other);Ffoot=F if b==other else -F;N=max(0.,Ffoot[2]);normal[k]+=N;footforce[k]+=Ffoot;footcop[k]+=N*c.pos;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[other]));v=jp@d.qvel;speedN[k]+=N*np.linalg.norm(v[:2]);rec['stance_material_point_velocity_world']=v.tolist()
  if rod in (a,b) and ('upper_bill_envelope' in (m.geom(a).name,m.geom(b).name) or 'lower_bill_envelope' in (m.geom(a).name,m.geom(b).name)):
   other=b if a==rod else a;k=0 if m.geom(other).name=='upper_bill_envelope' else 1;jawN[k]+=max(f[0],0);rodF+=(1 if b==rod else -1)*F
  namesab=(m.geom(a).name,m.geom(b).name);robot_a=0<int(m.geom_bodyid[a])<cart;robot_b=0<int(m.geom_bodyid[b])<cart;object_a=namesab[0].startswith('cart_') or namesab[0]=='probe_cargo_box';object_b=namesab[1].startswith('cart_') or namesab[1]=='probe_cargo_box';allowed=(rod in (a,b)) and any(n in namesab for n in ['upper_bill_envelope','lower_bill_envelope'])
  if ((robot_a and object_b) or (robot_b and object_a)) and not allowed and f[0]>.01:
   Fcart=F if object_b else -F;nonrodF+=Fcart;mouth_nonrod.append({'pair':namesab,'normal_N':float(f[0]),'force_on_cart_world_N':Fcart.tolist(),'all_robot_shell_guard':True})
  if robot_a and robot_b and f[0]>.01 and not getattr(env,'first_robot_self_contact_reported',False):
   env.first_robot_self_contact_reported=True;print('EVENT first_robot_self_contact_physical08',pret,namesab,float(c.dist),float(f[0]),flush=True)
  cs.append(rec)
 # True stance epoch changes require measured contact loss, not merely scheduled swing.
 for k in range(2):
  side='right' if k==0 else 'left';intended=side in currentmeta.get('stance',['left','right'])
  g=feet[k];mesh=int(m.geom_dataid[g]);vv=m.mesh_vert[int(m.mesh_vertadr[mesh]):int(m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh])];ww=vv@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];actual_clearance=float(ww[:,2].min())
  actual_air_streak[k]=actual_air_streak[k]+.001 if normal[k]<=2. and actual_clearance>.001 else 0.
  if actual_air_streak[k]>=.05:actual_air_seen[k]=True
  if normal[k]>2.:
   if actual_air_seen[k]:
    event={'side':side,'time_s':pret,'actual_air50ms_previously_verified':True,'actual_back_from_cold_m':float(initialfeet[k,0]-d.geom_xpos[g,0]),'normal_N':float(normal[k]),'sole_min_z_m':actual_clearance};env.actual_foot_landing_events=getattr(env,'actual_foot_landing_events',[])+[event];slip[k]=0.;stance_epochs[k]+=1;actual_air_seen[k]=False
   increment=.001*speedN[k]/normal[k];slip[k]+=increment;slip_total_since_cold[k]+=increment # planned swing still dragging counts; lifetime never resets
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
 projected=np.array(currentmeta.get('force_prediction',{}).get('total_tau18_Nm',d.ctrl));tau_delta=d.ctrl-projected
 row={'physics_dt_s':.001,'drive_Hz':200,'command_Hz':50,'drive200Hz_updated':drivefirst,'command50Hz_index':len(commands)-1,'held_total_tau18':d.ctrl.tolist(),'hypothetical_PD_recomputed_at_fine_state_not_applied':expected.tolist(),'projected_total_tau18_Nm':projected.tolist(),'actual_minus_projected_total_tau18_Nm':tau_delta.tolist(),'motor_profile_details':details,'same_state_original_q_only_shadow_tau18_Nm':shadow_tau.tolist(),'same_state_total_tau_shadow_error_Nm':shadow_error,'wheel_bearing_actual_reactions':wheel_reactions,'foot_geometry_min_z_m':footbottom,'pre_bodies':{name:{'xpos':d.xpos[m.body(name).id].tolist(),'xmat':d.xmat[m.body(name).id].tolist()} for name in ['head_roll','beak_hinge','cart_base','cart_roll','probe_cargo','torso']},'jaw_rod_normal_N':jawN.tolist(),'tow_on_rod_world_N':rodF.tolist(),'mouth_nonrod_contacts':mouth_nonrod,'nonrod_force_on_cart_world_N':nonrodF.tolist(),'cart_back_m':float(cart0[0]-d.xpos[cart,0]),'cargo_back_m':float(cargo0[0]-d.xpos[cargo,0]),'cart_pose':np.r_[d.xpos[cart],d.xquat[cart]].tolist(),'cargo_pose':np.r_[d.xpos[cargo],d.xquat[cargo]].tolist(),'rod_world':d.geom_xpos[rod].tolist(),'pre_time_s':pret,'pre_qpos':preq.tolist(),'pre_qvel':prev.tolist(),'root_pose':d.qpos[:7].tolist(),'root_velocity6':d.qvel[:6].tolist(),'root_back_m':rootback,'root_down_m':rootdown,'torso_pose':np.r_[d.xpos[torso],d.xquat[torso]].tolist(),'COM_world':d.subtree_com[torso].tolist(),'torso_tilt_deg':tilt,'actual_TCP_world':tcp.tolist(),'actual_TCP_error_m':actualtcperr,'command_metadata':currentmeta,'foot_normal_N':normal.tolist(),'foot_force_world_N':footforce.tolist(),'foot_CoP_world_m':footcop.tolist(),'foot_material_slip_integral_m':slip.tolist(),'foot_geom_center_excursion_m':np.linalg.norm((d.geom_xpos[feet]-initialfeet)[:,:2],axis=1).tolist(),'stance_intent':currentmeta.get('stance',['left','right']),'stance_epoch_ids':stance_epochs.copy(),'intended_support_ok':intended_support_ok,'dual_support':both,'contacts':cs,'max_penetration_m':maxpen,'equalities':eq,'requested_PD_tau18':requested.tolist(),'actual_torque18':d.actuator_force.tolist(),'actual_drive_speed18':d.qvel[env.drive_vidx].tolist(),'positive_mechanical_power_W':power,'actual_slew_target18':env.target.tolist(),'warnings':[int(w.number) for w in d.warning]}
 row['pre_qacc']=d.qacc.tolist();row['actual_air_streak_s']=actual_air_streak.tolist();row['actual_air50ms_seen']=actual_air_seen.copy();row['source_requested_minus_applied18']=(np.array(details.get('combined_requested_tau18_Nm',d.ctrl))-d.actuator_force).tolist()
 row['all_sole_vertices_world_z_m']={};row['foot_wrench6_world_on_sole_about_geomcenter']={};row['mouth_wrench6_world_on_robot_about_COM']=np.zeros(6);row['mouth_wrench6_world_on_rod_about_rodcenter']=np.zeros(6)
 for k,g in enumerate(feet):
  side=['right','left'][k];mesh=m.geom_dataid[g];vv=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];row['all_sole_vertices_world_z_m'][side]=(vv@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].tolist();W=np.zeros(6)
  for c0 in cs:
   if g in (c0['geom1_id'],c0['geom2_id']) and any(gg in (c0['geom1_id'],c0['geom2_id']) for gg in foot_ground_geoms):
    sg=1 if c0['geom2_id']==g else -1;F0=sg*np.array(c0['force_on_geom2_world']);T0=sg*np.array(c0['world_torque_on_geom2_at_contact_Nm']);W[:3]+=F0;W[3:]+=T0+np.cross(np.array(c0['pos_world'])-d.geom_xpos[g],F0)
  row['foot_wrench6_world_on_sole_about_geomcenter'][side]=W.tolist()
 for c0 in cs:
  if rod in (c0['geom1_id'],c0['geom2_id']) and any(n in [c0['geom1'],c0['geom2']] for n in ['upper_bill_envelope','lower_bill_envelope']):
   sg=1 if c0['geom2_id']==rod else -1;Fr=sg*np.array(c0['force_on_geom2_world']);Tr=sg*np.array(c0['world_torque_on_geom2_at_contact_Nm']);pos=np.array(c0['pos_world']);row['mouth_wrench6_world_on_rod_about_rodcenter']+=np.r_[Fr,Tr+np.cross(pos-d.geom_xpos[rod],Fr)];row['mouth_wrench6_world_on_robot_about_COM']+=np.r_[-Fr,-Tr+np.cross(pos-d.subtree_com[torso],-Fr)]
 for key in ['mouth_wrench6_world_on_robot_about_COM','mouth_wrench6_world_on_rod_about_rodcenter']:row[key]=row[key].tolist()
 row['wrench6_convention']='[Fx,Fy,Fz,Mx,My,Mz],world; named作用对象 and reference point; contacts geom2 receives frame.T force/torque'
 row['handle_axes_actual']={}
 for axis in ['yaw','pitch','roll']:
  j=m.joint('cart_'+axis+'_hinge').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];row['handle_axes_actual'][axis]={'q_rad':float(d.qpos[qa]),'qd_rad_s':float(d.qvel[va]),'qdd_rad_s2':float(d.qacc[va]),'spring_Nm':float(-m.jnt_stiffness[j]*(d.qpos[qa]-m.qpos_spring[qa])),'passive_total_Nm':float(d.qfrc_passive[va]),'stiffness_Nm_rad':float(m.jnt_stiffness[j]),'spring_ref_rad':float(m.qpos_spring[qa])}
 Jcom=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,Jcom,torso);row['COM_velocity_world_m_s']=(Jcom@d.qvel).tolist()
 dsok=min(row['foot_normal_N'])>20 and min(jawN)>=5 and abs(row['COM_world'][2]-currentmeta.get('COM_reference_world',row['COM_world'])[2])<=.005 and np.linalg.norm(np.array(row['COM_world'][:2])-np.array(currentmeta.get('COM_reference_world',row['COM_world'])[:2]))<=.005 and tilt<=5 and max(slip)<=.005
 env.actual_DS_gate={'continuous_DS_conditions_s':getattr(env,'actual_DS_gate',{}).get('continuous_DS_conditions_s',0.)+.001 if dsok else 0.,'measured_conditions':bool(dsok)};row['actual_DS_gate']=env.actual_DS_gate.copy()
 if True: # exact same fine PRE-state wrench, not one command-old
  from actual_centroidal_wrench import measure
  row['actual_centroidal_wrench']=measure(m,d);env.last_wrench_command_index=len(commands)-1
 # Actual loaded support manifold, never a geometry/AABB stand-in.
 plant_dist={};plant_tilt={};plant_points={};comref=currentmeta.get('COM_reference_world',row['COM_world']);pamp=.022*(1+comref[2]/9.81*(2*np.pi*1.5)**2)
 for k,side in enumerate(['right','left']):
  points=np.array([c['pos_world'][:2] for c in cs if c['force_contact6'][0]>.01 and any(n in [c['geom1'],c['geom2']] for n in foot_ground_names) and side+'_actual_sole' in [c['geom1'],c['geom2']]])
  point=np.array([comref[0],comref[1]+(pamp if side=='left' else -pamp)]);dist=float('inf')
  if len(points):
   points=np.unique(points,axis=0)
   if len(points)>=3:
    from scipy.spatial import ConvexHull,QhullError
    try:
     hull=ConvexHull(points);inside=max(hull.equations[:,:2]@point+hull.equations[:,2])<=0.;poly=points[hull.vertices]
    except QhullError:inside=False;poly=points
   else:inside=False;poly=points
   if inside:dist=0.
   else:
    distances=[]
    for j,a in enumerate(poly):
     b=poly[(j+1)%len(poly)];v=b-a;u=float(np.clip(np.dot(point-a,v)/max(np.dot(v,v),1e-30),0,1));distances.append(float(np.linalg.norm(point-a-u*v)))
    dist=min(distances)
  g=feet[k];normal=d.geom_xmat[g].reshape(3,3)@teacher.native_normal[side];tiltsole=float(np.degrees(np.arccos(np.clip(normal[2],-1,1))));plant_dist[side]=dist;plant_tilt[side]=tiltsole;plant_points[side]=points.tolist() if len(points) else []
 oldplant=getattr(env,'actual_plant_gate',{});plant_ok=all(plant_dist[s]<=.003 for s in ['left','right']) and min(row['foot_normal_N'])>20
 env.actual_plant_gate={'continuous_support_domain_s':oldplant.get('continuous_support_domain_s',0.)+.001 if plant_ok else 0.,'expected_pointmass_ZMP_distance_to_actual_contact_hull_m':plant_dist,'actual_native_sole_tilt_deg':plant_tilt,'actual_contact_points_xy':plant_points,'pointmass_assumption_not_dynamic_feasibility_proof':True};row['actual_plant_gate']=env.actual_plant_gate.copy();env.cumulative_foot_slip=slip.tolist();env.cumulative_foot_slip_total=slip_total_since_cold.tolist();row['foot_material_slip_total_since_cold_m']=slip_total_since_cold.tolist()
 if currentmeta.get('phase')=='loaded_stable_grip_hold5s':
  hm={'samples':0,'max_COM_height_error_m':0.,'max_tilt_deg':0.,'min_jaw_normal_N':float('inf')}|getattr(env,'loaded_hold_metrics',{});hm['samples']+=1;hm['max_COM_height_error_m']=max(hm['max_COM_height_error_m'],abs(row['COM_world'][2]-currentmeta['COM_reference_world'][2]));hm['max_COM_XY_error_m']=max(hm.get('max_COM_XY_error_m',0.),float(np.linalg.norm(np.array(row['COM_world'][:2])-np.array(currentmeta['COM_reference_world'][:2]))));hm['max_tilt_deg']=max(hm['max_tilt_deg'],tilt);hm['min_foot_normal_N']=min(hm.get('min_foot_normal_N',float('inf')),min(row['foot_normal_N']));hm['max_COM_speed_m_s']=max(hm.get('max_COM_speed_m_s',0.),float(np.linalg.norm(row['COM_velocity_world_m_s'])));hm['last_COM_velocity_world_m_s']=row['COM_velocity_world_m_s'];hm['min_jaw_normal_N']=min(hm['min_jaw_normal_N'],min(jawN));env.loaded_hold_metrics=hm
 if currentmeta.get('motion_start_time_s') is not None and 'capture_context' in currentmeta:
  cc=currentmeta['capture_context'];heading=np.array(currentmeta['motion_reference_heading']);row['motion_elapsed_s']=pret-currentmeta['motion_start_time_s'];row['motion_back_root_cart_cargo_m']=[float((np.array(cc[key])-pos)@heading) for key,pos in [('observed_root_world',d.qpos[:3]),('observed_cart_world',d.xpos[cart]),('observed_cargo_world',d.xpos[cargo])]];row['motion_root_cart_cargo_linear_velocity_world_m_s']=[d.qvel[:3].tolist(),d.qvel[m.jnt_dofadr[m.body_jntadr[cart]]:m.jnt_dofadr[m.body_jntadr[cart]]+3].tolist(),d.qvel[m.jnt_dofadr[m.body_jntadr[cargo]]:m.jnt_dofadr[m.body_jntadr[cargo]]+3].tolist()]
 if min(jawN)>=5:env.capture_bilateral_contact_seen=True
 grip_loss_streak=grip_loss_streak+.001 if getattr(env,'capture_bilateral_contact_seen',False) and min(jawN)<1. else 0.
 if grip_loss_streak>=.1:failure='real_grip_normal_lost_over100ms'
 height_error_streak=height_error_streak+.001 if 'COM_reference_world' in currentmeta and currentmeta['COM_reference_world'][2]-row['COM_world'][2]>.010 else 0.
 if height_error_streak>=.2:failure='COM_drop_over10mm_for200ms'
 if 'COM_reference_world' in currentmeta and currentmeta['COM_reference_world'][2]-row['COM_world'][2]>.015:failure='COM_drop_over15mm_from_reference'
 if pret>1. and abs(nonrodF[0])>max(2.,abs(rodF[0])):failure='nonrod_contact_is_main_X_transport_force'
 if tilt>25:failure='tilt_over25deg'
 if np.max(slip)>.005 and qualification_failure is None:qualification_failure={'reason':'stance_material_slip_over5mm','first_time_s':pret,'foot_material_slip_m':slip.tolist(),'root_back_m':rootback}
 if actualtcperr>.005:headerr_streak+=.001
 else:headerr_streak=0.
 if np.max(slip)>.020:failure='diagnostic_gross_material_slip_over20mm'
 if pret>=1 and not intended_support_ok:supportloss+=.001
 else:supportloss=0.
 if supportloss>.10:failure='dual_support_lost_over100ms_after_settle'
 if any(row['warnings']) or not np.isfinite(d.qpos).all():failure='nonfinite_or_mujoco_warning'
 if failure:
  dump(S/'terminal_abort_prestate.json',dict(row,abort_reason=failure,physics_step_executed=False));raise PhysicalAbort(failure)
 json.dumps(row,separators=(',',':')) # zero-side-effect serialization check before physical integration
 source_step(m,d,nstep=1);row.update(post_time_s=float(d.time),post_qpos=d.qpos.tolist(),post_qvel=d.qvel.tolist());log.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row);qtraj.append(d.qpos.copy());vtraj.append(d.qvel.copy());ts.append(float(d.time))
 if len(rows)%200==0:
  threads=max(threads,int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))));print('EVENT',num,round(d.time,2),'backmm',round(rootback*1000,2),'downmm',round(rootdown*1000,2),'Fn',row['foot_normal_N'],'slipmm',(slip*1000).round(2).tolist(),'tilt',round(tilt,2),'cartmm',round(row['cart_back_m']*1000,2),'jawN',jawN.round(1).tolist(),'phase',currentmeta.get('phase'),flush=True)
def logged(model,data,nstep=1):
 assert nstep==5
 for subindex in range(5):logged_one(model,data,nstep=1,drivefirst=subindex==0)
mj.mj_step=logged;error=None;bridge_completed=False
try:
 for tick in range(int(round(teacher.duration/.02))):
  assert datetime.datetime.now(datetime.timezone.utc)<deadline
  action,currentmeta=teacher.action(env,float(d.time),d.geom_xpos[rod].copy())
  if 'capture_context' in currentmeta and not (S/'actual_capture_to_motion_context.json').exists():
   dump(S/'actual_capture_to_motion_context.json',dict(currentmeta['capture_context'],prefix_epoch_slip_R_L=slip.tolist(),prefix_total_slip_R_L=slip_total_since_cold.tolist(),motion_start_time_s=teacher.motion_start));np.savez_compressed(S/'actual_motion_start_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),held_ctrl=d.ctrl.copy(),adapter_target=env.target.copy())
  if currentmeta.get('complete_requested_motion_hold'):break
  if gain!=1.:
   desired=np.array(currentmeta['requested_joint_targets18'])+(gain-1.)*tau/env.kp;action=(desired-env.neutral)/env.scale;currentmeta['requested_joint_targets18']=desired.tolist()
  currentmeta['bias_multiplier']=gain
  if 'capture_context' in currentmeta and not (S/'actual_capture_to_motion_context.json').exists():
   dump(S/'actual_capture_to_motion_context.json',dict(currentmeta['capture_context'],prefix_epoch_slip_R_L=slip.tolist(),prefix_total_slip_R_L=slip_total_since_cold.tolist(),motion_start_time_s=teacher.motion_start));np.savez_compressed(S/'actual_motion_start_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),held_ctrl=d.ctrl.copy(),adapter_target=env.target.copy())
  if currentmeta.get('complete_requested_motion_hold'):break
  if currentmeta.get('teacher_abort_reason'):
   failure=currentmeta['teacher_abort_reason'];dump(S/'teacher_abort_current_poststate.json',dict(time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),qacc=d.qacc.tolist(),ctrl=d.ctrl.tolist(),metadata=currentmeta));break
  if 'torque_command' in currentmeta:
   cmd=currentmeta['torque_command'];commands.append({'time_s':float(d.time),'metadata':currentmeta.copy()});env.step_torque(cmd['tau_ff18_Nm'],cmd['q_diagnostic18_rad'],cmd['qd_diagnostic18_rad_s'])
  elif 'segment_motor_command' in currentmeta:
   from runner_segment_dispatch import dispatch_segment_20ms
   cmd=currentmeta['segment_motor_command'];commands.append({'time_s':float(d.time),'metadata':currentmeta.copy()});dispatch_segment_20ms(env,cmd['packet'],cmd['tau_ff18_Nm'],teacher.motion_start,logged)
  elif 'motor_command' in currentmeta:
   cmd=currentmeta['motor_command'];qref=np.array(cmd['q_ref18_rad']);qdref=np.array(cmd['qd_ref18_rad_s']);ff=np.array(cmd['tau_ff18_Nm']);commands.append({'time_s':float(d.time),'metadata':currentmeta.copy()});env.step_motor(qref,qdref,ff)
  elif float(d.time)<2.:
   commands.append({'time_s':float(d.time),'action18':action.tolist(),'metadata':currentmeta.copy()});env.step(action)
  else:
   alpha=float(np.clip((float(d.time)-2.)/1.,0.,1.));q_only=np.array(currentmeta['requested_joint_targets18']);bias=np.array(currentmeta['actual_applied_position_bias_tau18_Nm']);ff=alpha*bias*env.torque_mask;qref=q_only-ff/env.kp;currentmeta.update(q_only_reference_request18=q_only.tolist(),requested_joint_targets18=qref.tolist(),actual_applied_position_bias_tau18_Nm=(bias-ff).tolist(),independent_ff_command18_Nm=ff.tolist(),bumpless_transfer_alpha=alpha,qd_ref18_rad_s=[0.]*18);action=(qref-env.neutral)/env.scale;commands.append({'time_s':float(d.time),'action18':action.tolist(),'metadata':currentmeta.copy()});env.step_motor(qref,np.zeros(18),ff)
  if failure:break
except PhysicalAbort:pass
except BaseException as ex:error={'error':repr(ex),'traceback':traceback.format_exc()}
finally:
 mj.mj_step=source_step;mj.mj_forward(m,d);np.savez_compressed(S/'true_terminal_poststate.npz',qpos=d.qpos,qvel=d.qvel,qacc=d.qacc,ctrl=d.ctrl,actuator_force=d.actuator_force,drive_speed=d.qvel[env.drive_vidx],efc_force=d.efc_force[:d.nefc],efc_J=d.efc_J[:d.nefc*m.nv],xpos=d.xpos,xmat=d.xmat,xipos=d.xipos);dump(S/'true_terminal_poststate.json',dict(time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),actual_tau18_Nm=d.actuator_force.tolist(),actual_drive_speed18=d.qvel[env.drive_vidx].tolist(),metadata=currentmeta,contacts=[dict(geom1=m.geom(c.geom1).name,geom2=m.geom(c.geom2).name,pos=c.pos.tolist(),frame=c.frame.tolist(),distance=float(c.dist)) for c in d.contact]));log.close();np.savez_compressed(S/'trajectory.npz',qpos=np.array(qtraj),qvel=np.array(vtraj),time=np.array(ts));dump(S/'commands50hz.json',commands);hold=[r for r in rows if r['command_metadata'].get('phase')=='final_DS_hold_no_return'];hclosure=max([abs(x) for r in hold for e in r['equalities'] for x in e['residual']],default=0);hpen=max([r['max_penetration_m'] for r in hold],default=0);gate=bool(len(hold)>=400 and min(r['root_back_m'] for r in hold)>=.015 and all(r['dual_support'] for r in hold) and max(max(r['foot_material_slip_integral_m']) for r in hold)<=.005 and hclosure<=.0005 and hpen<=.0005 and failure is None and error is None and qualification_failure is None);out={'completed_requested_horizon':bool(getattr(teacher,'complete_requested_motion_hold',False) and failure is None and error is None),'task_pass':False,'task_pass_semantics':'requires separate actual distance/speed/stop/alternatingstep/hold analysis; no reference-request award','actual_capture_hold_complete':bool(getattr(teacher,'capture_complete',False)),'actual_foot_landing_events_fine':getattr(env,'actual_foot_landing_events',[]),'actual_first_step_observed':getattr(teacher,'actual_first_step',None),'completion_not_transport_or_unloading_qualification':True,'model':'physics_v0_2_20261020_08_candidate','source_manifest_entries':79,'source_tracked_files_including_manifest':80,'attempt':num,'mode':'LOADED_ONLINE_TEACHER_PHYSICS1000_DRIVE200_COMMAND50','physics_Hz':1000,'drive_Hz':200,'command_Hz':50,'actual_drive_updates':env.controller_updates,'actual_command_updates':env.policy_updates,'MIT_substeps':env.mit_ticks,'maximum_same_state_total_tau_shadow_error_Nm':max(shadow_errors,default=0.),'actual_steps':int(round(float(d.time)/.001)),'logged_completed_rows':len(rows),'sim_duration_s':float(d.time),'failure':failure,'error':error,'final_cart_back_m':rows[-1]['cart_back_m'] if rows else None,'final_cargo_back_m':rows[-1]['cargo_back_m'] if rows else None,'max_cart_back_m':max([r['cart_back_m'] for r in rows],default=0),'jaw_capture_ever':any(min(r['jaw_rod_normal_N'])>.01 for r in rows),'final_jaw_rod_normal_N':rows[-1]['jaw_rod_normal_N'] if rows else None,'TCP_5mm_for200ms_recorded_diagnostic_only':headerr_streak>=.2,'requested_cycles_diagnostic_only':4,'planned_transport_back_m':.525,'unloaded_gate_pass':False,'irreversible_qualification_failure':qualification_failure,'diagnostic_only_not_loaded_admission':True,'hold_steps':len(hold),'hold_steps_semantics':'legacy stop_hold selector; new controller hold reported separately','new_loaded_hold_metrics':getattr(env,'loaded_hold_metrics',{}),'actual_plant_gate':getattr(env,'actual_plant_gate',{}),'actual_DS_gate':getattr(env,'actual_DS_gate',{}),'actual_landing_events':teacher.landing_events,'endpoint_semantics':'final_* fields are last logged PRE state; true terminal separately in true_terminal_poststate.json','hold_min_root_back_m':min([r['root_back_m'] for r in hold],default=None),'hold_root_back_range_m':[min([r['root_back_m'] for r in hold],default=0),max([r['root_back_m'] for r in hold],default=0)],'hold_max_closure_m_or_rad':hclosure,'hold_max_contact_penetration_m':hpen,'final_root_back_m':rows[-1]['root_back_m'] if rows else None,'max_root_back_m':max([r['root_back_m'] for r in rows],default=0),'foot_material_slip_integral_m':slip.tolist(),'foot_material_slip_total_since_cold_m':slip_total_since_cold.tolist(),'max_tilt_deg':max([r['torso_tilt_deg'] for r in rows],default=0),'max_positive_power_W':max([r['positive_mechanical_power_W'] for r in rows],default=0),'max_observed_threads':threads,'maxRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_manifest79_plus_manifest80_unchanged':all(sha(Path(p))==h for p,h in H.items()),'GPU':0,'training':0};dump(S/'summary.json',out);print('STATUS',json.dumps(out),flush=True)

# Preserve actual capture context for later separately admitted gait; no reset/anchor write.
if teacher.b.boot:
 dump(S/'capture_handoff_context.json',dict(actual_qpos=d.qpos.tolist(),actual_qvel=d.qvel.tolist(),original_env_target=env.target.tolist(),held_motor_command=env.command_details,cold_rod_fixed_anchor=teacher.b.cold.tolist(),restoring_stance_anchors={n:x.tolist() for n,x in teacher.b.anchors.items()},observed_actual_foot_anchors={n:d.geom_xpos[m.geom(n+'_actual_sole').id].tolist() for n in ['left','right']},observed_foot_rotations={n:d.geom_xmat[m.geom(n+'_actual_sole').id].tolist() for n in ['left','right']},COM_kinematic_calibration=teacher.b.COM_kinematic_calibration,plant_integral={n:x.tolist() for n,x in teacher.b.plant_integral.items()},hold_start=teacher.b.hold_start,hold_complete=teacher.capture_complete,clock_s=float(d.time),future_entry_must_match_current_target_qd_FF_no_reset=True))
