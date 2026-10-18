import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
from pathlib import Path
import json,gzip,hashlib,subprocess
import numpy as np,mujoco as mj
R=Path(__file__).parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');P=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';base=mj.MjModel.from_xml_path(str(P/'robot.xml'));C=json.loads((P/'physical_contract.json').read_text());drive=json.loads((P/'actuator_contract.json').read_text());out=[]
def plain(x):return x.tolist() if hasattr(x,'tolist') else str(x)
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=plain))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for number,S in enumerate([R,R/'attempt_002',R/'attempt_003',R/'attempt_004',R/'attempt_005'],1):
 rows=[json.loads(s) for s in gzip.open(S/'substeps.jsonl.gz','rt')];m=mj.MjModel.from_binary_path(str(S/'actual_scene.mjb'));d=mj.MjData(m);traj=np.load(S/'trajectory.npz');checks=0
 for typ,names,fields in [('body',[base.body(i).name for i in range(1,base.nbody)],['body_mass','body_ipos','body_iquat','body_inertia','body_pos','body_quat','body_parentid']),('joint',[base.joint(i).name for i in range(base.njnt)],['jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_type','jnt_bodyid']),('geom',[base.geom(i).name for i in range(base.ngeom)],['geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_friction','geom_solref','geom_solimp','geom_condim','geom_contype','geom_conaffinity','geom_margin','geom_gap','geom_priority','geom_solmix']),('actuator',[base.actuator(i).name for i in range(base.nu)],['actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_dyntype','actuator_gaintype','actuator_biastype'])]:
  for n in names:
   ia=getattr(base,typ)(n).id;ib=getattr(m,typ)(n).id
   for f in fields:assert np.array_equal(getattr(base,f)[ia],getattr(m,f)[ib]),(typ,n,f);checks+=1
 for j in [base.joint(i).name for i in range(1,base.njnt)]:
  a=int(base.joint(j).dofadr[0]);b=int(m.joint(j).dofadr[0])
  for f in ['dof_armature','dof_frictionloss','dof_damping','dof_solref','dof_solimp']:assert np.array_equal(getattr(base,f)[a],getattr(m,f)[b]),(j,f);checks+=1
 for i in range(base.neq):
  for f in ['eq_type','eq_obj1id','eq_obj2id','eq_data','eq_solref','eq_solimp','eq_active0']:assert np.array_equal(getattr(base,f)[i],getattr(m,f)[i]),(i,f);checks+=1
 for i in range(base.nmesh):
  n=base.mesh(i).name;k=m.mesh(n).id;a=base.mesh_vertadr[i];b=m.mesh_vertadr[k];N=base.mesh_vertnum[i];assert N==m.mesh_vertnum[k] and np.array_equal(base.mesh_vert[a:a+N],m.mesh_vert[b:b+N]);checks+=1
 phases={}
 for phase in ['open','approach','close','pull','hold','release']:
  a=[r for r in rows if r['phase']==phase]
  if not a:continue
  phase0=rows.index(a[0]);prior=rows[max(0,phase0-1)];phases[phase]={'duration_s':len(a)*.005,'cart_backward_delta_m':a[-1]['cart_back_m']-prior['cart_back_m'],'cart_backward_start_m':prior['cart_back_m'],'cart_backward_end_m':a[-1]['cart_back_m'],'rod_z_min_max_m':[min(r['rod_world'][2] for r in a),max(r['rod_world'][2] for r in a)],'jaw_normal_mean_N':np.mean([r['jaw_normal_N'] for r in a],axis=0),'jaw_normal_min_N':np.min([r['jaw_normal_N'] for r in a],axis=0),'tow_on_rod_mean_world_N':np.mean([r['tow_on_rod_world_N'] for r in a],axis=0),'bilateral_duty':np.mean([min(r['jaw_normal_N'])>.01 for r in a]),'slip_end_m':a[-1]['slip_from_capture_m'],'angle_end_rad':a[-1]['angle_from_capture_rad'],'slip_max_m':max([r['slip_from_capture_m'] or 0 for r in a])}
 torque=np.array([r['torque18'] for r in rows]);vel=np.array([r['drive_speed18_incoming'] for r in rows]);caps=np.zeros_like(torque)
 for i,axis in enumerate(drive['axes']):
  curve=np.array(axis['selected_torque_speed_curve_rad_s_nm']);caps[:,i]=np.minimum(C['joints'][i]['continuous_design_limit_nm'],np.interp(abs(vel[:,i]),curve[:,0],curve[:,1],right=0))
 assert np.max(abs(torque)-caps)<1e-8
 contacts=[c for r in rows for c in r['contacts']];shaft=[c for c in contacts if any(x in c['a']+c['b'] for x in ['upper_bill','lower_bill']) and any(x in c['a']+c['b'] for x in ['rotating_shaft','bite_yoke','pitch_offset']) and c['force_contact'][0]>.01]
 ground=m.geom('ground').id;footnames=['right_actual_sole','left_actual_sole'];feet=[m.geom(n).id for n in footnames];slip=np.zeros(2);foot0=None;footmax=np.zeros(2);tilt=[];cargo_rel=[];support=[]
 for k,r in enumerate(rows):
  d.qpos[:]=traj['qpos'][k+1];d.qvel[:]=traj['qvel'][k+1];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
  pos=d.geom_xpos[feet].copy()
  if foot0 is None:foot0=pos.copy()
  footmax=np.maximum(footmax,np.linalg.norm((pos-foot0)[:,:2],axis=1));tilt.append(float(np.arccos(np.clip(d.xmat[m.body('torso').id].reshape(3,3)[2,2],-1,1))))
  for i,g in enumerate(feet):
   if r['feet_normal_N'][i]>.01:
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacGeom(m,d,jp,jr,g);slip[i]+=.005*np.linalg.norm((jp@d.qvel)[:2])
  cart=m.body('cart_base').id;cargo=m.body('probe_cargo').id;cargo_rel.append(d.xmat[cart].reshape(3,3).T@(d.xpos[cargo]-d.xpos[cart]))
  support.append(sum(c['force_contact'][0] for c in r['contacts'] if set([c['a'],c['b']])==set(['cart_deck_frame','probe_cargo_box'])))
 video=S/'continuous_original_speed.mp4';probe=subprocess.run(['ffmpeg','-i',str(video),'-f','null','-'],capture_output=True,text=True);meta={'decoded_returncode':probe.returncode,'ffmpeg_readback':probe.stderr};assert probe.returncode==0
 result={'attempt':number,'directory':str(S),'steps':len(rows),'sim_s':rows[-1]['time_s'],'scope':'LOCAL_FIXED_TORSO_NECK_HEAD_ACTIVE','phases':phases,'max_torque_nm18':np.max(abs(torque),axis=0),'max_drive_speed_rad_s18':np.max(abs(vel),axis=0),'torque_cap_saturation_duty18':np.mean(abs(torque)>=caps-1e-6,axis=0),'max_positive_power_W':max(r['positive_power_W'] for r in rows),'caps_verified_against_selected_curve':True,'max_jaw_penetration_m':max(r['jaw_pen_m'] for r in rows),'max_all_contact_penetration_m':max(-c['dist_m'] for c in contacts),'nonbar_mouth_shaft_yoke_positive_contact_rows':len(shaft),'original_internal_closure_max_abs_m_or_rad':max(abs(x) for r in rows for eq in r['equalities'][:3] for x in eq['residual']),'fixture_max_abs_residual_m_or_rad':max(abs(x) for r in rows for x in r['equalities'][3]['residual']),'max_body_tilt_deg':np.rad2deg(max(tilt)),'foot_contact_center_slip_integral_m':slip,'foot_geom_center_max_excursion_from_first_step_m':footmax,'cargo_cart_local_end_m':cargo_rel[-1],'cargo_local_max_excursion_m':np.max(np.abs(np.array(cargo_rel)-cargo_rel[0]),axis=0),'cargo_deck_support_mean_N':np.mean(support),'compiled_robot_exact_checks':checks,'warning_max':max(max(r['warning_counts']) for r in rows),'video':str(video),'video_probe':meta,'summary':json.loads((S/'summary.json').read_text())};dump(S/'analysis.json',result);out.append(result)
H=json.loads((R/'source_before.json').read_text());frozen={p:h for p,h in H.items() if '__pycache__' not in p};assert len(frozen)==53 and all(sha(Path(p))==h for p,h in frozen.items());order=json.loads((W/'experiments/goose_v0_2/cart_drag_probe_005/order.json').read_text());inputs=[{'path':v['path'],'expected':v['sha256'],'actual':sha(Path(v['path'])),'match':sha(Path(v['path']))==v['sha256']} for v in order['sources']];assert all(x['match'] for x in inputs);dump(R/'immutable_sources_verified.json',{'frozen53':frozen,'order_inputs':inputs,'all_match':True,'compiled_exact_checks_each':[a['compiled_robot_exact_checks'] for a in out]});dump(R/'all_attempts_analysis.json',out);print('ANALYSIS_OK',sum(a['steps'] for a in out),[a['compiled_robot_exact_checks'] for a in out])
