"""Bounded sealed04 real-joint ground drag. Only one explicit dynamics executor."""
from pathlib import Path
import os, sys, json, time, datetime, resource, signal, atexit, hashlib, gzip, importlib.util, traceback
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent;START=time.time();A=json.loads((ROOT/'acceptance.json').read_text());DEADLINE=datetime.datetime.fromisoformat(A['absolute_deadline_UTC']).timestamp();STOP=datetime.datetime.fromisoformat(A['stop_new_experiments_UTC']).timestamp()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
def timeout(*_):raise TimeoutError('Second order absolute deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(DEADLINE-time.time()-15)))
import numpy as np
import mujoco as mj
from frozen_helpers import check_scene, fullI, payload_bottom
assert mj.__version__=='3.10.0'
PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip());C=json.loads((PKG/'physical_contract.json').read_text());B=json.loads((ROOT/'base_object_100g.json').read_text());P=json.loads((ROOT/'kinematic_plan.json').read_text())
spec=importlib.util.spec_from_file_location('sealed04_runtime_ground_drag',PKG/'runtime_adapter.py');ADAPTER=importlib.util.module_from_spec(spec);spec.loader.exec_module(ADAPTER)
BASE=mj.MjModel.from_xml_path(str(PKG/'robot.xml'));COUNTS={'dynamic_attempts':0,'physics_integrations':0,'drive_updates':0,'command_updates':0,'NN':0,'training':0,'GPU':0,'max_observed_threads':1}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def receipt():
    r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN);nt=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')));COUNTS['max_observed_threads']=max(COUNTS['max_observed_threads'],nt)
    write(ROOT/'resource_exit_dynamics.json',{'pid':os.getpid(),'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_s_from_acceptance':time.time()-datetime.datetime.fromisoformat(A['received_at_UTC']).timestamp(),'process_wall_s':time.time()-START,'within_budget':time.time()<DEADLINE,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_address_space_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'maxRSS_KiB':r.ru_maxrss,'child_maxRSS_KiB':ch.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'counts':COUNTS})
atexit.register(receipt)
def object_mci(mass):
    assert .1<=mass<=4
    x=ET.Element('mujoco');ET.SubElement(x,'compiler',inertiafromgeom='true');w=ET.SubElement(x,'worldbody');b=ET.SubElement(w,'body',name='payload');ET.SubElement(b,'freejoint');components=[]
    for g,kg in zip(B['geoms'],[.004,.003,.003,.090]):
        typ={3:'capsule',6:'box'}[g['type']];sz=g['size_m'][:2] if typ=='capsule' else g['size_m'];ET.SubElement(b,'geom',type=typ,size=' '.join(map(str,sz)),pos=' '.join(map(str,g['pos_body_m'])),quat=' '.join(map(str,g['quat_wxyz'])),mass=str(kg));components.append({'name':g['name'],'mass_kg':kg,'pos_m':g['pos_body_m'],'size_m':sz})
    extra=mass-.1;inner=[.034,.064,.034];density=extra/(8*np.prod(inner))
    assert density<=7850
    if extra>0:ET.SubElement(b,'geom',type='box',size=' '.join(map(str,inner)),pos='.08 0 -.115',mass=str(extra));components.append({'name':'physical_internal_uniform_ballast','mass_kg':extra,'pos_m':[.08,0,-.115],'half_size_m':inner,'equivalent_density_kg_m3':float(density)})
    mo=mj.MjModel.from_xml_string(ET.tostring(x,encoding='unicode'));bid=mo.body('payload').id
    return {'mass_kg':float(mo.body_mass[bid]),'com_local_m':mo.body_ipos[bid].tolist(),'inertia_local_kg_m2':fullI(mo,bid).tolist(),'component_ledger':components,'outer_contact_geometry_unchanged':True}
def scene(name,config):
    out=ROOT/'scenes'/name;out.mkdir(parents=True,exist_ok=False);xml=ET.parse(PKG/'robot.xml').getroot();xml.find('compiler').set('meshdir',os.path.relpath(PKG,out));mc=object_mci(float(config.get('mass_kg',.5)))
    pos=np.array(P['selected']['handle_target_m']);pos[0]+=float(config.get('initial_handle_x_offset_m',0.));b=ET.SubElement(xml.find('worldbody'),'body',name='pilot_payload',pos=' '.join(map(str,pos)));ET.SubElement(b,'freejoint',name='pilot_payload_free');I=np.array(mc['inertia_local_kg_m2']);ET.SubElement(b,'inertial',mass=str(mc['mass_kg']),pos=' '.join(map(str,mc['com_local_m'])),fullinertia=' '.join(map(str,[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])))
    g=BASE.geom('upper_bill_envelope').id;contact=xml.find('contact')
    for i,row in enumerate(B['geoms']):
        typ={3:'capsule',6:'box'}[row['type']];sz=row['size_m'][:2] if typ=='capsule' else row['size_m'];nameg='pilot_'+row['name'].split('/')[-1]
        ET.SubElement(b,'geom',name=nameg,type=typ,size=' '.join(map(str,sz)),pos=' '.join(map(str,row['pos_body_m'])),quat=' '.join(map(str,row['quat_wxyz'])),density='0',contype='4',conaffinity='3',condim='3',friction=' '.join(map(str,BASE.geom_friction[g])),solref=' '.join(map(str,BASE.geom_solref[g])),solimp=' '.join(map(str,BASE.geom_solimp[g])),rgba='.82 .45 .08 1' if i==0 else '.28 .48 .68 1')
        ET.SubElement(contact,'pair',name='declared_object_ground_'+nameg,geom1='ground',geom2=nameg,condim='3',friction='.3 .3 .01 .002 .002',solref=' '.join(map(str,BASE.geom_solref[g])),solimp=' '.join(map(str,BASE.geom_solimp[g])),margin='0',gap='0')
    local=config.get('mode','LOCAL_TORSO_FIXED_NECK_PULL')=='LOCAL_TORSO_FIXED_NECK_PULL'
    if local:ET.SubElement(xml.find('equality'),'weld',name='pilot_stationary_torso_fixture',body1='torso',solref='.005 1',solimp='.99 .9999 .0001',torquescale='1')
    if config.get('negative_no_jaw_contact'):
        for body in ['head_roll','beak_hinge']:ET.SubElement(contact,'exclude',body1='pilot_payload',body2=body)
    ET.ElementTree(xml).write(out/'robot.xml',encoding='utf-8',xml_declaration=True)
    for n in ['physical_contract.json','actuator_contract.json']:(out/n).write_bytes((PKG/n).read_bytes())
    write(out/'scene_declaration.json',{'config':config,'payload':mc,'payload_initial_pose':[*pos,1,0,0,0],'fixture':'stationary torso-only external soft weld; head and neck are NEVER fixed' if local else 'NONE; free root and no added equalities','object_ground_pairs':'explicit declared material mu=.3; no robot material or solver changes','source_robot_XML_SHA256':sha(PKG/'robot.xml'),'scene_XML_SHA256':sha(out/'robot.xml'),'kinematic_plan_SHA256':sha(ROOT/'kinematic_plan.json')})
    return out,mc
def path_request(env,t,config,cold_named):
    q=np.array([cold_named.get(n,0.) for n in env.names]);q[5]=0.
    u=float(np.clip((t-float(config.get('pull_start_s',1.)))/2.,0,1));u=u*u*(3-2*u)
    if config.get('mode','LOCAL_TORSO_FIXED_NECK_PULL')=='LOCAL_TORSO_FIXED_NECK_PULL':
        rows=P['neck_pull_path_60mm'];z=np.array([np.interp(u,[x['u'] for x in rows],[x['neck_rad'][k] for x in rows]) for k in range(3)])
        for n,v in zip(P['neck_axes'],z):q[env.names.index(n)]=v
    else:
        assert P['free_base_path_admitted_kinematically']
        rows=P['free_base_short_double_stance_joint_path']
        for n in rows[0]['named_rad']:q[env.names.index(n)]=np.interp(u,[x['u'] for x in rows],[x['named_rad'][n] for x in rows])
        rows=P['free_base_neck_height_compensation']
        for k,n in enumerate(P['neck_axes']):q[env.names.index(n)]=np.interp(u,[x['u'] for x in rows],[x['neck_rad'][k] for x in rows])
    return (q-env.neutral)/env.scale,u
def run(config):
    assert time.time()<STOP and COUNTS['dynamic_attempts']<5
    registry= json.loads((ROOT/'attempt_registry.json').read_text()) if (ROOT/'attempt_registry.json').exists() else []
    number=len(registry)+1;name=f'attempt_{number:03d}';out=ROOT/name;out.mkdir(exist_ok=False);sc,mc=scene(name,config);env=ADAPTER.FrozenPhysics(sc);m,d=env.model,env.data;exact=check_scene(m,BASE)
    named=P['selected']['neck_named_rad'].copy();named['beak_hinge']=float(config.get('cold_jaw_q_rad',.19682523003160665));env.reset(named)
    init_contacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist),'effective_mu5':c.friction.tolist(),'dim':int(c.dim)} for c in d.contact];deep=[c for c in init_contacts if c['dist_m']<-.0001];assert not deep,deep
    base_mode=config.get('mode','LOCAL_TORSO_FIXED_NECK_PULL');expected_ne=BASE.neq+(1 if base_mode=='LOCAL_TORSO_FIXED_NECK_PULL' else 0);assert m.neq==expected_ne
    pbody=m.body('pilot_payload').id;head=m.body('head_roll').id;torso=m.body('torso').id;ground=m.geom('ground').id;payload={m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith('pilot_')};jaws=[m.geom('upper_bill_envelope').id,m.geom('lower_bill_envelope').id];feet=[m.geom('right_actual_sole').id,m.geom('left_actual_sole').id]
    assert abs(m.body_mass[pbody]-mc['mass_kg'])<1e-12 and np.max(abs(fullI(m,pbody)-mc['inertia_local_kg_m2']))<1e-12
    mj.mj_saveModel(m,str(out/'actual_scene.mjb'));np.savez(out/'cold_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy());qrecord=[d.qpos.copy()];vrecord=[d.qvel.copy()];times=[0.];rows=[];commands=[];first_failure=None;both_ever=False;lost_start=None;physical_index=0;threadmax=1
    cold_obj=d.xpos[pbody].copy();cold_torso=d.xpos[torso].copy();cold_rel=d.xmat[head].reshape(3,3).T@(d.xpos[pbody]-d.xpos[head]);coldR=d.xmat[head].reshape(3,3).T@d.xmat[pbody].reshape(3,3);initial_footpos=d.geom_xpos[feet].copy();pull_obj_ref=None;pull_body_ref=None;first20=None
    pin=mj.MjData(m);post=mj.MjData(m);source_step=mj.mj_step;lasttarget=env.target.copy();logicalq=env.qidx.copy();logicalv=env.vidx.copy();actualv=env.drive_vidx.copy();foot_slip=np.zeros(2);support_state=np.zeros(2,dtype=bool);stance_start=initial_footpos.copy();stance_peak=np.zeros(2);maxstance=np.zeros(2)
    registry.append({'attempt':number,'status':'admitted_before_dynamics','config':config,'scene_SHA256':sha(sc/'robot.xml'),'harness_SHA256':sha(Path(__file__)),'controller_SHA256':sha(PKG/'runtime_adapter.py'),'initial_contacts':init_contacts,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()});write(ROOT/'attempt_registry.json',registry);write(out/'declaration.json',{**registry[-1],'frozen_scene_readback':exact,'payload_MCI':mc,'actual_cold_named_q_rad':named,'mode':base_mode,'control':'bounded offline FK/IK joint path;50Hz requests through unchanged200Hz frozen adapter; no learned policy or root writes','no_head_fixture':True,'initial_pose_source':'cold_state.npz','cold_relative_head_position_m':cold_rel.tolist(),'escape_margin_definition':'conservative head-X projection of rod radius6mm within shared78.4mm zero-angle CAD working-face interval; not a force closure test'})
    COUNTS['dynamic_attempts']+=1
    def logged_step(model,data,nstep=1):
        nonlocal physical_index,lasttarget,first_failure,both_ever,lost_start,pull_obj_ref,pull_body_ref,first20,threadmax
        assert model is m and nstep==1 and time.time()<STOP
        q0=data.qpos.copy();v0=data.qvel.copy();pin.qpos[:]=q0;pin.qvel[:]=v0;mj.mj_kinematics(m,pin);mj.mj_comPos(m,pin)
        requested=env.kp*(env.target-q0[logicalq])-env.kd*v0[logicalv];expected=env.limit_torque(requested,v0[actualv]);assert np.max(abs(expected-data.ctrl))<1e-12;assert np.max(abs(env.target-lasttarget)-env.slew_speed*.005)<1e-12;lasttarget=env.target.copy();assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
        source_step(m,data,nstep=1);physical_index+=1;COUNTS['physics_integrations']+=1;COUNTS['drive_updates']+=1;force=data.actuator_force.copy();assert np.max(abs(force-expected))<1e-10
        pplus0=float(np.maximum(force*v0[actualv],0).sum());pplus1=float(np.maximum(force*data.qvel[actualv],0).sum());assert max(pplus0,pplus1)<=350+1e-8
        caps=env.cont.copy()
        for i,e in enumerate(env.entries):
            curve=np.array(e['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=1 if e['reference_voltage_v']==5 else env.voltage_speed_factor;caps[i]=min(caps[i],float(np.interp(abs(v0[actualv[i]]),curve[:,0],curve[:,1],right=0)))
        assert np.max(abs(force)-caps)<1e-9
        normjaw=np.zeros(2);Wjaw=np.zeros((2,6));Wpayload=np.zeros(6);groundN=0.;groundF=np.zeros(3);footN=np.zeros(2);footW=np.zeros((2,6));footv=np.zeros(2);contacts=[];extras=[];maxjawpen=0.;maxfloorpen=0.;nonfoot_ground=[]
        for ci,c in enumerate(data.contact):
            f=np.zeros(6)
            if c.efc_address>=0:mj.mj_contactForce(m,data,ci,f)
            a,b=int(c.geom1),int(c.geom2);R=c.frame.reshape(3,3);rec={'geom1':m.geom(a).name,'geom2':m.geom(b).name,'distance_m':float(c.dist),'dim':int(c.dim),'effective_friction5':c.friction.tolist(),'solref':c.solref.tolist(),'solimp':c.solimp.tolist(),'point_world_m':c.pos.tolist(),'frame_rows_world':c.frame.tolist(),'actual_force_contact_6D':f.tolist(),'efc_address':int(c.efc_address)}
            if a in payload or b in payload:
                sign=1 if b in payload else -1;other=a if b in payload else b;F=sign*R.T@f[:3];T=sign*R.T@f[3:]+np.cross(c.pos-pin.xipos[pbody],F);w=np.r_[F,T];Wpayload+=w;rec['wrench_on_payload_about_incoming_COM_world']=w.tolist()
                if other in jaws:
                    k=jaws.index(other);Wjaw[k]+=w;maxjawpen=max(maxjawpen,-float(c.dist))
                    if m.geom(a if a in payload else b).name=='pilot_grip':normjaw[k]+=max(0.,f[0])
                    elif f[0]>.001:extras.append('non_handle_mouth_contact:'+m.geom(a if a in payload else b).name)
                    assert c.dim==3 and np.allclose(c.friction[:2],[.65,.65])
                elif other==ground:
                    groundN+=max(0.,F[2]);groundF+=F;maxfloorpen=max(maxfloorpen,-float(c.dist));assert c.dim==3 and np.allclose(c.friction[:2],[.3,.3])
                elif f[0]>.001:extras.append(m.geom(other).name)
            if ground in [a,b]:
                other=b if a==ground else a
                if other in feet:
                    k=feet.index(other);sign=1 if b==other else -1;F=sign*R.T@f[:3];T=sign*R.T@f[3:]+np.cross(c.pos-pin.geom_xpos[other],F);footN[k]+=max(0.,F[2]);footW[k]+=np.r_[F,T];assert np.allclose(c.friction[:2],[.65,.65]);jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jac(m,pin,jp,jr,c.pos,int(m.geom_bodyid[other]));vel=jp@v0;rec['foot_point_velocity_world_m_s']=vel.tolist();footv[k]+=max(0.,F[2])*float(np.linalg.norm(vel[:2]))
                elif other not in payload and f[0]>.01:nonfoot_ground.append({'geom':m.geom(other).name,'normal_N':float(f[0])})
            contacts.append(rec)
        for k in range(2):
            if footN[k]>2:
                foot_slip[k]+=.005*footv[k]/footN[k]
                if not support_state[k]:stance_start[k]=pin.geom_xpos[feet[k]];stance_peak[k]=0.
                stance_peak[k]=max(stance_peak[k],float(np.linalg.norm((pin.geom_xpos[feet[k]]-stance_start[k])[:2])));maxstance[k]=max(maxstance[k],stance_peak[k])
            support_state[k]=footN[k]>2
        post.qpos[:]=data.qpos;mj.mj_kinematics(m,post);mj.mj_comPos(m,post);H=post.xmat[head].reshape(3,3);rel=H.T@(post.xpos[pbody]-post.xpos[head]);relR=H.T@post.xmat[pbody].reshape(3,3);margin=.0392-.006-abs(float(rel[0]-P['tip_point_head_local_m'][0]));bottom=payload_bottom(m,post);up=float(post.xmat[torso].reshape(3,3)[2,2]);obj_back=float(cold_obj[0]-post.xpos[pbody,0]);body_back=float(cold_torso[0]-post.xpos[torso,0]);both=bool(min(normjaw)>1e-6);both_ever=both_ever or both
        if data.time>=float(config.get('pull_start_s',1.))-1e-10 and pull_obj_ref is None:pull_obj_ref=post.xpos[pbody].copy();pull_body_ref=post.xpos[torso].copy()
        if obj_back>=.02 and first20 is None:first20=float(data.time)
        if both:lost_start=None
        elif both_ever and lost_start is None:lost_start=float(data.time)
        failures=[]
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failures.append('nonfinite')
        if any(w.number for w in data.warning):failures.append('mujoco_warning')
        if up<.64:failures.append('body_tilt_over50deg')
        if any(x['normal_N']>5 for x in nonfoot_ground):failures.append('nonfoot_ground_support')
        if extras:failures.append('non_grip_robot_payload_contact')
        if maxjawpen>.01 or maxfloorpen>.01:failures.append('penetration_over10mm')
        if margin<-.006:failures.append('handle_outside_conservative_grasp_domain')
        if lost_start is not None and data.time-lost_start>.3:failures.append('bilateral_grip_lost_over300ms')
        if not both_ever and data.time>1.5:failures.append('no_bilateral_capture_by1p5s')
        limited=np.where(m.jnt_limited)[0];jq=data.qpos[m.jnt_qposadr[limited]];rangeerr=float(np.maximum(m.jnt_range[limited,0]-jq,jq-m.jnt_range[limited,1]).max())
        if rangeerr>.02:failures.append('joint_range_violation_over20mrad')
        if failures and first_failure is None:first_failure={'time_s':float(data.time),'reasons':failures,'object_backwards_m':obj_back,'torso_backwards_m':body_back,'margin_m':margin,'jaw_positive_normal_N':normjaw.tolist(),'ground_N':groundN}
        constraints=[]
        for ei in range(m.neq):
            idx=np.where((data.efc_type[:data.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id[:data.nefc]==ei))[0];row={'name':m.equality(ei).name,'position_residual':data.efc_pos[idx].tolist(),'actual_constraint_row_force':data.efc_force[idx].tolist()}
            if ei>=BASE.neq:
                J=data.efc_J.reshape(data.nefc,m.nv);Q=J[idx].T@data.efc_force[idx];jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacBody(m,pin,jp,jr,torso);G=np.vstack([jp,jr]).T;w=np.linalg.lstsq(G,Q,rcond=None)[0];row.update(actual_generalized_reaction=Q.tolist(),fixture_wrench_world_about_torso_origin=w.tolist(),wrench_reconstruction_error=float(abs(G@w-Q).max()))
            constraints.append(row)
        rec={'step':physical_index,'time_s':float(data.time),'incoming_time_s':float(data.time)-.005,'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'incoming_qpos':q0.tolist(),'incoming_qvel':v0.tolist(),'target18':env.target.tolist(),'requested_PD18':requested.tolist(),'actual_torque18':force.tolist(),'limited_torque18':expected.tolist(),'torque_speed_caps18':caps.tolist(),'actual_drive_velocity_incoming18':v0[actualv].tolist(),'actual_drive_velocity_after18':data.qvel[actualv].tolist(),'selected_voltage_derated_slew18':env.slew_speed.tolist(),'positive_mechanical_power_incoming_w':pplus0,'positive_mechanical_power_after_w':pplus1,'bus_voltage_v':env.bus_voltage_v,'relative_payload_head_position_m':rel.tolist(),'relative_payload_head_R':relR.tolist(),'grip_translation_from_true_cold_m':float(np.linalg.norm(rel-cold_rel)),'conservative_grasp_X_margin_m':margin,'jaw_positive_normal_N':normjaw.tolist(),'jaw_wrenches_about_COM_world':Wjaw.tolist(),'all_payload_contact_wrench_about_COM_world':Wpayload.tolist(),'bilateral_positive':both,'object_backwards_from_cold_m':obj_back,'object_backwards_from_pull_entry_m':None if pull_obj_ref is None else float(pull_obj_ref[0]-post.xpos[pbody,0]),'torso_backwards_from_cold_m':body_back,'torso_backwards_from_pull_entry_m':None if pull_body_ref is None else float(pull_body_ref[0]-post.xpos[torso,0]),'torso_up_cosine':up,'payload_lowest_z_m':bottom,'object_ground_normal_N':groundN,'object_ground_force_world_N':groundF.tolist(),'foot_ground_normal_N':footN.tolist(),'foot_wrenches_about_incoming_geom_origins_world':footW.tolist(),'foot_stance_on':support_state.tolist(),'foot_contact_slip_integral_m':foot_slip.tolist(),'foot_max_stance_center_excursion_m':maxstance.tolist(),'nonfoot_ground_support':nonfoot_ground,'non_grip_robot_payload_contacts':extras,'max_jaw_payload_penetration_m':max(0.,maxjawpen),'max_object_floor_penetration_m':max(0.,maxfloorpen),'joint_range_max_violation_rad':max(0.,rangeerr),'equalities':constraints,'contacts':contacts,'warnings':[int(w.number) for w in data.warning],'external_applied_forces_zero':not bool(data.qfrc_applied.any() or data.xfrc_applied.any())}
        log.write(json.dumps(rec,separators=(',',':'),allow_nan=False)+'\n');qrecord.append(data.qpos.copy());vrecord.append(data.qvel.copy());times.append(float(data.time));rows.append({k:rec[k] for k in ['time_s','jaw_positive_normal_N','bilateral_positive','object_backwards_from_cold_m','object_backwards_from_pull_entry_m','torso_backwards_from_cold_m','torso_up_cosine','object_ground_normal_N','foot_ground_normal_N','conservative_grasp_X_margin_m','max_jaw_payload_penetration_m','max_object_floor_penetration_m','positive_mechanical_power_after_w']})
        if physical_index%200==0:
            nt=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')));COUNTS['max_observed_threads']=max(COUNTS['max_observed_threads'],nt);write(out/'progress.json',{'steps':physical_index,'last':rows[-1],'first_failure':first_failure});print('EVENT',name,round(float(data.time),3),'obj_back_mm',round(obj_back*1000,2),'body_back_mm',round(body_back*1000,2),'Fn',normjaw.round(2).tolist(),'groundN',round(groundN,2),flush=True)
    error=None;mj.mj_step=logged_step
    try:
        with gzip.open(out/'actual_substeps.jsonl.gz','wt',compresslevel=3) as log:
            for tick in range(int(float(config.get('duration_s',6.))/.02)):
                action,u=path_request(env,float(d.time),config,named);commands.append({'tick':tick,'incoming_time_s':float(d.time),'action18':action.tolist(),'smooth_path_u':u});env.step(action);COUNTS['command_updates']+=1
                if first_failure is not None and float(d.time)>first_failure['time_s']+.2:break
    except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:
        mj.mj_step=source_step;np.savez_compressed(out/'actual_trajectory.npz',qpos=np.array(qrecord),qvel=np.array(vrecord),time=np.array(times));write(out/'commands50Hz.json',commands);write(out/'compact_metrics.json',rows)
        loaded=[r for r in rows if r['time_s']>=float(config.get('pull_start_s',1.))];weight=mc['mass_kg']*9.81;meanN=None if not loaded else float(np.mean([r['object_ground_normal_N'] for r in loaded]));duty=None if not loaded else float(np.mean([r['object_ground_normal_N']>.01*weight for r in loaded]));bothduty=None if not loaded else float(np.mean([r['bilateral_positive'] for r in loaded]));flat='UNOBSERVED' if meanN is None else ('FLAT_NOMINAL' if meanN>=.9*weight else ('PARTIAL_UNLOADING' if meanN>=.1*weight else 'NEAR_SUSPENSION_EXCLUDED'))
        s={'attempt':number,'mode':base_mode,'config':config,'admission_passed':True,'mass_kg':mc['mass_kg'],'actual_steps':physical_index,'actual_command_updates':env.policy_updates,'actual_drive_updates':env.controller_updates,'sim_duration_s':float(d.time),'bilateral_capture_ever':both_ever,'bilateral_loaded_duty':bothduty,'first_failure':first_failure,'runtime_error':error,'startup20mm_time_s':first20,'startup20mm_within5s':first20 is not None and first20<=5,'final_object_backwards_m':None if not rows else rows[-1]['object_backwards_from_cold_m'],'max_object_backwards_m':max([r['object_backwards_from_cold_m'] for r in rows],default=0),'final_object_backwards_from_pull_entry_m':None if not rows else rows[-1]['object_backwards_from_pull_entry_m'],'final_torso_backwards_m':None if not rows else rows[-1]['torso_backwards_from_cold_m'],'max_torso_backwards_m':max([r['torso_backwards_from_cold_m'] for r in rows],default=0),'loaded_mean_ground_normal_N':meanN,'weight_N':weight,'loaded_ground_contact_duty':duty,'ground_support_classification':flat,'min_grasp_margin_m':min([r['conservative_grasp_X_margin_m'] for r in rows],default=None),'max_jaw_payload_penetration_m':max([r['max_jaw_payload_penetration_m'] for r in rows],default=0),'max_object_floor_penetration_m':max([r['max_object_floor_penetration_m'] for r in rows],default=0),'foot_contact_slip_integral_m':foot_slip.tolist(),'foot_max_stance_center_excursion_m':maxstance.tolist(),'max_positive_mechanical_power_w':max([r['positive_mechanical_power_after_w'] for r in rows],default=0),'continuous60s':'NOT_EXECUTED','formal_0p5m_whole_robot':'NOT_QUALIFIED_SHORT_PILOT','actual_trajectory_SHA256':sha(out/'actual_trajectory.npz'),'actual_raw_SHA256':sha(out/'actual_substeps.jsonl.gz'),'model_MJB_SHA256':sha(out/'actual_scene.mjb'),'NN':0,'GPU':0,'training':0}
        write(out/'summary.json',s);registry[-1].update(status='returned',summary=str(out/'summary.json'),actual_integrations=physical_index);write(ROOT/'attempt_registry.json',registry);print('STATUS',json.dumps(s,separators=(',',':')),flush=True)
    return s
def serve():
    write(ROOT/'experiment_process.json',{'pid':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'single_executor':True,'deadline_UTC':A['absolute_deadline_UTC']});print('READY second order single executor; explicit commands only',flush=True)
    for line in sys.stdin:
        if not line.strip():continue
        cmd=json.loads(line)
        with (ROOT/'executed_requests.jsonl').open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':cmd})+'\n')
        if cmd['operation']=='stop':break
        assert cmd['operation']=='attempt';run(cmd['config']);print('READY',flush=True)
if __name__=='__main__':
    try:serve()
    except BaseException as e:write(ROOT/'worker_exception.json',{'error':repr(e),'traceback':traceback.format_exc(),'counts':COUNTS});raise
