"""Bounded04 pilot. Robot/drive immutable; all writes confined to this result tree."""
from pathlib import Path
import argparse, atexit, datetime, hashlib, importlib.util, json, os, resource, signal, sys, time, traceback
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
ACCEPT=json.loads((ROOT/'acceptance.json').read_text())
DEADLINE=datetime.datetime.fromisoformat(ACCEPT['absolute_deadline_UTC']).timestamp()
STOP_EXPERIMENT=datetime.datetime.fromisoformat(ACCEPT['stop_new_experiments_UTC']).timestamp()
STAGE=sys.argv[1] if len(sys.argv)>1 else 'audit'
START=time.time()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','LP_NUM_THREADS']:
 os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
os.environ['GALLIUM_DRIVER']='softpipe'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
COUNTERS={'physics_integrations':0,'attempts_started':0,'training_updates':0,'NN_forwards':0,'max_observed_threads':1}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def plain(x):
 if hasattr(x,'tolist'):return x.tolist()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x

def resource_exit():
 r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
 status=Path('/proc/self/status').read_text();threads=int(next(x.split()[1] for x in status.splitlines() if x.startswith('Threads:')))
 COUNTERS['max_observed_threads']=max(threads,COUNTERS['max_observed_threads'])
 write(ROOT/f'resource_exit_{STAGE}.json',{'stage':STAGE,'pid':os.getpid(),'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'process_wall_s':time.time()-START,'wall_s_from_acceptance':time.time()-datetime.datetime.fromisoformat(ACCEPT['received_at_UTC']).timestamp(),'deadline_UTC':ACCEPT['absolute_deadline_UTC'],'within_wall_budget':time.time()<=DEADLINE,'CPU_affinity':sorted(os.sched_getaffinity(0)),'memory_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'self_maxRSS_KiB':r.ru_maxrss,'child_maxRSS_KiB':ch.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'children_CPU_s':ch.ru_utime+ch.ru_stime,'counters':COUNTERS,'GPU_devices_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'GPU_use':0})
atexit.register(resource_exit)
def timeout(signum,frame):raise TimeoutError('Absolute accepted resource deadline reached')
signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(DEADLINE-time.time()-10)))
import numpy as np
import mujoco as mj
PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip())
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ADAPTER=module('sealed04_runtime',PKG/'runtime_adapter.py')
C=json.loads((PKG/'physical_contract.json').read_text());DRIVE=json.loads((PKG/'actuator_contract.json').read_text());OBJECT=json.loads((ROOT/'historical_full_object.json').read_text())

def mat(q):
 a=np.empty(9);mj.mju_quat2Mat(a,np.asarray(q));return a.reshape(3,3)
def fullI(m,b):return mat(m.body_iquat[b])@np.diag(m.body_inertia[b])@mat(m.body_iquat[b]).T

def make_scene(name,grip_x=None,negative=False,upper_gap=.001):
 out=ROOT/'scenes'/name;out.mkdir(parents=True,exist_ok=False)
 xml=ET.parse(PKG/'robot.xml').getroot();xml.find('compiler').set('meshdir',os.path.relpath(PKG,out))
 ref=np.array(C['grip_reference_zero_m']);ref[2]+=float(xml.find('./worldbody/body[@name="torso"]').get('pos').split()[2]);ref[0]=float(ref[0] if grip_x is None else grip_x)
 pos=ref+np.array([0.,0.,-.006-upper_gap])
 body=ET.SubElement(xml.find('worldbody'),'body',name='pilot_payload',pos=' '.join(map(str,pos)))
 ET.SubElement(body,'freejoint',name='pilot_payload_free')
 I=np.array(OBJECT['inertia_local_kg_m2'])
 ET.SubElement(body,'inertial',mass=str(OBJECT['mass_kg']),pos=' '.join(map(str,OBJECT['com_local_m'])),fullinertia=' '.join(map(str,[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])))
 base=mj.MjModel.from_xml_path(str(PKG/'robot.xml'));g=base.geom('upper_bill_envelope').id
 for k,row in enumerate(OBJECT['geoms']):
  typ={3:'capsule',6:'box'}[row['type']];sz=row['size_m'][:2] if typ=='capsule' else row['size_m']
  ET.SubElement(body,'geom',name='pilot_'+row['name'].split('/')[-1],type=typ,size=' '.join(map(str,sz)),pos=' '.join(map(str,row['pos_body_m'])),quat=' '.join(map(str,row['quat_wxyz'])),density='0',contype='4',conaffinity='3',condim='3',friction=' '.join(map(str,base.geom_friction[g])),solref=' '.join(map(str,base.geom_solref[g])),solimp=' '.join(map(str,base.geom_solimp[g])),rgba='.8 .45 .12 1' if k==0 else '.3 .5 .7 1')
 eq=xml.find('equality')
 for bodyname in ['torso','head_roll']:
  ET.SubElement(eq,'weld',name='pilot_fixture_'+bodyname,body1=bodyname,solref='.005 1',solimp='.99 .9999 .0001',torquescale='1')
 if negative:
  for bodyname in ['head_roll','beak_hinge']:ET.SubElement(xml.find('contact'),'exclude',body1='pilot_payload',body2=bodyname)
 scene=out/'robot.xml';ET.ElementTree(xml).write(scene,encoding='utf-8',xml_declaration=True)
 for n in ['physical_contract.json','actuator_contract.json']:(out/n).write_bytes((PKG/n).read_bytes())
 write(out/'scene_declaration.json',{'scene_SHA256':sha(scene),'payload_mass_kg':.1,'full_payload_geometry':OBJECT['geoms'],'payload_initial_position_m':pos.tolist(),'payload_initial_quat_wxyz':[1,0,0,0],'cold_jaw_angle_rad':.25,'robot_changes':{'only_external_fixture_constraints':['pilot_fixture_torso','pilot_fixture_head_roll'],'body_and_head_fixed_for_LOCAL_DIAGNOSTIC_only':True,'full_robot_qualification':False,'original3internal_equalities_retained':True},'negative_control_jaw_payload_excluded':negative,'no_payload_equality_or_force_or_pose_write_after_birth':True,'contact_material_source':'04 upper jaw numeric friction/solref/solimp, condim3, no oldcontactparameters imported'})
 return out

def check_scene(m,base):
 checks=[]
 for kind,names,fields in [
  ('body',[b['name'] for b in C['bodies']],['body_mass','body_ipos','body_iquat','body_inertia','body_pos','body_quat','body_parentid']),
  ('joint',[j['name'] for j in C['joints']+C['passive_linkage_joints']],['jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_type','jnt_bodyid']),
  ('geom',[base.geom(i).name for i in range(base.ngeom)],['geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_friction','geom_solref','geom_solimp','geom_condim','geom_contype','geom_conaffinity','geom_margin','geom_gap','geom_priority','geom_solmix']),
  ('actuator',[base.actuator(i).name for i in range(base.nu)],['actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_dyntype','actuator_gaintype','actuator_biastype'])]:
  for n in names:
   a=getattr(base,kind)(n).id;b=getattr(m,kind)(n).id
   for f in fields:
    same=np.array_equal(getattr(base,f)[a],getattr(m,f)[b]);checks.append(same)
    assert same,(kind,n,f)
 for j in C['joints']+C['passive_linkage_joints']:
  a=int(base.joint(j['name']).dofadr[0]);b=int(m.joint(j['name']).dofadr[0])
  for f in ['dof_armature','dof_frictionloss','dof_damping','dof_solref','dof_solimp']:
   assert np.array_equal(getattr(base,f)[a],getattr(m,f)[b]),(j['name'],f);checks.append(True)
 for f in ['mesh_vert','mesh_face','mesh_graph','mesh_graphadr','mesh_vertnum','mesh_facenum']:
  assert np.array_equal(getattr(base,f),getattr(m,f)),f;checks.append(True)
 options={}
 for n in dir(base.opt):
  if n.startswith('_'):continue
  a=getattr(base.opt,n);b=getattr(m.opt,n)
  if isinstance(a,(int,float,np.ndarray)):
   assert np.array_equal(a,b),('options',n);options[n]=plain(a)
 for i in range(base.neq):
  for f in ['eq_type','eq_obj1id','eq_obj2id','eq_data','eq_solref','eq_solimp','eq_active0']:
   assert np.array_equal(getattr(base,f)[i],getattr(m,f)[i]),(i,f)
 return {'named_robot_fields_exact_count':len(checks),'all_named_robot_fields_and_original_internal_equalities_exact':True,'options_exact':options,'extra_constraints_declared':int(m.neq-base.neq)}

def audit():
 assert mj.__version__=='3.10.0',mj.__version__
 checker=module('sealed04_verify_physics',PKG/'verify_physics.py');_,_,ledger=checker.contract_check(PKG)
 write(ROOT/'mass_ledger_source_audit.json',ledger)
 base=mj.MjModel.from_xml_path(str(PKG/'robot.xml'));env=ADAPTER.FrozenPhysics(PKG)
 rows=[]
 for b in C['bodies']:
  bid=base.body(b['name']).id;r={'name':b['name'],'mass_error':abs(float(base.body_mass[bid])-b['mass_kg']),'COM_error_m':float(abs(base.body_ipos[bid]-b['com_local_m']).max()),'fullI_error_kgm2':float(abs(fullI(base,bid)-b['inertia_at_com_body_kg_m2']).max())};assert max(r[k] for k in r if k!='name')<1e-10;rows.append(r)
 # Independent full assembly MCI from the four component masses, not stem-only.
 ob=ET.Element('mujoco');ET.SubElement(ob,'compiler',inertiafromgeom='true');wb=ET.SubElement(ob,'worldbody');bb=ET.SubElement(wb,'body',name='object');ET.SubElement(bb,'freejoint')
 for mass,g in zip([.004,.003,.003,.090],OBJECT['geoms']):
  typ={3:'capsule',6:'box'}[g['type']];sz=g['size_m'][:2] if typ=='capsule' else g['size_m'];ET.SubElement(bb,'geom',type=typ,size=' '.join(map(str,sz)),pos=' '.join(map(str,g['pos_body_m'])),quat=' '.join(map(str,g['quat_wxyz'])),mass=str(mass))
 om=mj.MjModel.from_xml_string(ET.tostring(ob,encoding='unicode'));oid=om.body('object').id
 objerrs={'mass_kg':abs(float(om.body_mass[oid])-.1),'COM_m':float(abs(om.body_ipos[oid]-OBJECT['com_local_m']).max()),'fullI_kg_m2':float(abs(fullI(om,oid)-OBJECT['inertia_local_kg_m2']).max())}
 # MCI source may use explicitly selected component inertias; preserve all differences.
 write(ROOT/'object_MCI_audit.json',{'component_masses_kg':[.004,.003,.003,.09],'complete_components':[g['name'] for g in OBJECT['geoms']],'historical':OBJECT,'recomputed_COM_m':om.body_ipos[oid].tolist(),'recomputed_fullI':fullI(om,oid).tolist(),'difference':objerrs,'uniform_primitive_component_reconstruction_exact':max(objerrs.values())<1e-10})
 assert max(objerrs.values())<1e-8,('Complete payload MCI inconsistent',objerrs)
 scene=make_scene('audit_fixture');m=mj.MjModel.from_xml_path(str(scene/'robot.xml'));same=check_scene(m,base);rt=ADAPTER.FrozenPhysics(scene);rt.reset({'beak_hinge':.25});m,d=rt.model,rt.data
 pids=[i for i in range(m.ngeom) if m.geom(i).name.startswith('pilot_')];payload=m.body('pilot_payload').id;grip=m.geom('pilot_grip').id
 grippairs=[]
 for name in ['upper_bill_envelope','lower_bill_envelope']:
  g=m.geom(name).id;seg=np.empty(6);distance=mj.mj_geomDistance(m,d,g,grip,.1,seg);grippairs.append({'name':name,'open_025_real_gap_m':float(distance),'closest_segment':seg.tolist(),'frozen_condim':int(m.geom_condim[g]),'solref':m.geom_solref[g].tolist(),'solimp':m.geom_solimp[g].tolist(),'friction':m.geom_friction[g].tolist(),'geom_bbox_size':m.geom_size[g].tolist()})
 contacts=[]
 for ct in d.contact:
  if ct.geom1 in pids or ct.geom2 in pids:contacts.append({'g1':m.geom(ct.geom1).name,'g2':m.geom(ct.geom2).name,'dist':float(ct.dist),'dim':int(ct.dim),'friction5':ct.friction.tolist(),'solref':ct.solref.tolist(),'solimp':ct.solimp.tolist()})
 assert not any(x['dist']<-.0001 for x in contacts),('Initial object penetration',contacts)
 assert m.nu==18 and base.nbody-1==23 and sum(base.geom_bodyid>0)==14
 hull=[{'geom':base.geom(g).name,'hull_vertices':int(base.mesh_graph[base.mesh_graphadr[base.geom_dataid[g]]]),'hull_triangles':int(base.mesh_graph[base.mesh_graphadr[base.geom_dataid[g]]+1])} for g in range(base.ngeom) if base.geom_bodyid[g]>0]
 r={'audit_passed':True,'human_authorization_verified':True,'engine':mj.__version__,'engine_path':mj.__file__,'active_solver':'unmodified sealed04','base_robot_mass_kg':float(base.body_mass.sum()),'base_robot_bodies_excluding_world':base.nbody-1,'robot_collision_leaves':len(hull),'hull_vertices':sum(x['hull_vertices'] for x in hull),'hull_triangles':sum(x['hull_triangles'] for x in hull),'hulls':hull,'MCI_body_readback_errors':rows,'named_mapping':[{'name':n,'logical_q_index':int(q),'logical_v_index':int(v),'actual_motor_dof':int(av),'actuator_joint':m.joint(int(m.actuator_trnid[i,0])).name} for i,(n,q,v,av) in enumerate(zip(rt.names,rt.qidx,rt.vidx,rt.drive_vidx))],'same_robot_scene':same,'full_object_MCI_check':objerrs,'local_fixture_only':True,'fixture':'torso and head_roll external welds; no payload equality; originalmechanism3 constraints retained','robot_body_inertia_or_mesh_modifications':False,'real_open_gaps':grippairs,'initial_payload_contacts':contacts,'active_contact_condim':3,'no_condim6_or_oldribs_or_custom_collision_callback':True,'bus_voltage_v':rt.bus_voltage_v,'torque_dt_s':rt.drive_dt,'physics_dt_s':rt.dt,'command_dt_s':C['policy_dt_s'],'torque_updates_per_command':rt.ticks,'peak_output_enabled':DRIVE['peak_output_enabled'],'mechanical_positive_budget_w':DRIVE['positive_mechanical_power_limit_w'],'object_ground_and_foot_ground_effective_sliding_friction':.65,'material_coefficients_are_unmeasured_design_assumptions':True,'actual_integrations':0,'NN':0,'training_updates':0}
 write(ROOT/'preflight_audit.json',r);print('AUDIT_PASS',json.dumps({k:r[k] for k in ['engine','base_robot_mass_kg','robot_collision_leaves','hull_vertices','full_object_MCI_check','real_open_gaps']}),flush=True)


import gzip

def counters_threads():
 n=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')))
 COUNTERS['max_observed_threads']=max(COUNTERS['max_observed_threads'],n)
 assert n<=4,('CPU thread budget exceeded',n)

def payload_bottom(m,p):
 vals=[]
 for g in range(m.ngeom):
  if not m.geom(g).name.startswith('pilot_'):continue
  R=p.geom_xmat[g].reshape(3,3);typ=m.geom_type[g];sz=m.geom_size[g]
  extent=float(sz[0]+sz[1]*abs(R[2,2])) if typ==mj.mjtGeom.mjGEOM_CAPSULE else float(abs(R[2])@sz)
  vals.append(float(p.geom_xpos[g,2]-extent))
 return min(vals)

def run_attempt(config):
 assert time.time()<STOP_EXPERIMENT,'Reserved closeout period reached'
 audit=json.loads((ROOT/'preflight_audit.json').read_text());assert audit['audit_passed'] and mj.__version__=='3.10.0'
 registry_file=ROOT/'attempt_registry.json';registry=json.loads(registry_file.read_text()) if registry_file.exists() else []
 assert len(registry)<5,'Five-attempt cap reached'
 number=len(registry)+1;name=f'attempt_{number:03d}';out=ROOT/name;out.mkdir(exist_ok=False)
 scene=make_scene(name,config.get('grip_x'),bool(config.get('negative',False)),config.get('upper_gap_m',.001))
 env=ADAPTER.FrozenPhysics(scene);m,d=env.model,env.data;zero=mj.MjModel.from_xml_path(str(PKG/'robot.xml'));exact=check_scene(m,zero)
 qstart=float(config.get('qstart',.25))
 if config.get('tangent_birth'):
  # Declared cold initialization only. Solve geometric lower contact at zero upper gap.
  lo,hi=.05,.35;grip=m.geom('pilot_grip').id;lower=m.geom('lower_bill_envelope').id
  for _ in range(45):
   mid=(lo+hi)/2;env.reset({'beak_hinge':mid});seg=np.empty(6);gap=mj.mj_geomDistance(m,d,lower,grip,.1,seg)
   if gap<0:lo=mid
   else:hi=mid
  qstart=hi
 env.reset({'beak_hinge':qstart})
 cold=mj.MjData(m);cold.qpos[:]=d.qpos;mj.mj_kinematics(m,cold);mj.mj_comPos(m,cold)
 pbody=m.body('pilot_payload').id;head=m.body('head_roll').id;payload={g for g in range(m.ngeom) if m.geom(g).name.startswith('pilot_')};jaws=[m.geom('upper_bill_envelope').id,m.geom('lower_bill_envelope').id]
 object_initial_contacts=[]
 for c in d.contact:
  if c.geom1 in payload or c.geom2 in payload:object_initial_contacts.append({'g1':m.geom(c.geom1).name,'g2':m.geom(c.geom2).name,'dist_m':float(c.dist)})
 admission=not any(c['dist_m']<-.0001 for c in object_initial_contacts)
 rowreg={'attempt':number,'status':'started' if admission else 'not_admitted','config':config,'cold_qjaw_rad':qstart,'scene':str(scene/'robot.xml'),'scene_SHA256':sha(scene/'robot.xml'),'controller_source_SHA256':sha(PKG/'runtime_adapter.py'),'harness_source_SHA256':sha(Path(__file__)),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_object_contacts':object_initial_contacts}
 registry.append(rowreg);write(registry_file,registry);COUNTERS['attempts_started']+=1
 write(out/'declaration.json',{**rowreg,'frozen_robot_readback':exact,'scope':'LOCAL_FIXED_HEAD_AND_TORSO_DIAGNOSTIC','full_robot_claim':False,'peak_enabled':False,'bus_voltage_v':env.bus_voltage_v,'payload_MCI':json.loads((ROOT/'object_MCI_audit.json').read_text())})
 np.savez(out/'cold_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy());mj.mj_saveModel(m,str(out/'actual_scene.mjb'))
 if not admission:
  r={'attempt':number,'admission_passed':False,'initial_contacts':object_initial_contacts,'physics_integrations':0,'success5s':False,'success30s':False};write(out/'summary.json',r);registry[-1].update(status='not_admitted',summary=str(out/'summary.json'));write(registry_file,registry);return r
 probe=mj.MjData(m);incoming=mj.MjData(m);first_rel=None;first_R=None;first_contact_pose=None;hold_start=None;hold_ref=None;hold_R=None;hold_failure=None;first_failure=None;decision5=None;decision30=None;max_hold_drift=0.;max_hold_angle=0.;min_clearance=999.;max_penetration=0.;max_drive_error=0.;max_power=0.;max_incoming_power=0.;max_peak_speed_ratio=0.;selected_speed_events=0;holds=[];records_q=[d.qpos.copy()];records_v=[d.qvel.copy()];records_t=[0.];allsmall=[];warnings=[];drive_updates=[];sourceStep=mj.mj_step;last_target=env.target.copy();incomingpos={};fixture_ids=[m.equality('pilot_fixture_'+n).id for n in ['torso','head_roll']];positive_lower_count=0
 start_integration=COUNTERS['physics_integrations'];local_steps=0
 def log_step(model,data,nstep=1):
  nonlocal first_rel,first_R,first_contact_pose,hold_start,hold_ref,hold_R,hold_failure,first_failure,decision5,decision30,max_hold_drift,max_hold_angle,min_clearance,max_penetration,max_drive_error,max_power,max_incoming_power,max_peak_speed_ratio,selected_speed_events,last_target,local_steps,positive_lower_count
  assert model is m and nstep==1 and abs(env.drive_dt-.005)<1e-15
  if time.time()>=STOP_EXPERIMENT:raise TimeoutError('Closeout reserve started')
  q0=data.qpos.copy();v0=data.qvel.copy();incoming.qpos[:]=q0;mj.mj_kinematics(m,incoming);mj.mj_comPos(m,incoming)
  requested=env.kp*(env.target-q0[env.qidx])-env.kd*v0[env.vidx];expected=env.limit_torque(requested,v0[env.drive_vidx]);assert np.max(abs(expected-data.ctrl))<1e-12
  assert np.all(abs(env.target-last_target)<=env.slew_speed*.005+1e-12);last_target=env.target.copy()
  assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
  sourceStep(model,data,nstep=1);COUNTERS['physics_integrations']+=1;local_steps+=1
  actuator=data.actuator_force.copy();max_drive_error=max(max_drive_error,float(abs(actuator-expected).max()));assert max_drive_error<1e-10
  actual_p=float(np.maximum(actuator*data.qvel[env.drive_vidx],0).sum());inc_p=float(np.maximum(actuator*v0[env.drive_vidx],0).sum());max_power=max(max_power,actual_p);max_incoming_power=max(max_incoming_power,inc_p)
  speedratio=abs(data.qvel[env.drive_vidx])/env.slew_speed;max_peak_speed_ratio=max(max_peak_speed_ratio,float(speedratio.max()));selected_speed_events+=int(any(speedratio>1+1e-7))
  bounds=np.zeros(18)
  for i,a in enumerate(env.entries):
   curve=np.array(a['selected_torque_speed_curve_rad_s_nm']);curve=curve.copy()
   if a['reference_voltage_v']!=5:curve[:,0]*=env.voltage_speed_factor
   bounds[i]=min(env.cont[i],float(np.interp(abs(v0[env.drive_vidx[i]]),curve[:,0],curve[:,1],right=0)))
  assert np.all(abs(actuator)<=bounds+1e-9) and inc_p<=350+1e-8 and actual_p<=350+1e-8
  netW=np.zeros(6);sideW=np.zeros((2,6));normal=np.zeros(2);extra=[];contacts=[]
  for i,c in enumerate(data.contact):
   f=np.zeros(6)
   if c.efc_address>=0:mj.mj_contactForce(m,data,i,f)
   a,b=int(c.geom1),int(c.geom2);onpayload=(a in payload or b in payload);cf={'index':i,'geom1':m.geom(a).name,'geom2':m.geom(b).name,'dist_m':float(c.dist),'dim':int(c.dim),'efc_address':int(c.efc_address),'pos_world_m':c.pos.tolist(),'frame_rows_world':c.frame.tolist(),'actual_force_contact_frame_6D':f.tolist(),'effective_friction5':c.friction.tolist(),'solref':c.solref.tolist(),'solimp':c.solimp.tolist()}
   if onpayload:
    sign=1 if b in payload else -1;other=a if b in payload else b;R=c.frame.reshape(3,3);F=sign*(R.T@f[:3]);T=sign*(R.T@f[3:])+np.cross(c.pos-incoming.xipos[pbody],F);w=np.r_[F,T];netW+=w;cf['actual_wrench_on_payload_about_incoming_COM_world']=w.tolist()
    if other in jaws:
     k=jaws.index(other);sideW[k]+=w;normal[k]+=max(0.,f[0])
    elif f[0]>1e-6:extra.append(m.geom(other).name)
    max_penetration=max(max_penetration,max(0.,-float(c.dist)))
   contacts.append(cf)
  probe.qpos[:]=data.qpos;mj.mj_kinematics(m,probe);mj.mj_comPos(m,probe)
  H=probe.xmat[head].reshape(3,3);P=probe.xmat[pbody].reshape(3,3);relative=H.T@(probe.xpos[pbody]-probe.xpos[head]);relativeR=H.T@P
  if first_rel is None:first_rel=relative.copy();first_R=relativeR.copy()
  cold_drift=float(np.linalg.norm(relative-first_rel));cquat=np.empty(4);mj.mju_mat2Quat(cquat,(relativeR@first_R.T).ravel());cold_angle=float(2*np.arctan2(np.linalg.norm(cquat[1:]),abs(cquat[0]))*180/np.pi)
  both=bool(min(normal)>1e-6)
  if both and first_contact_pose is None:first_contact_pose={'time_s':float(data.time),'relative_m':relative.tolist(),'relative_R':relativeR.tolist()}
  bottom=payload_bottom(m,probe);min_clearance=min(min_clearance,bottom)
  if hold_start is None and env.target[5]<=1e-10 and both:
   hold_start=float(data.time);hold_ref=relative.copy();hold_R=relativeR.copy()
  drift=angle=None;elapsed=None
  if hold_start is not None:
   drift=float(np.linalg.norm(relative-hold_ref));qt=np.empty(4);mj.mju_mat2Quat(qt,(relativeR@hold_R.T).ravel());angle=float(2*np.arctan2(np.linalg.norm(qt[1:]),abs(qt[0]))*180/np.pi);elapsed=float(data.time)-hold_start;max_hold_drift=max(max_hold_drift,drift);max_hold_angle=max(max_hold_angle,angle)
   failures=[]
   if drift>.002:failures.append('relative_translation_over_2mm')
   if angle>2:failures.append('relative_rotation_over_2deg')
   if extra:failures.append('other_payload_support_or_contact')
   if not both:failures.append('bilateral_positive_force_interrupted')
   if failures and hold_failure is None:hold_failure={'time_s':float(data.time),'hold_elapsed_s':elapsed,'reasons':failures,'relative_drift_m':drift,'relative_angle_deg':angle,'jaw_normal_N':normal.tolist()}
   if elapsed>=5-1e-10 and decision5 is None:decision5={'passed':hold_failure is None,'time_s':float(data.time),'hold_start_s':hold_start,'max_relative_translation_m':max_hold_drift,'max_relative_rotation_deg':max_hold_angle,'failure':hold_failure}
   if elapsed>=30-1e-10 and decision30 is None:decision30={'passed':hold_failure is None,'time_s':float(data.time),'hold_start_s':hold_start,'max_relative_translation_m':max_hold_drift,'max_relative_rotation_deg':max_hold_angle,'failure':hold_failure}
  # A drop/acquisition failure is retained even if later some geometry touches again.
  fail=[]
  if np.linalg.norm(relative-first_rel)>.03:fail.append('grip_center_departed_30mm')
  if extra:fail.append('non_jaw_payload_positive_contact')
  if max_penetration>.005:fail.append('payload_penetration_over_5mm')
  if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():fail.append('nonfinite')
  if any(w.number for w in data.warning):fail.append('mujoco_warning')
  limited=np.where(m.jnt_limited)[0];jq=data.qpos[m.jnt_qposadr[limited]];range_err=float(np.maximum(m.jnt_range[limited,0]-jq,jq-m.jnt_range[limited,1]).max());
  if range_err>.001:fail.append('joint_limit_violation_over_1mrad')
  if fail and first_failure is None:first_failure={'time_s':float(data.time),'reasons':fail,'relative_from_cold_m':cold_drift,'relative_from_cold_deg':cold_angle,'extra_payload_contact':extra,'payload_lowest_z_m':bottom}
  fixture=[]
  for eqid,bodyname in zip(fixture_ids,['torso','head_roll']):
   idx=np.where((data.efc_type[:data.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id[:data.nefc]==eqid))[0];J=data.efc_J.reshape(data.nefc,m.nv);Q=J[idx].T@data.efc_force[idx];jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacBody(m,incoming,jp,jr,m.body(bodyname).id);w,_,_,_=np.linalg.lstsq(np.vstack([jp,jr]).T,Q,rcond=None);err=float(abs(np.vstack([jp,jr]).T@w-Q).max());fixture.append({'equality':m.equality(eqid).name,'constraint_rows':idx.tolist(),'row_force':data.efc_force[idx].tolist(),'row_position_error':data.efc_pos[idx].tolist(),'generalized_reaction':Q.tolist(),'recovered_world_wrench_about_body_origin':w.tolist(),'wrench_reconstruction_max_error':err})
  looprows=np.where((data.efc_type[:data.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id[:data.nefc]<3))[0]
  rec={'step':local_steps,'time_s':float(data.time),'incoming_time_s':float(data.time)-.005,'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'incoming_qpos':q0.tolist(),'incoming_qvel':v0.tolist(),'target18':env.target.tolist(),'requested_PD_torque18':requested.tolist(),'limited_torque18':expected.tolist(),'actual_actuator_force18':actuator.tolist(),'continuous_torque_speed_caps18':bounds.tolist(),'actual_drive_velocity_incoming18':v0[env.drive_vidx].tolist(),'actual_drive_velocity_after18':data.qvel[env.drive_vidx].tolist(),'drive_power_positive_incoming_w':inc_p,'drive_power_positive_after_w':actual_p,'selected_voltage_derated_speed18':env.slew_speed.tolist(),'drive_error_max_Nm':float(abs(actuator-expected).max()),'bus_voltage_v':env.bus_voltage_v,'relative_payload_head_translation_m':relative.tolist(),'relative_payload_head_R':relativeR.tolist(),'relative_from_cold_translation_m':cold_drift,'relative_from_cold_rotation_deg':cold_angle,'hold_elapsed_s':elapsed,'hold_relative_drift_m':drift,'hold_relative_angle_deg':angle,'payload_lowest_z_m':bottom,'jaw_positive_normal_N':normal.tolist(),'bilateral_positive':both,'extra_payload_positive_contacts':extra,'actual_payload_contact_wrench_about_incoming_COM_world':netW.tolist(),'actual_jaw_wrenches_about_incoming_COM_world':sideW.tolist(),'contacts':contacts,'fixture_reactions':fixture,'internal_mechanism_eq_max_abs_position_error':float(abs(data.efc_pos[looprows]).max()) if len(looprows) else None,'warnings':[int(w.number) for w in data.warning],'external_applied_force_zero':not bool(data.qfrc_applied.any() or data.xfrc_applied.any()),'joint_range_max_violation_rad':max(0.,range_err)}
  log.write(json.dumps(rec,allow_nan=False,separators=(',',':'))+'\n');records_q.append(data.qpos.copy());records_v.append(data.qvel.copy());records_t.append(float(data.time));allsmall.append({'time_s':float(data.time),'normal_N':normal.tolist(),'hold_drift_m':drift,'hold_angle_deg':angle,'cold_drift_m':cold_drift,'cold_angle_deg':cold_angle,'bottom_m':bottom,'both':both,'extra':extra,'loop_error':rec['internal_mechanism_eq_max_abs_position_error'],'power_w':actual_p,'fixture_wrench':[f['recovered_world_wrench_about_body_origin'] for f in fixture]})
  if local_steps%200==0:
   counters_threads();write(out/'progress.json',{'steps':local_steps,'time_s':float(data.time),'hold_start_s':hold_start,'decision5':decision5,'first_failure':first_failure,'hold_failure':hold_failure});print('EVENT',name,round(float(data.time),3),'Fn',normal.round(3).tolist(),'hold',None if elapsed is None else round(elapsed,3),flush=True)
 mj.mj_step=log_step;error=None;control=[]
 try:
  with gzip.open(out/'actual_substeps.jsonl.gz','wt',compresslevel=3) as log:
   for tick in range(1800):
    requested=env.neutral.copy();requested[5]=max(0.,env.target[5]-float(config.get('closing_step_rad',1.)))
    action=(requested-env.neutral)/env.scale;control.append({'tick':tick,'incoming_time_s':float(d.time),'action18':action.tolist(),'jaw_request_rad':float(requested[5])});env.step(action)
    if first_failure is not None and (float(d.time)>first_failure['time_s']+.6 or float(d.time)>=5):break
    if decision30 is not None:break
    if decision5 is not None and not decision5['passed']:break
    if hold_start is None and float(d.time)>=3.:
     if first_failure is None:first_failure={'time_s':float(d.time),'reasons':['no_admitted_bilateral_hold_within_3s']}
     break
   else:error={'reason':'pilot_simulation_horizon_exhausted'}
 except Exception as e:
  error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  mj.mj_step=sourceStep
  np.savez_compressed(out/'actual_trajectory.npz',qpos=np.array(records_q),qvel=np.array(records_v),time=np.array(records_t));write(out/'commands50Hz.json',control);write(out/'compact_metrics.json',allsmall)
  r={'attempt':number,'scope':'LOCAL_FIXED_HEAD_AND_TORSO_ONLY','admission_passed':True,'config':config,'initial_qjaw_rad':qstart,'first_positive_bilateral_pose':first_contact_pose,'hold_start_s':hold_start,'hold_reference_policy':'first bilateral step after canonical jaw target reached0, no hold-reference reset, no future reclassification after failure','cold_to_final_relative_translation_m':None if not allsmall else allsmall[-1]['cold_drift_m'],'cold_to_final_relative_rotation_deg':None if not allsmall else allsmall[-1]['cold_angle_deg'],'cold_acquisition_max_translation_m':max([s['cold_drift_m'] for s in allsmall],default=0),'cold_acquisition_max_rotation_deg':max([s['cold_angle_deg'] for s in allsmall],default=0),'hold5s':decision5,'hold30s':decision30,'first_physical_failure':first_failure,'hold_first_failure':hold_failure,'runtime_exception':error,'actual_steps':local_steps,'actual_50Hz_commands':env.policy_updates,'actual_200Hz_drive_updates':env.controller_updates,'sim_duration_s':float(d.time),'max_payload_penetration_m':max_penetration,'min_full_object_clearance_m':min_clearance,'max_drive_recomputation_error_Nm':max_drive_error,'max_total_positive_mechanical_power_after_w':max_power,'max_total_positive_mechanical_power_incoming_w':max_incoming_power,'max_selected_voltage_derated_speed_ratio':max_peak_speed_ratio,'selected_speed_exceeded_substeps':selected_speed_events,'extra_support_any':any(s['extra'] for s in allsmall),'mechanism_max_position_error_m_or_rad':max([s['loop_error'] or 0 for s in allsmall],default=0),'warnings':[int(w.number) for w in d.warning],'scene_SHA256':sha(scene/'robot.xml'),'model_MJB_SHA256':sha(out/'actual_scene.mjb'),'trajectory_SHA256':sha(out/'actual_trajectory.npz'),'raw_substeps_SHA256':sha(out/'actual_substeps.jsonl.gz'),'NN':0,'GPU':0,'training_updates':0}
  write(out/'summary.json',r);registry[-1].update(status='returned',summary=str(out/'summary.json'),actual_integrations=local_steps);write(registry_file,registry)
  print('STATUS',json.dumps(r,separators=(',',':')),flush=True)
 return r

def serve():
 assert json.loads((ROOT/'preflight_audit.json').read_text())['audit_passed']
 write(ROOT/'experiment_process.json',{'pid':os.getpid(),'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'single_experiment_executor':True,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_address_space_bytes':resource.getrlimit(resource.RLIMIT_AS)[0]})
 print('READY pilot commands as JSON lines; no automatic batch',flush=True)
 for line in sys.stdin:
  if not line.strip():continue
  cmd=json.loads(line)
  with (ROOT/'executed_requests.jsonl').open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':cmd})+'\n')
  if cmd.get('operation')=='stop':break
  if cmd.get('operation')=='attempt':run_attempt(cmd.get('config',{}))
  elif cmd.get('operation')=='render':render_attempt(int(cmd['attempt']))
  else:raise ValueError('Unknown operation')
  print('READY',flush=True)

if __name__=='__main__':
 try:
  if STAGE=='audit':audit()
  elif STAGE=='serve':serve()
  else:raise ValueError('Unknown stage')
 except Exception as e:
  write(ROOT/f'failure_{STAGE}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'counts':COUNTERS});raise
