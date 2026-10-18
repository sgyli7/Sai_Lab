import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
from pathlib import Path
import json,gzip,hashlib
import numpy as np,mujoco as mj
R=Path(__file__).parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');P=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation');base=mj.MjModel.from_xml_path(str(P/'robot.xml'));C=json.loads((P/'physical_contract.json').read_text());D=json.loads((P/'actuator_contract.json').read_text());checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for S in sorted(R.glob('attempt_*')):
 if not (S/'summary.json').exists():continue
 m=mj.MjModel.from_binary_path(str(S/'actual_scene.mjb'));cnt=0
 for typ,names,fields in [('body',[base.body(i).name for i in range(1,base.nbody)],['body_mass','body_ipos','body_iquat','body_inertia','body_pos','body_quat','body_parentid']),('joint',[base.joint(i).name for i in range(base.njnt)],['jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_type','jnt_bodyid']),('geom',[base.geom(i).name for i in range(base.ngeom)],['geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_friction','geom_solref','geom_solimp','geom_condim','geom_contype','geom_conaffinity','geom_margin','geom_gap','geom_priority','geom_solmix']),('actuator',[base.actuator(i).name for i in range(base.nu)],['actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_dyntype','actuator_gaintype','actuator_biastype'])]:
  for n in names:
   a=getattr(base,typ)(n).id;b=getattr(m,typ)(n).id
   for f in fields:assert np.array_equal(getattr(base,f)[a],getattr(m,f)[b]),(typ,n,f);cnt+=1
 for i in range(1,base.njnt):
  j=base.joint(i).name;a=int(base.joint(j).dofadr[0]);b=int(m.joint(j).dofadr[0])
  for f in ['dof_armature','dof_frictionloss','dof_damping','dof_solref','dof_solimp']:assert np.array_equal(getattr(base,f)[a],getattr(m,f)[b]);cnt+=1
 for i in range(base.neq):
  for f in ['eq_type','eq_obj1id','eq_obj2id','eq_data','eq_solref','eq_solimp','eq_active0']:assert np.array_equal(getattr(base,f)[i],getattr(m,f)[i]);cnt+=1
 for i in range(base.nmesh):
  n=base.mesh(i).name;k=m.mesh(n).id;a=base.mesh_vertadr[i];b=m.mesh_vertadr[k];N=base.mesh_vertnum[i];assert N==m.mesh_vertnum[k] and np.array_equal(base.mesh_vert[a:a+N],m.mesh_vert[b:b+N]);cnt+=1
 rows=[json.loads(x) for x in gzip.open(S/'substeps.jsonl.gz','rt')];tau=np.array([r['actual_torque18'] for r in rows]);v=np.array([r['drive_speed18'] for r in rows]);caps=np.zeros_like(tau)
 for i,ax in enumerate(D['axes']):
  curve=np.array(ax['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=(1 if ax['reference_voltage_v']==5 else min(D['power_supply']['nominal_bus_v']/24,1));caps[:,i]=np.minimum(C['joints'][i]['continuous_design_limit_nm'],np.interp(abs(v[:,i]),curve[:,0],curve[:,1],right=0))
 assert np.max(abs(tau)-caps)<1e-8
 H=json.loads((S/'source_before.json').read_text());assert len(H)==53 and all(sha(p)==h for p,h in H.items());checks.append({'attempt':S.name,'compiled_robot_exact_checks':cnt,'source53_unchanged':True,'actual_voltage_v':D['power_supply']['nominal_bus_v'],'caps_verified':True,'max_torque18':np.max(abs(tau),axis=0).tolist(),'max_actual_drive_speed18':np.max(abs(v),axis=0).tolist(),'cap_saturation_duty18':np.mean(abs(tau)>=caps-1e-6,axis=0).tolist()})
manifest=json.loads((L/'analysis/cart_collision_004/artifact_manifest.json').read_text());asset=[{'path':str(L/x['path']),'expected':x['sha256'],'actual':sha(L/x['path'])} for x in manifest['files']];assert len(asset)==37 and all(x['actual']==x['expected'] for x in asset)
out={'per_trial':checks,'cart004_37_manifest':asset,'all_cart00437_match':True,'order006_sha256':sha(W/'experiments/goose_v0_2/cart_drag_probe_006/order.json'),'zero_new_integration_in_this_verifier':True};(R/'immutable_drive_verification.json').write_text(json.dumps(out,indent=2));print('VERIFIED',[(x['attempt'],x['compiled_robot_exact_checks']) for x in checks],len(asset))
