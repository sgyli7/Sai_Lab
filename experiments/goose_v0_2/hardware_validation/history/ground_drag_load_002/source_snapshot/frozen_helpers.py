from pathlib import Path
import json
import numpy as np
import mujoco as mj
ROOT=Path(__file__).resolve().parent
PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip())
C=json.loads((PKG/'physical_contract.json').read_text())
def plain(x):
 if hasattr(x,'tolist'):return x.tolist()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def mat(q):
 a=np.empty(9);mj.mju_quat2Mat(a,np.asarray(q));return a.reshape(3,3)

def fullI(m,b):return mat(m.body_iquat[b])@np.diag(m.body_inertia[b])@mat(m.body_iquat[b]).T

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

def payload_bottom(m,p):
 vals=[]
 for g in range(m.ngeom):
  if not m.geom(g).name.startswith('pilot_'):continue
  R=p.geom_xmat[g].reshape(3,3);typ=m.geom_type[g];sz=m.geom_size[g]
  extent=float(sz[0]+sz[1]*abs(R[2,2])) if typ==mj.mjtGeom.mjGEOM_CAPSULE else float(abs(R[2])@sz)
  vals.append(float(p.geom_xpos[g,2]-extent))
 return min(vals)
