"""Independent R2 cart audit: saved CAD, compilation and FK only; never calls mj_step."""
from pathlib import Path
import json, hashlib, collections, time
import numpy as np
import build123d as bd
import mujoco as mj
from scipy.spatial import ConvexHull
W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots')
L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation')
E=W/'robots/goose_v0_2/evidence/low_profile_stock_cart_01'; A=L/'assets/low_profile_stock_cart_01'; O=L/'analysis/cart_independent_review_001'; R=L/'inputs/physics_v0_2_20261018_04'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def save(n,d): (O/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def par(m,d): return m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
def mat(q):
 r=np.zeros(9);mj.mju_quat2Mat(r,np.asarray(q));return r.reshape(3,3)
start=time.monotonic(); p=W/'robots/goose_v0_2/cad/exports/low_profile_stock_cart_01/goose_low_profile_stock_cart_01.step'; native=bd.import_step(p)
screen=load(E/'geometry_mass_screen.json'); parts={r['name']:r for r in screen['parts']}; param=load(W/'robots/goose_v0_2/cad/source/low_profile_stock_cart_01/parameters.json')
print('native imported',flush=True)
# Independent native shape volumes, COMs and complete inertia, not just producer flags.
rows=[]; shapes={}
for child in native.children:
 c=np.array(tuple(child.center(bd.CenterOf.MASS)));v=child.volume
 matches=[n for n,q in parts.items() if abs(q['volume_mm3']-v)<1e-4 and np.max(abs(c-q['COM_mm']))<1e-5]
 assert len(matches)==1,(child.label,matches)
 assert matches[0] not in shapes
 shapes[matches[0]]=child
assert len(shapes)==122
tote=bd.Box(600,400,150).translate((325,0,256.5))
for name,s in shapes.items():
 q=parts[name]; rho=param['materials'][q['material']]['density_kg_m3']*1e-9; v=s.volume; c=np.array(tuple(s.center(bd.CenterOf.MASS))); I=np.array(s.matrix_of_inertia)*rho*1e-6
 common=s.intersect(tote); overlap=0 if common is None else common.volume
 rows.append(dict(name=name,solids=len(s.solids()),positive_volume=v>0,valid=s.is_valid,volume_error_mm3=abs(v-q['volume_mm3']),mass_error_kg=abs(v*rho-q['mass_kg']),COM_error_mm=float(np.max(abs(c-q['COM_mm']))),fullI_error_kg_m2=float(np.max(abs(I-q['full_inertia_about_COM_kg_m2']))),tote_overlap_mm3=float(overlap)))
M=sum(q['mass_kg'] for q in parts.values()); cm=sum(q['mass_kg']*np.array(q['COM_mm']) for q in parts.values())/M
native_report=dict(step_sha256=sha(p),part_count=len(shapes),solid_count=len(native.solids()),valid=native.is_valid,bounds_mm=[list(native.bounding_box().min),list(native.bounding_box().max)],CAD_mass_kg=M,CAD_COM_mm=cm.tolist(),reserve_kg=screen['unmodelled_reserve_kg'],estimated_empty_mass_kg=M+screen['unmodelled_reserve_kg'],max_volume_error_mm3=max(r['volume_error_mm3'] for r in rows),max_COM_error_mm=max(r['COM_error_mm'] for r in rows),max_fullI_error_kg_m2=max(r['fullI_error_kg_m2'] for r in rows),positive_tote_overlaps=[r for r in rows if r['tote_overlap_mm3']>.001],all_positive_valid=all(r['positive_volume'] and r['valid'] and r['solids']==1 for r in rows),gusset_bounds={n:[list(s.bounding_box().min),list(s.bounding_box().max)] for n,s in shapes.items() if n.startswith('mast_base_gusset')},parts=rows)
save('independent_native_geometry.json',native_report);print('native checks complete',flush=True)
# Compile saved assets; read actual engine graph, not display triangles.
m=mj.MjModel.from_xml_path(str(A/'goose_cart_scene.xml')); d=mj.MjData(m); old=mj.MjModel.from_xml_path(str(R/'robot.xml'))
active=[i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]; cart=[i for i in active if m.geom(i).name.startswith('cart_')]; robot=[i for i in active if m.geom_bodyid[i] and i not in cart]
meshrows=[]
for g in cart:
 if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
  mid=m.geom_dataid[g]; adr=m.mesh_graphadr[mid]; graph=m.mesh_graph[adr:];nv,nf=map(int,graph[:2]); meshrows.append(dict(geom=m.geom(g).name,loaded_vertices=int(m.mesh_vertnum[mid]),actual_support_vertices=nv,actual_hull_triangles=nf))
# Native CAD-ledger -> each body parallel-axis aggregate -> actual engine full tensors.
bmap=load(A/'body_mass_map.json'); used=[]; massrows=[]
for name,rec in bmap['body_map'].items():
 bid=m.body(name).id; items=[parts[n] for n in rec['source_parts']]; used+=rec['source_parts']; rs=[(q['mass_kg'],np.array(q['COM_mm'])*.001,np.array(q['full_inertia_about_COM_kg_m2'])) for q in items]
 if rec['reserve_kg']:
  dm=np.array([.650,.450,.030]);ri=.35/12*np.diag([dm[1]**2+dm[2]**2,dm[0]**2+dm[2]**2,dm[0]**2+dm[1]**2]);rs.append((.35,np.array([.260,0,.180]),ri))
 mass=sum(x[0] for x in rs); c=sum(v*x for v,x,I in rs)/mass; I=sum(I+par(v,x-c) for v,x,I in rs); engineI=mat(m.body_iquat[bid])@np.diag(m.body_inertia[bid])@mat(m.body_iquat[bid]).T
 massrows.append(dict(body=name,mass_kg=mass,mass_error_kg=float(abs(mass-m.body_mass[bid])),ledger_COM_error_m=float(np.max(abs(c-rec['COM_cart_m']))),engine_local_COM_error_m=float(np.max(abs(m.body_ipos[bid]-rec['COM_body_m']))),fullI_error_kg_m2=float(np.max(abs(I-engineI))),source_count=len(items)))
# Robot compiled contracts remain exact, even with separate free cart added.
attrs=['body_mass','body_ipos','body_inertia','body_iquat','body_pos','body_quat','body_parentid','jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_stiffness','dof_damping','dof_frictionloss','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_trntype','actuator_gainprm','actuator_biasprm','actuator_dynprm','eq_data','eq_type','eq_active0']
unchanged={}
for at in attrs:
 x=getattr(m,at);y=getattr(old,at);unchanged[at]=bool(np.array_equal(x[:len(y)],y))
context=load(E/'goose_cart_context.json'); d.qpos[:old.nq]=context['full_qpos'];mj.mj_forward(m,d)
# Direct actual signed distances at the saved FK, all near-mouth vs all robot exterior.
near=[g for g in cart if m.geom(g).name in ('cart_bite_bar','cart_bite_yoke_right','cart_bite_yoke_left','cart_rotating_shaft','cart_drawbar','cart_pitch_clevis_left','cart_pitch_clevis_right','cart_yaw_housing','cart_mast')]
distances=[]
for g in near:
 for r in robot:
  v=float(mj.mj_geomDistance(m,d,g,r,1,np.zeros(6)))
  distances.append(dict(cart=m.geom(g).name,robot=m.geom(r).name,distance_m=v))
contactrows=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,distance_m=float(c.dist)) for c in d.contact]
# Cavity/tote exact analytic slab bounds, convex half-space for each mesh.
inside=[]
for x in [.025,.325,.625]:
 for y in [-.200,0,.200]:
  for z in [.1815001,.235,.3315]:
   pt=np.array([x+.65,y,z]);hits=[]
   for g in cart:
    local=d.geom_xmat[g].reshape(3,3).T@(pt-d.geom_xpos[g]);typ=m.geom_type[g];sz=m.geom_size[g]
    if typ==mj.mjtGeom.mjGEOM_BOX: hit=np.all(abs(local)<sz-1e-9)
    elif typ==mj.mjtGeom.mjGEOM_CYLINDER:hit=np.linalg.norm(local[:2])<sz[0]-1e-9 and abs(local[2])<sz[1]-1e-9
    elif typ==mj.mjtGeom.mjGEOM_MESH:
     mid=m.geom_dataid[g];vs=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];h=ConvexHull(vs);hit=np.all(h.equations[:,:3]@local+h.equations[:,3]<-1e-9)
    else:hit=False
    if hit:hits.append(m.geom(g).name)
   inside.append(dict(p_cart_m=[x,y,z],hits=hits))
# Input identity and exact cart map coverage.
inputs={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()}; stored=load(L/'analysis/cart_collision_001/source_input_immutability.json')
report=dict(engine=mj.__version__,source_identity=load(A/'source_identity.json'),current_source_hashes={k:sha(Path(v['path'])) for k,v in load(A/'source_identity.json').items()},asset_hashes={str(p.relative_to(A)):sha(p) for p in sorted(A.rglob('*')) if p.is_file()},active_cart_leaves=len(cart),active_robot_leaves=len(robot),ground_geoms=[m.geom(g).name for g in active if not m.geom_bodyid[g]],cart_types=dict(collections.Counter(mj.mjtGeom(int(m.geom_type[g])).name for g in cart)),mesh_graph=meshrows,cart_body_count=len(bmap['body_map']),cart_hinges=[dict(name=m.joint(j).name,axis=m.jnt_axis[j].tolist(),limited=bool(m.jnt_limited[j]),range=m.jnt_range[j].tolist(),stiffness=float(m.jnt_stiffness[j])) for j in range(old.njnt,m.njnt)],cart_MCI_checks=massrows,parts_covered_once=len(used)==122 and len(set(used))==122 and set(used)==set(parts),cart_mass_kg=sum(r['mass_kg'] for r in massrows),compiled_robot_arrays_unchanged=unchanged,robot_actuators=m.nu,baseline_robot_actuators=old.nu,equality_count=m.neq,baseline_equality_count=old.neq,timestep=float(m.opt.timestep),baseline_timestep=float(old.opt.timestep),cart_root_free=bool(m.jnt_type[m.joint('cart_root').id]==mj.mjtJoint.mjJNT_FREE),near_mouth_distances=distances,nearest_unintended_contacts=sorted([r for r in distances if r['cart']!='cart_bite_bar'],key=lambda x:x['distance_m'])[:10],saved_pose_contacts=contactrows,cavity_probes=inside,input_file_hashes=inputs,forward_calls=1,physics_steps=0,training_updates=0,GPU=0,wall_seconds=time.monotonic()-start)
save('independent_compiled_cart.json',report)
print(json.dumps({k:report[k] for k in ['engine','active_cart_leaves','active_robot_leaves','cart_types','mesh_graph','cart_mass_kg','parts_covered_once','compiled_robot_arrays_unchanged','nearest_unintended_contacts','saved_pose_contacts','wall_seconds']},indent=2))
