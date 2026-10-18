"""Read saved cart003: compile/FK only; no integration, optimization, parameter edits."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import mujoco as mj
L=Path(__file__).resolve().parents[2];W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');A=L/'assets/low_profile_stock_cart_02';O=Path(__file__).parent;E=W/'robots/goose_v0_2/evidence/low_profile_stock_cart_02';R=L/'inputs/physics_v0_2_20261018_04'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
def par(m,d):return m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
params=load(W/'robots/goose_v0_2/cad/source/low_profile_stock_cart_02/parameters.json');sel=load(W/'artifacts/goose_v0_2/low_handle_study_01/selected_420_interface.json');native=load(E/'geometry_mass_screen.json');parts={q['name']:q for q in native['parts']}
m=mj.MjModel.from_xml_path(str(A/'goose_cart_scene.xml'));old=mj.MjModel.from_xml_path(str(R/'robot.xml'));d=mj.MjData(m);d.qpos[:29]=params['robot_full_qpos'];ci=int(m.joint('cart_root').qposadr[0]);origin=np.array(params['cart_origin_robot_frame_mm'])*.001;d.qpos[ci:ci+3]=origin;d.qpos[ci+3:ci+7]=[1,0,0,0];d.qpos[int(m.joint('cart_pitch_hinge').qposadr[0])]=np.deg2rad(params['drawhead_pitch_deg']);mj.mj_forward(m,d)
attrs=['body_mass','body_ipos','body_inertia','body_iquat','body_pos','body_quat','body_parentid','jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_stiffness','dof_damping','dof_frictionloss','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_trntype','actuator_gainprm','actuator_biasprm','actuator_dynprm','eq_data','eq_type','eq_active0'];unchanged={at:bool(np.array_equal(getattr(m,at)[:len(getattr(old,at))],getattr(old,at))) for at in attrs}
cart=[g for g in range(m.ngeom) if m.geom(g).name.startswith('cart_') and (m.geom_contype[g] or m.geom_conaffinity[g])];robot=[g for g in range(m.ngeom) if g not in cart and m.geom_bodyid[g] and (m.geom_contype[g] or m.geom_conaffinity[g])];graph=[]
for g in cart:
 if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
  mid=m.geom_dataid[g];a=m.mesh_graphadr[mid];nv,nf=map(int,m.mesh_graph[a:a+2]);graph.append(dict(geom=m.geom(g).name,actual_supports=nv,actual_hull_faces=nf))
bmap=load(A/'body_mass_map.json');used=[];bodyrows=[]
for name,rec in bmap['body_map'].items():
 used+=rec['source_parts'];rows=[(parts[n]['mass_kg'],np.array(parts[n]['COM_mm'])*.001,np.array(parts[n]['full_inertia_about_COM_kg_m2'])) for n in rec['source_parts']]
 if rec.get('reserve_kg',0):
  size=np.array([.650,.450,.030]);I=.35/12*np.diag([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2]);rows.append((.35,np.array([.260,0,.180]),I))
 mass=sum(q[0] for q in rows);c=sum(mm*cc for mm,cc,II in rows)/mass;I=sum(II+par(mm,cc-c) for mm,cc,II in rows);bid=m.body(name).id;rot=d.ximat[bid].reshape(3,3);actualI=rot@np.diag(m.body_inertia[bid])@rot.T
 bodyrows.append(dict(body=name,mass_kg=mass,mass_error=float(abs(mass-m.body_mass[bid])),COM_cart_m=c.tolist(),engine_COM_cart_m=(d.xipos[bid]-origin).tolist(),COM_error_m=float(np.max(abs(c-(d.xipos[bid]-origin)))),fullI_error_kg_m2=float(np.max(abs(I-actualI)))))
bar=m.geom('cart_bite_bar').id;barcenter=d.geom_xpos[bar];hc=d.xpos[m.body('head_roll').id]+d.xmat[m.body('head_roll').id].reshape(3,3)@np.array(sel['point_definition']['tcp_head_local_m']);nativecenter=origin+np.array(parts['replaceable_bite_sleeve_d12']['COM_mm'])*.001
near=[g for g in cart if m.geom(g).name in ('cart_bite_bar','cart_rotating_shaft','cart_bite_yoke_left','cart_bite_yoke_right','cart_drawbar','cart_mast')];dist=[]
for g in near:
 for r in robot:
  p=np.zeros(6);v=float(mj.mj_geomDistance(m,d,g,r,2,p));dist.append(dict(cart=m.geom(g).name,robot=m.geom(r).name,distance_m=v,closest_world_m=p.tolist()))
foot=[]
for name in ('left_actual_sole','right_actual_sole'):
 g=m.geom(name).id;mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];foot.append(dict(geom=name,min_world_m=v.min(0).tolist(),max_world_m=v.max(0).tolist()))
limits=[]
for j in range(m.njnt):
 if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
  q=float(d.qpos[m.jnt_qposadr[j]]);lo,hi=m.jnt_range[j];limits.append(dict(joint=m.joint(j).name,q=q,range=[float(lo),float(hi)],margin_rad=float(min(q-lo,hi-q))))
contacts=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,distance_m=float(c.dist)) for c in d.contact];negative=[c for c in contacts if c['distance_m']<-.0001 and not ('cart_bite_bar' in (c['a'],c['b']))]
out=dict(candidate='low_profile_stock_cart_02_collision_003',engine=mj.__version__,runtime_identity={str(p.relative_to(A)):sha(p) for p in sorted(A.rglob('*')) if p.is_file()},active_cart_leaves=len(cart),active_robot_leaves=len(robot),cart_types=dict(collections.Counter(mj.mjtGeom(int(m.geom_type[g])).name for g in cart)),actual_mesh_graph=graph,body_MCI=bodyrows,parts_once=len(used)==122 and len(set(used))==122,cart_mass_kg=sum(x['mass_kg'] for x in bodyrows),robot_compiled_arrays_unchanged=unchanged,robot_actuators=m.nu,baseline_actuators=old.nu,equalities=m.neq,baseline_equalities=old.neq,dt=float(m.opt.timestep),barcenter_world_m=barcenter.tolist(),native_barcenter_world_m=nativecenter.tolist(),head_material_center_world_m=hc.tolist(),native_bar_center_error_m=float(np.linalg.norm(barcenter-nativecenter)),head_material_center_error_m=float(np.linalg.norm(barcenter-hc)),near_mouth_distances=dist,minimum_unintended_distance=sorted([r for r in dist if r['cart']!='cart_bite_bar'],key=lambda x:x['distance_m'])[:10],feet=foot,shoe_platform_plane_gap_m=float(origin[0]-max(x['max_world_m'][0] for x in foot)),limits=limits,negative_nonbar_contacts_beyond_0p1mm=negative,saved_pose_contacts=contacts,forward_calls=1,physics_steps=0,training=0,GPU=0)
(O/'independent_compiled_cart003.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('active_cart_leaves','active_robot_leaves','cart_types','actual_mesh_graph','cart_mass_kg','barcenter_world_m','native_barcenter_world_m','head_material_center_world_m','native_bar_center_error_m','head_material_center_error_m','shoe_platform_plane_gap_m','minimum_unintended_distance','negative_nonbar_contacts_beyond_0p1mm')},indent=2))
