import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
import json,resource,time
from pathlib import Path
import numpy as np,mujoco as mj
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));start=time.time()
O=Path(__file__).resolve().parent;s=json.loads((O/'selected_420_interface.json').read_text());r=s['selected'];L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation')
m=mj.MjModel.from_xml_path(str(L/'assets/low_profile_stock_cart_01/goose_cart_scene.xml'));d=mj.MjData(m);d.qpos[:29]=s['robot_full_qpos'];d.qpos[int(m.joint('cart_root').qposadr[0])]=r['cart_origin_world_m'][0];d.qpos[int(m.joint('cart_pitch_hinge').qposadr[0])]=r['cart_pitch_rad'];m.body_pos[m.body('cart_yaw').id,2]=r['cart_tow_yaw_pitch_pivot_height_m'];mj.mj_forward(m,d)

def bb(i):
 R=d.geom_xmat[i].reshape(3,3);c=d.geom_xpos[i];t=int(m.geom_type[i]);size=m.geom_size[i]
 if t==7:
  k=int(m.geom_dataid[i]);v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]@R.T+c;return np.array([v.min(0),v.max(0)])
 if t==6:e=abs(R)@size
 elif t==5:e=abs(R[:,2])*size[1]+np.sqrt(np.maximum(0,1-R[:,2]**2))*size[0]
 else:raise ValueError(t)
 return np.array([c-e,c+e])
names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)];cart=[i for i,n in enumerate(names) if n.startswith('cart_') and n!='cart_bite_bar'];robot=[i for i,n in enumerate(names) if not n.startswith('cart_') and n!='ground'];rows=[]
for a in robot:
 for b in cart:
  pts=np.zeros(6);dd=float(mj.mj_geomDistance(m,d,a,b,2,pts));ba=bb(a);bb_=bb(b);gap=np.maximum(np.maximum(ba[0]-bb_[1],bb_[0]-ba[1]),0);lb=float(np.linalg.norm(gap))
  rows.append({'robot':names[a],'cart':names[b],'engine_signed_distance_m':dd,'AABB_separation_lower_bound_m':lb,'proven_separated_by_AABB':lb>1e-8,'engine_zero_but_AABB_separated':dd==0 and lb>1e-8})
contacts=[{'a':names[c.geom1],'b':names[c.geom2],'distance_m':float(c.dist)} for c in d.contact if c.dist<-.0001 and names[c.geom1]!='cart_bite_bar' and names[c.geom2]!='cart_bite_bar' and (names[c.geom1].startswith('cart_') != names[c.geom2].startswith('cart_')) and names[c.geom1]!='ground' and names[c.geom2]!='ground']
soles=[bb(m.geom(n+'_actual_sole').id) for n in ['left','right']];deck=bb(m.geom('cart_deck_frame').id)
out={'status':'FINITE_SAVED_FK_PROXY_ONLY','pairs_checked':len(rows),'pairs':rows,'nonbar_robot_cart_penetrations_over0p1mm':contacts,'exact_engine_zero_but_AABB_separation_count':sum(r['engine_zero_but_AABB_separated'] for r in rows),'zero_distance_scope':'When engine distance0 contradicts strict separated AABBs, do not claim exact0 collision. AABB separation proves no overlap only, exact distance unqualified.','shoe_to_deck_X_plane_separation_m':float(deck[0,0]-max(v[1,0] for v in soles)),'robot_floor_lowest_sole_z_m':float(min(v[0,2] for v in soles)),'geometry_note':'All initial robot14+cart18 compiled proxies. New02 native CAD is independently pending; narrow obstacles, omitted fork contacts and sweeps not checked.','physics_integrations':0,'new_forward_calls':1,'GPU':0,'training_updates':0,'wall_s':time.time()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(O/'selected_420_clearance.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='pairs'})
