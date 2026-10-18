from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
import json,hashlib,math,time,resource
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));start=time.time()
O=Path(__file__).resolve().parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation');R=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';scene=L/'assets/low_profile_stock_cart_03/goose_cart_scene.xml';m=mj.MjModel.from_xml_path(str(scene));d=mj.MjData(m);k=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_kinematics(m,d);qzero=d.qpos.copy();hb=m.body('head_roll').id;rod=m.geom('cart_bite_bar').id;rc=d.geom_xpos[rod].copy();local=d.xmat[hb].reshape(3,3).T@(rc-d.xpos[hb]);qi=np.array([int(m.joint(n).qposadr[0]) for n in ['neck_pitch','neck_mid_pitch','head_pitch']]);bounds=np.array([m.joint(n).range.copy() for n in ['neck_pitch','neck_mid_pitch','head_pitch']]);jaw=int(m.joint('beak_hinge').qposadr[0]);cr=int(m.joint('cart_root').qposadr[0]);C=json.loads((R/'physical_contract.json').read_text());names=[m.geom(i).name for i in range(m.ngeom)];cart=[i for i,n in enumerate(names) if n.startswith('cart_')];robot=[i for i,n in enumerate(names) if n!='ground' and not n.startswith('cart_')];bill=[m.geom(n).id for n in ['upper_bill_envelope','lower_bill_envelope']]
inputs=[scene,R/'robot.xml',R/'physical_contract.json',O.parent/'attempt_003/declaration.json',O.parent/'run_gravity_target.py'];source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
def pose(data,q,targetjaw,cartdx=0):
 data.qpos[:]=qzero;data.qpos[qi]=q;data.qpos[jaw]=targetjaw;data.qpos[cr]+=cartdx
 for j in C['passive_linkage_joints']:data.qpos[int(m.joint(j['name']).qposadr[0])]=j['mimic_multiplier']*data.qpos[int(m.joint(j['mimic_joint']).qposadr[0])]+j['mimic_offset_rad']
 mj.mj_kinematics(m,data)
def bb(i):
 R=d.geom_xmat[i].reshape(3,3);c=d.geom_xpos[i];sz=m.geom_size[i];typ=int(m.geom_type[i])
 if typ==7:
  mi=int(m.geom_dataid[i]);v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]@R.T+c;return v.min(0),v.max(0)
 e=abs(R)@sz if typ==6 else abs(R[:,2])*sz[1]+np.sqrt(np.maximum(0,1-R[:,2]**2))*sz[0]
 return c-e,c+e

def path(kind):
 if kind=='sloped6to2_then_vertical_pitch10':
  pre=rc+[-.03,0,.006];p=[('open',pre,.28+(i/1)*.01,0,False) for i in range(2)]
  p += [('approach_horizontal',pre+[.03*u,0,-.004*u],.29,0,False) for u in np.linspace(0,1,12)]
  p += [('seat_vertical',rc+[0,0,.002*(1-u)],.29,0,u==1) for u in np.linspace(0,1,10)]
  nclose=10;npull=16;phi=math.radians(10)
 else:
  h=.030 if kind=='vertical30_pitch10' else .006;phi=math.radians(10) if kind=='vertical30_pitch10' else .4;pre=rc+[0,0,h];p=[('open',pre,.28+i*.01,0,False) for i in range(2)];p += [('approach_vertical',rc+[0,0,h*(1-u)],.29,0,u==1) for u in np.linspace(0,1,16)];nclose=10;npull=22
 p += [('close',rc,.29+(.19682523003160665-.29)*u,0,True) for u in np.linspace(0,1,nclose)]
 p += [('ideal_loaded_pull',rc+[-.05*u,0,0],.19682523003160665,-.05*u,True) for u in np.linspace(0,1,npull)]
 assert len(p)==50;return phi,p
allrows=[];summary=[]
for kind in ['sloped6to2_then_vertical_pitch10','vertical30_pitch10','original006_pitch0p4_height6']:
 phi,seq=path(kind);seed=qzero[qi].copy();rows=[]
 for index,(phase,target,j,dx,allowed) in enumerate(seq):
  def cost(q):
   pose(k,q,j,dx);tcp=k.xpos[hb]+k.xmat[hb].reshape(3,3)@local;return np.r_[(tcp-target)[[0,2]],.08*(sum(q)-phi)]
  fit=least_squares(cost,np.clip(seed,bounds[:,0]+.10001,bounds[:,1]-.10001),bounds=(bounds[:,0]+.1,bounds[:,1]-.1),max_nfev=60,ftol=1e-10,xtol=1e-10,gtol=1e-10);seed=fit.x;pose(d,seed,j,dx);mj.mj_collision(m,d)
  err=float(np.linalg.norm(cost(seed)));margin=float(np.min(np.minimum(seed-bounds[:,0],bounds[:,1]-seed)));distances=[];zeros=[];boundsall={i:bb(i) for i in cart+robot}
  for a in robot:
   for b in cart:
    if b==rod and a in bill and allowed:continue
    ft=np.zeros(6);dist=float(mj.mj_geomDistance(m,d,a,b,2,ft));ba=boundsall[a];bc=boundsall[b];gap=np.maximum(np.maximum(ba[0]-bc[1],bc[0]-ba[1]),0);lb=float(np.linalg.norm(gap))
    if dist==0 and lb>1e-8:dist=lb;zeros.append([names[a],names[b]])
    distances.append((dist,names[a],names[b]))
  negatives=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact if c.dist<-.000001 and ((c.geom1 in cart and c.geom2 in robot) or (c.geom2 in cart and c.geom1 in robot)) and not(allowed and rod in [c.geom1,c.geom2] and (c.geom1 in bill or c.geom2 in bill))]
  selfbad=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact if c.dist<-.000001 and c.geom1 in robot and c.geom2 in robot]
  row={'path':kind,'index':index,'phase':phase,'target_tcp_world_m':np.array(target).tolist(),'q_neck_mid_head_rad':seed.tolist(),'jaw_rad':j,'cart_dx_m':dx,'IK_equivalent_error_m':err,'min_neck_margin_rad':margin,'minimum_non_target_distance':sorted(distances)[0],'non_target_negative_contacts':negatives,'self_negative_contacts':selfbad,'allowed_expected_rod_contact':allowed,'body_qpos':d.qpos.tolist(),'AABB_resolved_falsezero_count':len(zeros)};rows.append(row)
 worst=min(rows,key=lambda r:r['minimum_non_target_distance'][0]);clean=all(r['IK_equivalent_error_m']<1e-5 and r['min_neck_margin_rad']>=.099999 and not r['non_target_negative_contacts'] and not r['self_negative_contacts'] and r['minimum_non_target_distance'][0]>=-1e-6 for r in rows)
 summary.append({'path':kind,'samples':len(rows),'clean_finite_path':clean,'max_IK_equivalent_error_m':max(r['IK_equivalent_error_m'] for r in rows),'minimum_neck_margin_rad':min(r['min_neck_margin_rad'] for r in rows),'worst_non_target':worst,'fail_samples':[r['index'] for r in rows if r['non_target_negative_contacts'] or r['self_negative_contacts']],'minimum_distance_by_phase':{p:min(r['minimum_non_target_distance'][0] for r in rows if r['phase']==p) for p in sorted(set(r['phase'] for r in rows))}});allrows+=rows;print(kind,'clean',clean,'worst',worst['minimum_non_target_distance'],'phase',worst['phase'],'margin',summary[-1]['minimum_neck_margin_rad'],flush=True)
out={'scope':'READ_ONLY_DETERMINISTIC_FINITE_IK_FK_DISTANCE_SCREEN','paths':summary,'sample_count':len(allrows),'inputs':source,'cold_rod_derived_head_local_tcp_m':local.tolist(),'all_inputs_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in source.items()),'physics_integrations':0,'training_updates':0,'LP_solves':0,'GPU':0,'wall_s':time.time()-start,'limitations':['Fifty samples per predeclared strategy, not continuous swept-volume certificate.','Close segment allows only expected upper/lower contact with true rod. No force, contact holding or friction inference.','Postcapture50mm pull moves only offline planned cart pose with TCP; no dynamic object repositioning or actual simulation.','Other robot/cart active contact and unfiltered non-target distance checked; source self-collision filters retained.','Original006 static planner path reconstructed at unchanged design cart pose; adaptive dynamic rod drift from006 is not replayed.']};(O/'summary.json').write_text(json.dumps(out,indent=2,default=lambda v:v.item() if hasattr(v,"item") else v)+'\n');(O/'sampled_paths.json').write_text(json.dumps(allrows,indent=2,default=lambda v:v.item() if hasattr(v,"item") else v)+'\n')
