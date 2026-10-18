import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
import json,math,time,hashlib,resource
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.optimize import least_squares,linprog
START=time.time();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
W=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
R=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';C=json.loads((R/'physical_contract.json').read_text())
L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation')
M=mj.MjModel.from_xml_path(str(L/'assets/low_profile_stock_cart_01/goose_cart_scene.xml'));D=mj.MjData(M)
source_hash={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
js=C['joints'];act=np.array([int(M.joint(j.get('actuation_joint',j['name'])).dofadr[0]) for j in js]);caps=np.array([j['continuous_design_limit_nm'] for j in js]);loss=M.dof_frictionloss[act]
hb=M.body('head_roll').id;neck=['neck_pitch','neck_mid_pitch','head_pitch'];qi=np.array([int(M.joint(n).qposadr[0]) for n in neck]);nvrobot=28
root=np.zeros(M.nq);mj.mj_resetData(M,D);root[:]=D.qpos
BAR_X=.300;BAR_PITCH=0.
def setq(q,legs,rootxz,bar_z):
 D.qpos[:]=root;D.qpos[int(M.joint('cart_root').qposadr[0])]=BAR_X+.095+.255*math.cos(BAR_PITCH);D.qpos[int(M.joint('cart_pitch_hinge').qposadr[0])]=BAR_PITCH;D.qpos[:3]=[rootxz[0],0,rootxz[1]];D.qpos[3:7]=[1,0,0,0]
 for n,v in legs.items():D.qpos[int(M.joint(n).qposadr[0])]=v
 D.qpos[qi]=q
 D.qpos[int(M.joint('beak_hinge').qposadr[0])]=.19682523003160665
 for p in C['passive_linkage_joints']:
  D.qpos[int(M.joint(p['name']).qposadr[0])]=p['mimic_multiplier']*D.qpos[int(M.joint(p['mimic_joint']).qposadr[0])]+p['mimic_offset_rad']
 # Only in-memory design fixture: all tow hardware moved together to proposed new mounting height.
 M.body_pos[M.body('cart_yaw').id,2]=bar_z-.255*math.sin(BAR_PITCH)
 mj.mj_kinematics(M,D)

# Derive grip frame in head coordinates from actual source zero at ROOT Z=0.
context=json.loads((W/'robots/goose_v0_2/evidence/low_profile_stock_cart_01/goose_cart_context.json').read_text())
D.qpos[:]=root;D.qpos[:29]=context['full_qpos'];mj.mj_kinematics(M,D)
LOCAL=D.xmat[hb].reshape(3,3).T@(np.array([.300,0,.480])-D.xpos[hb])
def tcp():return D.xpos[hb]+D.xmat[hb].reshape(3,3)@LOCAL
def bounds(gname):
 i=M.geom(gname).id;mid=M.geom_dataid[i];v=M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid]+M.mesh_vertnum[mid]];p=v@D.geom_xmat[i].reshape(3,3).T+D.geom_xpos[i];return p.min(0),p.max(0)

stances={}
for name,leg in [('upright',{}),('moderate_crouch',dict(right_hip_pitch=-.2,right_knee_pitch=.5,right_ankle_pitch=-.3,left_hip_pitch=.2,left_knee_pitch=-.5,left_ankle_pitch=.3))]:
 setq(np.zeros(3),leg,[0,0],.480);mj.mj_fwdPosition(M,D)
 bo=[bounds(n+'_actual_sole') for n in ['left','right']]
 rootz=-min(p[0][2] for p in bo)
 # Preserve actual foot XY positions of the upright reference when changing leg posture.
 centers=np.array([D.xpos[M.body(n+'_ankle_roll').id] for n in ['left','right']])
 if name=='upright':refcenters=centers.copy()
 rootx=float(np.mean(refcenters[:,0]-centers[:,0]))
 stances[name]={'legs':leg,'root_xz_m':[rootx,rootz]}

def equilibrium(tcp_point,F):
 jp=np.zeros((3,M.nv));jr=jp.copy();mj.mj_jac(M,D,jp,jr,tcp_point,hb)
 nvr=int(M.jnt_dofadr[M.joint('cart_root').id]);assert nvr==28
 A=np.zeros((nvr,18));A[act,np.arange(18)]=1
 eq=(np.asarray(D.efc_type)==int(mj.mjtConstraint.mjCNSTR_EQUALITY))
 E=np.asarray(D.efc_J).reshape(D.nefc,M.nv)[eq,:nvr].T
 columns=[];pts=[]
 for side,sy in [('left',1),('right',-1)]:
  bid=M.body(side+'_ankle_roll').id;y=sy*.125
  # Same six-pad finite support polygon as the existing static screen, adjusted only by actual stance FK translation.
  for x,yy in [(-.0299,y),(.0061,y-.036),(.09053355,y-.036),(.12653355,y),(.09053355,y+.036),(.0061,y+.036)]:
   point=np.array([x,yy,0]);J=np.zeros((3,M.nv));rot=J.copy();mj.mj_jac(M,D,J,rot,point,bid);columns.append(J[:,:nvr].T);pts.append(point)
 K=np.hstack(columns);ne=E.shape[1];N=18+36+ne+1;AA=np.c_[A,K,E,np.zeros(nvr)];ub=[];uv=[]
 for i in range(18):
  for sign in [1,-1]:
   r=np.zeros(N);r[i]=sign;r[-1]=-caps[i];ub.append(r);uv.append(-loss[i])
 for k in range(12):
  for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
   r=np.zeros(N);r[18+3*k]=sx;r[18+3*k+1]=sy;r[18+3*k+2]=-.65;ub.append(r);uv.append(0.)
 bnds=[(None,None)]*18+[(None,None),(None,None),(0,None)]*12+[(None,None)]*ne+[(0,None)]
 b=D.qfrc_bias[:nvr]-jp[:,:nvr].T@F;obj=np.zeros(N);obj[-1]=1
 r=linprog(obj,A_ub=np.array(ub),b_ub=np.array(uv),A_eq=AA,b_eq=b,bounds=bnds,method='highs')
 neckdof=np.array([int(M.joint(j['name']).dofadr[0]) for j in js[:5]])
 necktau=(D.qfrc_bias-jp.T@F)[neckdof];necku=(np.abs(necktau)+loss[:5])/caps[:5]
 out={'F_world_N':F.tolist(),'upper_chain_gravity_Nm':D.qfrc_bias[neckdof].tolist(),'upper_chain_external_JTF_Nm':(jp.T@F)[neckdof].tolist(),'upper_required_static_Nm':necktau.tolist(),'upper_chain_utilisation':necku.tolist(),'upper_chain_max_util':float(max(necku)),
 'external_pitch_moment_about_ground_Nm':float(tcp_point[2]*F[0]-tcp_point[0]*F[2]),'LP_success':r.success}
 if r.success:
  forces=r.x[18:54].reshape(12,3);u=(abs(r.x[:18])+loss)/caps;cop=(forces[:,2,None]*np.array(pts)[:,:2]).sum(0)/forces[:,2].sum()
  out.update(whole_min_max_util=float(r.x[-1]),worst_axis=js[int(u.argmax())]['name'],actual_static_tau18=r.x[:18].tolist(),foot_force_left_N=forces[:6].sum(0).tolist(),foot_force_right_N=forces[6:].sum(0).tolist(),cop_world_xy_m=cop.tolist(),support_fore_aft_margin_m=float(min(cop[0]+.0299,.12653355-cop[0])),equilibrium_residual_max=float(abs(AA@r.x-b).max()))
 return out

results=[];boundsq=np.array([M.joint(n).range.copy() for n in neck]);seeds=[np.array([.464,.736,-1.201]),np.array([1.1,.3,-1.2])]
for stance,s in stances.items():
 for height in [.480,.440,.420,.400,.380,.340]:
  for xpos,angle in [(x,a) for x in [.22,.26,.30] for a in [0,math.radians(10),math.radians(15),math.radians(30)]]:
   BAR_X=xpos;BAR_PITCH=angle
   def fun(q):
    setq(q,s['legs'],s['root_xz_m'],height);return np.r_[tcp()[[0,2]]-[xpos,height],.15*(sum(q)-angle)]
   fits=[least_squares(fun,np.clip(seed,boundsq[:,0]+1e-8,boundsq[:,1]-1e-8),bounds=(boundsq[:,0],boundsq[:,1]),max_nfev=100,xtol=1e-10,ftol=1e-10,gtol=1e-10) for seed in seeds]
   fit=min(fits,key=lambda r:np.linalg.norm(r.fun));setq(fit.x,s['legs'],s['root_xz_m'],height);mj.mj_forward(M,D);point=tcp().copy();error=float(np.linalg.norm(point[[0,2]]-[xpos,height]));pitch_error=float(sum(fit.x)-angle)
   row={'stance':stance,'target_x_m':xpos,'target_height_m':height,'target_pitch_deg':math.degrees(angle),'q_neck_mid_head_rad':fit.x.tolist(),'tcp_world_m':point.tolist(),'root_xz_m':s['root_xz_m'],'leg_pose_rad':s['legs'],'cart_pitch_rad':angle,'cart_origin_world_m':[float(D.qpos[int(M.joint('cart_root').qposadr[0])]),0,0],'cart_tow_yaw_pitch_pivot_height_m':height-.255*math.sin(angle),'cart_pitch_within_limit':angle<=math.radians(25),'pos_error_m':error,'pitch_error_rad':pitch_error,'reachable':error<.0001 and abs(pitch_error)<.001,'actual_head_pitch_deg':math.degrees(sum(fit.x)),
    'self_penetrations':[],'cart_nonbar_penetrations':[],'qpos':D.qpos[:29].tolist()}
   for con in D.contact:
    a=mj.mj_id2name(M,mj.mjtObj.mjOBJ_GEOM,con.geom1);b=mj.mj_id2name(M,mj.mjtObj.mjOBJ_GEOM,con.geom2)
    if con.dist<-.0001 and not a.startswith('cart_') and not b.startswith('cart_') and a!='ground' and b!='ground':row['self_penetrations'].append({'a':a,'b':b,'distance_m':float(con.dist)})
    if con.dist<-.0001 and a!='cart_bite_bar' and b!='cart_bite_bar' and (a.startswith('cart_')!=b.startswith('cart_')) and a!='ground' and b!='ground':row['cart_nonbar_penetrations'].append({'a':a,'b':b,'distance_m':float(con.dist)})
   row['floor_extra_contacts']=[{'a':mj.mj_id2name(M,mj.mjtObj.mjOBJ_GEOM,x.geom1),'b':mj.mj_id2name(M,mj.mjtObj.mjOBJ_GEOM,x.geom2),'distance_m':float(x.dist)} for x in D.contact if x.dist<-.0001 and (x.geom1==M.geom('ground').id or x.geom2==M.geom('ground').id)]
   if row['reachable']:
    row['force_cases']=[equilibrium(point,np.array(F,float)) for F in [[0,0,0],[10,0,0],[20,0,0],[20,0,-3],[20,3,0]]]
    row['neck_limit_margin_rad']=np.minimum(fit.x-boundsq[:,0],boundsq[:,1]-fit.x).tolist()
    row['robot_COM_world_m']=D.subtree_com[M.body('torso').id].tolist()
    pairs=[]
    for a in ['upper_bill_envelope','lower_bill_envelope','torso_upper_envelope','right_shin_envelope','left_shin_envelope','right_actual_sole','left_actual_sole']:
     for b in ['cart_deck_frame','cart_mast','cart_rotating_shaft','cart_bite_yoke_left','cart_bite_yoke_right']:
      pts=np.zeros(6);distance=mj.mj_geomDistance(M,D,M.geom(a).id,M.geom(b).id,2,pts);pairs.append({'a':a,'b':b,'distance_m':float(distance)})
    row['finite_clearance_minima']=sorted(pairs,key=lambda x:x['distance_m'])[:6]
    row['bar_upper_lower_distance_m']={a:float(mj.mj_geomDistance(M,D,M.geom(a).id,M.geom('cart_bite_bar').id,2,np.zeros(6))) for a in ['upper_bill_envelope','lower_bill_envelope']}
   results.append(row)
   print('CASE',stance,xpos,height,round(math.degrees(angle)),row['reachable'],'errmm',round(error*1000,3),'self',len(row['self_penetrations']),'cart',len(row['cart_nonbar_penetrations']), 'u20', None if not row.get('force_cases') else row['force_cases'][2].get('whole_min_max_util'),flush=True)

out={'status':'FINITE_STATIC_HEIGHT_COMPARISON_NOT_CONTROL_OR_DRAG_TEST','source_robot_SHA256':source_hash,'source_cart_scene_sha256':hashlib.sha256((L/'assets/low_profile_stock_cart_01/goose_cart_scene.xml').read_bytes()).hexdigest(),'source_robot_unchanged':source_hash=={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()},'tcp_head_local_m':LOCAL.tolist(),'torque_sign':'tau + foot_JT reactions + equality_JT reactions = gravity_bias - mouth_JT external_force; +Fx is load pulling robot toward cart','stances':stances,'results':results,'physics_integrations':0,'training_updates':0,'GPU':0,'wall_s':time.time()-START,'limitations':['Fixed upright torso and one moderate crouch with unchanged exact foot XY and horizontal foot orientation. No global posture optimum.','Minmax LP is ideal static force allocation with existing friction pyramid mu.65 and twelve finite support points; not an implemented controller.','This is force transmitted at grip reference on head frame. No proven clamp/pressure/friction force or jaw motor load allocation.','Cart tow mounting height changed only in-memory for finite clearance query. Existing cart collision002 and CAD files never changed.','Pitch spring/drawbar vertical self-weight balance unknown. Vertical -3N is sensitivity only.','All torques are zero-speed continuous nominal limits; no speed/thermal/endurance proof.']}
(O/'results.json').write_text(json.dumps(out,indent=2)+'\n')
