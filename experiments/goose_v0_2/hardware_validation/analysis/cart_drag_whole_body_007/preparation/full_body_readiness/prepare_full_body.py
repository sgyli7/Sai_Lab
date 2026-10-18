import os,sys,time
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
from pathlib import Path
import json,hashlib
import numpy as np,mujoco as mj
from scipy.optimize import least_squares,linprog
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
R=Path(__file__).parent;P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04');C=json.loads((P/'physical_contract.json').read_text());m=mj.MjModel.from_xml_path(str(P/'robot.xml'));d=mj.MjData(m);names=C['joint_order'];qidx=np.array([int(m.joint(n).qposadr[0]) for n in names]);vidx=np.array([int(m.joint(n).dofadr[0]) for n in names]);driveidx=np.array([int(m.joint(j.get('actuation_joint',j['name'])).dofadr[0]) for j in C['joints']]);cont=np.array([j['continuous_design_limit_nm'] for j in C['joints']]);cold=np.array(json.loads((R.parent/'planning_gate_attempts_001_002.json').read_text())['cold_qpos'])[:m.nq];cold[qidx[:6]]=[0,.677692881,.812988221,-1.316148177,0,.19682523];d.qpos[:]=cold;mj.mj_forward(m,d);head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674]);footref={side:(d.geom_xpos[m.geom(side+'_actual_sole').id].copy(),d.geom_xmat[m.geom(side+'_actual_sole').id].reshape(3,3).copy()) for side in ['left','right']};feetgeom={side:m.geom(side+'_actual_sole').id for side in ['left','right']};rows=[]
# Root poses are OFFLINE scratch planning variables only, never an online command channel.
def ikleg(side,targetpos,targetR,seed):
 ns=[side+'_'+j for j in ['hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll']];ix=np.array([int(m.joint(n).qposadr[0]) for n in ns]);bounds=np.array([m.jnt_range[m.joint(n).id] for n in ns]);g=feetgeom[side]
 def f(q):
  d.qpos[ix]=q;mj.mj_kinematics(m,d);rot=d.geom_xmat[g].reshape(3,3);return np.r_[d.geom_xpos[g]-targetpos,.08*Rotation.from_matrix(targetR.T@rot).as_rotvec()]
 sol=least_squares(f,np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0]+1e-8,bounds[:,1]-1e-8),max_nfev=100,gtol=1e-10,ftol=1e-10,xtol=1e-10);d.qpos[ix]=sol.x;return dict(zip(ns,sol.x.tolist())),float(np.linalg.norm(f(sol.x)))
def neck(target,pitch=.174532925):
 ix=qidx[:5];bb=np.array([m.jnt_range[m.joint(n).id] for n in names[:5]])
 def f(q):
  d.qpos[ix]=q;mj.mj_kinematics(m,d);H=d.xmat[head].reshape(3,3);e=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[d.xpos[head]+H@local-target,.08*(e[1]-pitch),.08*e[2]]
 sol=least_squares(f,np.clip(cold[ix],bb[:,0]+.10001,bb[:,1]-.10001),bounds=(bb[:,0]+.1,bb[:,1]-.1),max_nfev=120,gtol=1e-10,ftol=1e-10,xtol=1e-10);d.qpos[ix]=sol.x;return float(np.linalg.norm(f(sol.x)))
def solepoints(side):
 g=feetgeom[side];mesh=int(m.geom_dataid[g]);v=m.mesh_vert[int(m.mesh_vertadr[mesh]):int(m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh])];world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];low=world[world[:,2]<world[:,2].min()+1e-5]
 xy=np.unique(np.round(low[:,:2],9),axis=0)
 if len(xy)>2:xy=xy[ConvexHull(xy).vertices]
 return [np.r_[p,world[:,2].min()] for p in xy]
def static(support,tow):
 mj.mj_forward(m,d);points=[(side,p) for side in support for p in solepoints(side)];eqids=np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))[0];Je=d.efc_J.reshape(d.nefc,m.nv)[eqids];nf=len(points)*3;na=m.nu;ne=len(eqids);B=np.zeros((m.nv,na))
 for i in range(na):B[driveidx[i],i]=m.actuator_gear[i,0]
 J=[]
 for side,p in points:
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,p,int(m.geom_bodyid[feetgeom[side]]));J.append(jp.T)
 A=np.hstack([*J,B,Je.T]);b=d.qfrc_bias.copy();jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);tcp=d.xpos[head]+d.xmat[head].reshape(3,3)@local;mj.mj_jac(m,d,jp,jr,tcp,head);b-=jp.T@np.array([tow,0,0])
 # robot experiences +X cart reaction during backward towing; no vertical mouth support credit.
 Aub=[];bub=[]
 for k in range(len(points)):
  for th in np.arange(8)*np.pi/4:
   row=np.zeros(A.shape[1]);row[3*k]=np.cos(th);row[3*k+1]=np.sin(th);row[3*k+2]=-.65*np.cos(np.pi/8);Aub.append(row);bub.append(0)
 bounds=[]
 for k in points:bounds.extend([(None,None),(None,None),(0,None)])
 bounds += [(-float(x),float(x)) for x in cont]+[(None,None)]*ne
 A=np.c_[A,np.zeros(m.nv)];Aub=[np.r_[row,0.] for row in Aub]
 for i,cap in enumerate(cont):
  for sign in [-1,1]:
   row=np.zeros(A.shape[1]);row[nf+i]=sign;row[-1]=-cap;Aub.append(row);bub.append(0.)
 bounds.append((0,1));obj=np.zeros(A.shape[1]);obj[-1]=1.;result=linprog(obj,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=A,b_eq=b,bounds=bounds,method='highs',options={'time_limit':3.})
 return {'tow_reaction_world_X_N':tow,'support':support,'point_count':len(points),'solver_success':result.success,'solver_message':result.message,'static_only_no_integration':True,'contact_points_world_m':[p.tolist() for _,p in points],'COM_world_m':d.subtree_com[m.body('torso').id].tolist(),'max_force_balance_residual':None if not result.success else float(np.max(abs(A@result.x-b))),'tau18_Nm':None if not result.success else result.x[nf:nf+na].tolist(),'max_continuous_utilization':None if not result.success else float(np.max(abs(result.x[nf:nf+na])/cont)),'contact_forces_world_N':None if not result.success else result.x[:nf].reshape(-1,3).tolist()}
configs=[('double_neutral',0,0,0,0,0,['left','right']),('double_crouch',0,0,-.02,0,0,['left','right']),('double_back_020',-.02,0,-.02,0,0,['left','right']),('double_back_040',-.04,0,-.02,0,0,['left','right']),('double_back_060',-.06,0,-.02,0,0,['left','right']),('left_load_shift',-.02,.10,-.02,0,0,['left','right']),('left_single_support',-.02,.12,-.02,0,.015,['left']),('right_back_050_landing',-.02,.12,-.02,-.05,0,['left','right']),('double_back_100_unstepped',-.10,0,-.02,0,0,['left','right'])]
configs=[tuple(x)+(0.,) for x in configs]+[('left_shift_lean',-.02,.08,-.03,0,0,['left','right'],-.15),('left_support_right_lift',-.02,.08,-.03,0,.015,['left'],-.15),('right_back_050_with_lean',-.02,.08,-.03,-.05,0,['left','right'],-.15)]
for label,bx,by,bz,rightdx,rightdz,support,rootroll in configs:
 d.qpos[:]=cold;d.qpos[:3]+=np.array([bx,by,bz]);qq=Rotation.from_euler('x',rootroll).as_quat();d.qpos[3:7]=np.r_[qq[3],qq[:3]];named={};errors={}
 for side in ['left','right']:
  pos,rot=footref[side];pos=pos.copy()
  if side=='right':pos+=np.array([rightdx,0,rightdz])
  guess=[0,0,-.15,.3,-.15,0] if side=='right' else [0,0,.15,-.3,.15,0];q,e=ikleg(side,pos,rot,guess);named.update(q);errors[side]=e
 # mouth follows corresponding backward-moving cart at420mm; only head/neck interface, C owns collision-free approach.
 neckerr=neck(np.array([.3+bx,0,.42]));mj.mj_forward(m,d);bad=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist':float(c.dist)} for c in d.contact if c.dist<-.0005];lp=[static(support,F) for F in [0,8,20]];row={'name':label,'offline_root_xyz':d.qpos[:3].tolist(),'named_targets18':dict(zip(names,d.qpos[qidx].tolist())),'leg_IK_errors':errors,'neck_IK_equiv_error':neckerr,'actual_TCP_world':(d.xpos[head]+d.xmat[head].reshape(3,3)@local).tolist(),'robot_contacts_deeper0p5mm':bad,'static_checks':lp,'qpos_offline_only':d.qpos.tolist()};rows.append(row);print('POSE',label,errors,'neck',neckerr,'static',[x['solver_success'] for x in lp],flush=True)
out={'physics_steps':0,'GPU':0,'training':0,'model_sha256':hashlib.sha256((P/'robot.xml').read_bytes()).hexdigest(),'rows':rows,'warnings':'Static contact LP with actual sole lowest vertices, original actuator caps, internal linkage constraints, no passive-friction credit. Not dynamic balance/stability. Lateral root shift neck yaw not solved in this first pitch-only interface, so lateral poses do not certify mouth tracking.'};(R/'keyframes_static_checks.json').write_text(json.dumps(out,indent=2));np.savez(R/'offline_keyframe_qpos.npz',qpos=np.array([r['qpos_offline_only'] for r in rows]));print('DONE_STATIC_ONLY',flush=True)
