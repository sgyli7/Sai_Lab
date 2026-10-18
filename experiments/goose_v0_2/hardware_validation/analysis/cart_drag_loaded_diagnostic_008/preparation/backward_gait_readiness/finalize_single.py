import os,sys,time,resource,json,hashlib,itertools
sys.dont_write_bytecode=True
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.optimize import least_squares,linprog
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
O=Path(__file__).parent;started=time.monotonic();P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04');C=json.loads((P/'physical_contract.json').read_text());m=mj.MjModel.from_xml_path(str(P/'robot.xml'));d=mj.MjData(m);names=C['joint_order'];qidx=np.array([int(m.joint(n).qposadr[0]) for n in names]);vidx=np.array([int(m.joint(n).dofadr[0]) for n in names]);driveidx=np.array([int(m.joint(j.get('actuation_joint',j['name'])).dofadr[0]) for j in C['joints']]);cont=np.array([j['continuous_design_limit_nm'] for j in C['joints']]);cold=np.array(json.loads(Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/scene_preflight.json').read_text())['cold_qpos'])[:m.nq];d.qpos[:]=cold;mj.mj_forward(m,d);head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674]);feetgeom={s:m.geom(s+'_actual_sole').id for s in ['left','right']};footref={s:(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for s,g in feetgeom.items()};counts={'poses':0,'IK_solves':0,'IK_nfev':0,'FK_calls':0,'LP_solves':0,'forward_calls':1};rows=[]
def ikleg(side,pos,rot,seed):
 ns=[side+'_'+j for j in ['hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll']];ix=np.array([int(m.joint(n).qposadr[0]) for n in ns]);bounds=np.array([m.jnt_range[m.joint(n).id] for n in ns]);g=feetgeom[side]
 def f(q):
  counts['FK_calls']+=1;d.qpos[ix]=q;mj.mj_kinematics(m,d);return np.r_[d.geom_xpos[g]-pos,.08*Rotation.from_matrix(rot.T@d.geom_xmat[g].reshape(3,3)).as_rotvec()]
 sol=least_squares(f,np.clip(seed,bounds[:,0]+1e-6,bounds[:,1]-1e-6),bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=40,gtol=1e-9,ftol=1e-9,xtol=1e-9);counts['IK_solves']+=1;counts['IK_nfev']+=sol.nfev;d.qpos[ix]=sol.x;return float(np.linalg.norm(f(sol.x)))
def neck(target):
 ix=qidx[:5];bb=np.array([m.jnt_range[m.joint(n).id] for n in names[:5]])
 def f(q):
  counts['FK_calls']+=1;d.qpos[ix]=q;mj.mj_kinematics(m,d);H=d.xmat[head].reshape(3,3);e=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[d.xpos[head]+H@local-target,.08*(e[1]-np.deg2rad(10)),.08*e[2]]
 sol=least_squares(f,np.clip(cold[ix],bb[:,0]+.10001,bb[:,1]-.10001),bounds=(bb[:,0]+.1,bb[:,1]-.1),max_nfev=50,gtol=1e-9,ftol=1e-9,xtol=1e-9);counts['IK_solves']+=1;counts['IK_nfev']+=sol.nfev;d.qpos[ix]=sol.x;return float(np.linalg.norm(f(sol.x)))
def solepoints(side):
 g=feetgeom[side];mesh=int(m.geom_dataid[g]);v=m.mesh_vert[int(m.mesh_vertadr[mesh]):int(m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh])];world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];low=world[world[:,2]<world[:,2].min()+1e-5];xy=np.unique(np.round(low[:,:2],9),axis=0)
 if len(xy)>2:xy=xy[ConvexHull(xy).vertices]
 return [np.r_[p,world[:,2].min()] for p in xy]
def static(support,tow):
 counts['LP_solves']+=1;counts['forward_calls']+=1;mj.mj_forward(m,d);points=[(s,p) for s in support for p in solepoints(s)];eqids=np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))[0];Je=d.efc_J.reshape(d.nefc,m.nv)[eqids];nf=len(points)*3;na=m.nu;ne=len(eqids);B=np.zeros((m.nv,na))
 for i in range(na):B[driveidx[i],i]=m.actuator_gear[i,0]
 J=[]
 for s,p in points:
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,p,int(m.geom_bodyid[feetgeom[s]]));J.append(jp.T)
 A=np.hstack([*J,B,Je.T]);b=d.qfrc_bias.copy();jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);tcp=d.xpos[head]+d.xmat[head].reshape(3,3)@local;mj.mj_jac(m,d,jp,jr,tcp,head);b-=jp.T@np.array([tow,0,0]);Aub=[];bub=[]
 for k in range(len(points)):
  for th in np.arange(8)*np.pi/4:
   row=np.zeros(A.shape[1]);row[3*k]=np.cos(th);row[3*k+1]=np.sin(th);row[3*k+2]=-.65*np.cos(np.pi/8);Aub.append(row);bub.append(0)
 bounds=[z for p in points for z in [(None,None),(None,None),(0,None)]]+[(-float(c),float(c)) for c in cont]+[(None,None)]*ne
 A=np.c_[A,np.zeros(m.nv)];Aub=[np.r_[r,0.] for r in Aub]
 for i,c in enumerate(cont):
  for sg in [-1,1]:
   r=np.zeros(A.shape[1]);r[nf+i]=sg;r[-1]=-c;Aub.append(r);bub.append(0.)
 bounds.append((0,1));obj=np.zeros(A.shape[1]);obj[-1]=1;rr=linprog(obj,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=A,b_eq=b,bounds=bounds,method='highs',options={'time_limit':1.})
 return {'tow_N':tow,'success':bool(rr.success),'message':rr.message,'util':None if not rr.success else float(rr.x[-1]),'tau18':None if not rr.success else rr.x[nf:nf+na].tolist(),'contact_forces':None if not rr.success else rr.x[:nf].reshape(-1,3).tolist(),'sole_support_points_world_m':[p.tolist() for _,p in points],'residual':None if not rr.success else float(np.max(abs(A@rr.x-b)))}
def pose(label,bx,by,bz,roll,offsets,support,checkforces=(8,),feet_yaw=0.):
 counts['poses']+=1;d.qpos[:]=cold;d.qpos[:3]+=np.array([bx,by,bz]);qq=Rotation.from_euler('x',roll).as_quat();d.qpos[3:7]=np.r_[qq[3],qq[:3]];errs={}
 for s in ['left','right']:
  pos,rot=footref[s];target=pos+np.array(offsets.get(s,[0,0,0]));rot=Rotation.from_euler('z',-feet_yaw if s=='left' else feet_yaw).as_matrix()@rot;seed=[0,0,.2,-.4,.2,0] if s=='left' else [0,0,-.2,.4,-.2,0];errs[s]=ikleg(s,target,rot,seed)
 # Mouth follows cart root backward in X, stays centered on cart in Y. No new gripping geometry.
 
 for side in ['left','right']:d.qpos[int(m.joint(side+'_yaw_input_rotor').qposadr[0])]=d.qpos[int(m.joint(side+'_hip_yaw').qposadr[0])]
 ne=neck(np.array([.30+bx,0,.420]));counts['forward_calls']+=1;mj.mj_forward(m,d);bad=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact if c.dist<-.0005];feasible=max(errs.values())<.0005 and ne<.001 and not bad;lp=[static(support,F) for F in checkforces] if feasible else [];r={'name':label,'root_xyz_reference_only':d.qpos[:3].tolist(),'root_roll':roll,'offsets':offsets,'support':support,'leg_IK_errors_m':errs,'neck_IK_equiv_error_m':ne,'bad_contacts':bad,'geometric_ok':feasible,'static':lp,'qpos_reference_only':d.qpos.tolist(),'q18_target':d.qpos[qidx].tolist(),'TCP_world':(d.xpos[head]+d.xmat[head].reshape(3,3)@local).tolist(),'COM_world':d.subtree_com[m.body('torso').id].tolist(),'feet_world':{s:{'position':d.geom_xpos[g].tolist(),'rotation':d.geom_xmat[g].reshape(3,3).tolist(),'sole_z_min':min(p[2] for p in solepoints(s))} for s,g in feetgeom.items()}};rows.append(r);print('POSE',label,'err',errs,'neck',ne,'bad',[(x['a'],x['b']) for x in bad],'LP',[(x['tow_N'],x['success'],x['util']) for x in lp],flush=True);return r



def diagnostic(row):
 if not row['static']:return
 pts=np.array(row['static'][0]['sole_support_points_world_m']);cop=np.array(row['COM_world'][:2]);cop[0]+=8*.420/(sum(m.body_mass)*9.81);h=ConvexHull(pts[:,:2]);ineq=h.equations[:,:2]@cop+h.equations[:,2];row['required_static_CoP_world_xy_m']=cop.tolist();row['CoP_hull_max_violation_m']=float(max(ineq));row['static_failure_classification']='STATIC_SUPPORT_POLYGON_VIOLATION' if max(ineq)>1e-6 else 'CAPACITY_OR_INTERNAL_EQUILIBRIUM_PENDING';row['original_caps_Nm']=cont.tolist()
 if not row['static'][0]['success'] and max(ineq)<=1e-6:
  d.qpos[:]=row['qpos_reference_only'];old=cont.copy();cont[:]*=10;rr=static(row['support'],8);cont[:]=old;row['diagnostic_relaxed10x_caps_NOT_EXECUTABLE']=rr;row['static_failure_classification']='ORIGINAL_CONTINUOUS_CAPS_INSUFFICIENT' if rr['success'] else 'INTERNAL_EQUILIBRIUM_OR_GEOMETRY_PENDING'

a=json.loads((O/'stage2.json').read_text());old=next(r for r in a['rows'] if r['name']=='yaw_single_12');d.qpos[:]=old['qpos_reference_only']
for side in ['left','right']:d.qpos[int(m.joint(side+'_yaw_input_rotor').qposadr[0])]=d.qpos[int(m.joint(side+'_hip_yaw').qposadr[0])]
mj.mj_forward(m,d);eq=np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))[0];r={'source_stage':'stage2/yaw_single_12','status':'GEOMETRIC_SINGLE_STANCE_REFERENCE_ONLY_STATIC_FAIL','qpos_reference_only':d.qpos.tolist(),'q18_target':d.qpos[qidx].tolist(),'root_xyz_reference_only':d.qpos[:3].tolist(),'COM_world':d.subtree_com[m.body('torso').id].tolist(),'support':['left'],'equality_residual_max_m_or_rad':float(max(abs(d.efc_pos[eq]))),'bad_contacts':[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist_m':float(c.dist)} for c in d.contact if c.dist<-.0005],'static':[static(['left'],8)],'feet_world':{side:{'position':d.geom_xpos[g].tolist(),'rotation':d.geom_xmat[g].reshape(3,3).tolist()} for side,g in feetgeom.items()}}
diagnostic(r);points=np.array(r['static'][0]['sole_support_points_world_m']);r['necessary_lateral_LIP_acceleration_to_reach_inner_support_edge_m_s2']=float(9.81*(r['COM_world'][1]-points[:,1].min())/r['COM_world'][2]);r['acceleration_qualification']='Necessary simplified COM/ZMP estimate only; excludes multibody angular momentum and does not grant dynamic torque feasibility.';r['static0N']=static(['left'],0);r['static20N']=static(['left'],20)
(O/'best_single_stance_reference.json').write_text(json.dumps(r,indent=2));print('BEST_SINGLE',r['COM_world'],r['CoP_hull_max_violation_m'],r['necessary_lateral_LIP_acceleration_to_reach_inner_support_edge_m_s2'],r['bad_contacts'],'eq',r['equality_residual_max_m_or_rad'],flush=True)
# Adjacent double-support touchdown keeps both physical feet on the actual plane.
b=pose('adjacent_double_to_best_single',-.02,.04,-.025,0.,{'right':[0,.03,0]},['left','right'],checkforces=(0,8,20),feet_yaw=.25)
(O/'adjacent_double_support_reference.json').write_text(json.dumps(b,indent=2));(O/'final_diagnostic_counts.json').write_text(json.dumps(counts,indent=2))
