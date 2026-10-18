"""Offline FK/IK path admission only. No dynamics integration."""
from pathlib import Path
import os, json, time, resource
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
ROOT=Path(__file__).resolve().parent;PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip());START=time.time()
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
assert mj.__version__=='3.10.0'
C=json.loads((PKG/'physical_contract.json').read_text());m=mj.MjModel.from_xml_path(str(PKG/'robot.xml'));d=mj.MjData(m);d.qpos[:]=m.qpos0
head=m.body('head_roll').id
def fk():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
fk();upper=np.array(C['grip_reference_zero_m'])+np.array([0,0,m.qpos0[2]]);tip_local=d.xmat[head].reshape(3,3).T@(upper-d.xpos[head])+np.array([0,0,-.006])
necknames=['neck_pitch','neck_mid_pitch','head_pitch'];ni=np.array([int(m.joint(n).qposadr[0]) for n in necknames]);bounds=np.array([m.joint(n).range[0] for n in necknames])
# model views give range arrays with two values
bounds=np.array([m.jnt_range[m.joint(n).id] for n in necknames])
def tip():
    fk();H=d.xmat[head].reshape(3,3);return d.xpos[head]+H@tip_local,float(np.arctan2(H[0,2],H[0,0]))
solutions=[]
for x in [.38,.34,.30,.26,.22]:
    target=np.array([x,0.,.15005]);best=None
    for start in [[1.6,-.6,-1.0],[2.2,-1.,-1.2],[1.,.5,-1.5]]:
        def fun(q):
            d.qpos[ni]=q;p,a=tip();return np.r_[(p-target)[[0,2]],.1*a]
        sol=least_squares(fun,np.clip(start,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0]+1e-8,bounds[:,1]-1e-8),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=150)
        err=float(np.linalg.norm(fun(sol.x)))
        if best is None or err<best[0]:best=(err,sol.x.copy())
    d.qpos[ni]=best[1];d.qpos[int(m.joint('beak_hinge').qposadr[0])]=.19682523003160665
    for j in C['passive_linkage_joints']:d.qpos[int(m.joint(j['name']).qposadr[0])]=j['mimic_multiplier']*d.qpos[int(m.joint(j['mimic_joint']).qposadr[0])]+j['mimic_offset_rad']
    mj.mj_forward(m,d)
    bad=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'distance_m':float(c.dist)} for c in d.contact if c.dist<-.0001]
    soles=[]
    for name in ['right_actual_sole','left_actual_sole']:
        g=m.geom(name).id;mid=int(m.geom_dataid[g]);vs=m.mesh_vert[int(m.mesh_vertadr[mid]):int(m.mesh_vertadr[mid]+m.mesh_vertnum[mid])];world=vs@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        soles.append({'geom':name,'bounds_min_world_m':world.min(0).tolist(),'bounds_max_world_m':world.max(0).tolist(),'position_world_m':d.geom_xpos[g].tolist(),'rotation_world':d.geom_xmat[g].reshape(3,3).tolist()})
    solutions.append({'handle_target_m':target.tolist(),'neck_named_rad':dict(zip(necknames,best[1].tolist())),'IK_error_m_equivalent':best[0],'actual_handle_m':tip()[0].tolist(),'head_pitch_world_rad':tip()[1],'negative_contacts':bad,'robot_COM_world_m':d.subtree_com[m.body('torso').id].tolist(),'sole_geometry':soles})
valid=[x for x in solutions if x['IK_error_m_equivalent']<1e-6 and not x['negative_contacts']]
print('EVENT static neck candidates',json.dumps(solutions,separators=(',',':')),flush=True)
assert valid,'No collision-admitted lowhead solution'
# Prefer the shortest forward reach that preserves a 60mm neck retraction path.
selected=valid[0];d.qpos[:]=m.qpos0;d.qpos[ni]=list(selected['neck_named_rad'].values());startq=d.qpos.copy()
path=[];q=best[1].copy()
q=np.array(list(selected['neck_named_rad'].values()))
for u in np.linspace(0,1,31):
    target=np.array(selected['handle_target_m'])+np.array([-.06*u,0.,0.])
    def fun(z):
        d.qpos[ni]=z;p,a=tip();return np.r_[(p-target)[[0,2]],.1*a]
    sol=least_squares(fun,q,bounds=(bounds[:,0]+1e-8,bounds[:,1]-1e-8),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=70);q=sol.x
    path.append({'u':float(u),'neck_rad':q.tolist(),'error_m_equiv':float(np.linalg.norm(fun(q)))})
assert max(x['error_m_equiv'] for x in path)<1e-5
# Short double-stance lean path: desired root positions are FK planning variables only.
# No runtime root writes: only ordinary joint requests will be sent to the frozen drive.
d.qpos[:]=startq;fk();feet={n:(d.geom_xpos[m.geom(n).id].copy(),d.geom_xmat[m.geom(n).id].reshape(3,3).copy()) for n in ['right_actual_sole','left_actual_sole']}
legpaths=[]
for u in np.linspace(0,1,31):
    d.qpos[:]=startq;d.qpos[0]-=.06*u;row={};errors=[]
    for side in ['right','left']:
        names=[side+'_'+j for j in ['hip_pitch','knee_pitch','ankle_pitch']];idx=np.array([int(m.joint(n).qposadr[0]) for n in names]);bb=np.array([m.jnt_range[m.joint(n).id] for n in names]);g=m.geom(side+'_actual_sole').id;pos,R0=feet[side+'_actual_sole']
        def fleg(z):
            d.qpos[idx]=z;fk();R=d.geom_xmat[g].reshape(3,3);dp=d.geom_xpos[g]-pos;ang=np.arctan2((R@R0.T)[0,2],(R@R0.T)[0,0]);return np.r_[dp[[0,2]],.1*ang]
        guess=np.array([-.1,.2,-.1]) if side=='right' else np.array([.1,-.2,.1])
        sol=least_squares(fleg,guess,bounds=(bb[:,0]+1e-8,bb[:,1]-1e-8),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100);row.update(dict(zip(names,sol.x.tolist())));errors.append(float(np.linalg.norm(fleg(sol.x))))
    legpaths.append({'u':float(u),'named_rad':row,'IK_errors_m_equiv':errors})
out={'engine':mj.__version__,'physics_integrations':0,'tip_point_head_local_m':tip_local.tolist(),'neck_axes':necknames,'candidates':solutions,'selected':selected,'neck_pull_path_60mm':path,'free_base_short_double_stance_joint_path':legpaths,'free_base_path_admitted_kinematically':max(max(x['IK_errors_m_equiv']) for x in legpaths)<1e-5,'limitations':'Static reach/contact admission only. No claim of dynamic equilibrium or gait. Root shifts in this file are private offline FK planning queries, never online root writes.'}
(ROOT/'kinematic_plan.json').write_text(json.dumps(out,indent=2)+'\n')
r=resource.getrusage(resource.RUSAGE_SELF);(ROOT/'resource_exit_prepare.json').write_text(json.dumps({'physics_integrations':0,'CPU_affinity':sorted(os.sched_getaffinity(0)),'maxRSS_KiB':r.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'wall_s':time.time()-START,'GPU':0,'training':0},indent=2)+'\n')
print('STATUS static plan ready',selected['handle_target_m'],selected['neck_named_rad'],'leg_path',out['free_base_path_admitted_kinematically'],flush=True)
