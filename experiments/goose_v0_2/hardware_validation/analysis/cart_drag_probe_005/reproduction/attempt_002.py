# Archived numerical source with I/O-only relocation; not executed by integrator.
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,time,json,hashlib,sys,importlib.util,xml.etree.ElementTree as E,gzip,resource
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
L=Path(__file__).resolve().parents[3]
R=Path(os.environ['GOOSE_V02_FRESH_OUTPUT']).expanduser().resolve()
if R.exists() or R.is_relative_to(L) or R.is_relative_to(Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_005')):
 raise RuntimeError('Require a new output directory outside the archive and original005 run')
R.mkdir(parents=True,exist_ok=False)
W=L;P=L/'inputs/physics_v0_2_20261018_04';A=L/'assets/low_profile_stock_cart_03'
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)))
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.rglob('*')) if p.is_file()}
H=hashes();dump(R/'source_before.json',H)
spec=importlib.util.spec_from_file_location('adapter',P/'runtime_adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad)
X=E.parse(A/'goose_cart_scene.xml');root=X.getroot();root.find('compiler').set('meshdir',str(A))
for mesh in root.findall('./asset/mesh'):
 mesh.set('file',str((A/mesh.get('file')).resolve()))
root.find('./worldbody/geom[@name="ground"]').set('conaffinity', E.parse(P/'robot.xml').getroot().find('./worldbody/geom[@name="ground"]').get('conaffinity'))
E.SubElement(root.find('equality'),'weld',name='stationary_local_torso_fixture',body1='world',body2='torso',relpose='0 0 0.005000002805734566 1 0 0 0',solref='.005 1',solimp='.99 .9999 .0001')
b=E.SubElement(root.find('worldbody'),'body',name='probe_cargo',pos='.9711259770181133 0 .2565');E.SubElement(b,'freejoint',name='probe_cargo_free');E.SubElement(b,'inertial',pos='0 0 0',mass='20',diaginertia='0.3041666666666667 0.6375 0.8666666666666667');E.SubElement(b,'geom',name='probe_cargo_box',type='box',size='.3 .2 .075',density='0',contype='8',conaffinity='7',friction='.65 .01 .002',condim='3',solref='.005 1',rgba='.6 .35 .15 1')
for j in root.findall('.//joint'):
 if (j.get('name') or '').startswith('cart_wheel_spin'):j.set('frictionloss','.125')
key=root.find('./keyframe/key');key.set('qpos',key.get('qpos')+' .9711259770181133 0 .2565 1 0 0 0')
S=R/'scene';S.mkdir(exist_ok=True);X.write(S/'robot.xml');
for name in ['physical_contract.json','actuator_contract.json']:(S/name).write_bytes((P/name).read_bytes())
env=ad.FrozenPhysics(S);m,d=env.model,env.data;C=json.loads((P/'physical_contract.json').read_text());base=mj.MjModel.from_xml_path(str(P/'robot.xml'))
# Exact named immutable physical robot fields; scene extensions only.
checks=0
for typ,names,fields in [('body',[base.body(i).name for i in range(1,base.nbody)],['body_mass','body_ipos','body_iquat','body_inertia','body_pos','body_quat']),('joint',[base.joint(i).name for i in range(base.njnt)],['jnt_axis','jnt_pos','jnt_range','jnt_type']),('geom',[base.geom(i).name for i in range(base.ngeom)],['geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_solref']),('actuator',[base.actuator(i).name for i in range(base.nu)],['actuator_gear','actuator_ctrlrange','actuator_forcerange'])]:
 for n in names:
  ia=getattr(base,typ)(n).id;ib=getattr(m,typ)(n).id
  for f in fields:assert np.array_equal(getattr(base,f)[ia],getattr(m,f)[ib]),(typ,n,f);checks+=1
mj.mj_resetDataKeyframe(m,d,0);env.target=d.qpos[env.qidx].copy();mj.mj_forward(m,d)
head=m.body('head_roll').id;rod=m.geom('cart_bite_bar').id;cart=m.body('cart_base').id;cargo=m.body('probe_cargo').id;torso=m.body('torso').id;jaws=[m.geom('upper_bill_envelope').id,m.geom('lower_bill_envelope').id]
local=d.xmat[head].reshape(3,3).T@(d.geom_xpos[rod]-d.xpos[head]);pitch0=np.sum(d.qpos[[int(m.joint(n).qposadr[0]) for n in ['neck_yaw','neck_pitch','neck_mid_pitch','head_pitch','head_roll']]])
ni=np.array([env.names.index(n) for n in ['neck_yaw','neck_pitch','neck_mid_pitch','head_pitch','head_roll']]);qi=env.qidx[ni];lo=m.jnt_range[[m.joint(n).id for n in ['neck_yaw','neck_pitch','neck_mid_pitch','head_pitch','head_roll']],0];hi=m.jnt_range[[m.joint(n).id for n in ['neck_yaw','neck_pitch','neck_mid_pitch','head_pitch','head_roll']],1];kd=mj.MjData(m)
relative_initial=d.xmat[m.body('cart_roll').id].reshape(3,3).T@d.xmat[head].reshape(3,3)
fixed_orientation=None
def ik(target,seed):
 kd.qpos[:]=d.qpos
 desired_R=(d.xmat[m.body('cart_roll').id].reshape(3,3)@relative_initial) if fixed_orientation is None else fixed_orientation
 def f(q):
  kd.qpos[qi]=q;mj.mj_kinematics(m,kd);Rh=kd.xmat[head].reshape(3,3);p=kd.xpos[head]+Rh@local
  return np.r_[p-target,.04*Rotation.from_matrix(desired_R.T@Rh).as_rotvec()]
 sol=least_squares(f,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo+1e-9,hi-1e-9),max_nfev=35,ftol=1e-8,xtol=1e-8,gtol=1e-8);return sol.x,float(np.linalg.norm(f(sol.x)))
# Declared cold start: neck 15mm behind actual bar; jaw12mm cold angle, no preload.
q,err=ik(d.geom_xpos[rod].copy(),d.qpos[qi]);d.qpos[qi]=q;env.target=d.qpos[env.qidx].copy();mj.mj_forward(m,d)
contacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist':float(c.dist),'mu':c.friction.tolist()} for c in d.contact];bad=[c for c in contacts if c['dist']<-.0005]
admission={'checks_exact':checks,'initial_ik_error_m_equivalent':err,'initial_contacts':contacts,'bad_penetrations':bad,'scope':'LOCAL_FIXED_TORSO_HEAD_NECK_ACTUATED','wheel_frictionloss_Nm':.125,'equivalent_resistance_N':8,'free_cargo_mass_kg':20,'robot_mass_kg':sum(base.body_mass),'rod_head_local':local,'source_files':len(H),'physics_steps':0};dump(R/'preflight.json',admission);print('EVENT PREFLIGHT',json.dumps(admission,default=lambda a:a.tolist()),flush=True)
if bad:sys.exit(2)
# All qpos edits above are initialization only. Original adapter handles every dynamic step.
np.savez(R/'cold_state.npz',qpos=d.qpos,qvel=d.qvel);mj.mj_saveModel(m,str(R/'actual_scene.mjb'))
qtraj=[d.qpos.copy()];vtraj=[d.qvel.copy()];ts=[0.];rows=[];source_step=mj.mj_step;initcart=d.xpos[cart].copy();initcargo=d.xpos[cargo].copy();pulltarget=None;gripref=None;gripR=None;lost=None;ever=False;failure=None;phase='open';log=gzip.open(R/'substeps.jsonl.gz','wt',compresslevel=2)
def step(model,data,nstep=1):
 global ever,lost,failure,gripref,gripR
 assert nstep==1 and not data.xfrc_applied.any() and not data.qfrc_applied.any()
 incoming=data.qvel[env.drive_vidx].copy();expected=env.limit_torque(env.kp*(env.target-data.qpos[env.qidx])-env.kd*data.qvel[env.vidx],incoming);assert np.max(abs(expected-data.ctrl))<1e-10
 source_step(model,data,nstep=1);mj.mj_kinematics(m,d);mj.mj_comPos(m,d);normal=np.zeros(2);tow=np.zeros(3);cs=[];feet=np.zeros(2);penetration=0
 for ci,c in enumerate(d.contact):
  f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);a,b=int(c.geom1),int(c.geom2);rec={'a':m.geom(a).name,'b':m.geom(b).name,'dist_m':float(c.dist),'friction':c.friction.tolist(),'force_contact':f.tolist(),'pos':c.pos.tolist()};cs.append(rec)
  if rod in (a,b) and (a in jaws or b in jaws):
   k=jaws.index(b if a==rod else a);normal[k]+=max(f[0],0);tow+=(1 if b==rod else -1)*c.frame.reshape(3,3).T@f[:3];penetration=max(penetration,-c.dist)
  if 'ground' in (m.geom(a).name,m.geom(b).name):
   other=m.geom(b if m.geom(a).name=='ground' else a).name
   if other in ['right_actual_sole','left_actual_sole']:feet[['right_actual_sole','left_actual_sole'].index(other)]+=max(f[0],0)
 Hm=d.xmat[head].reshape(3,3);rel=Hm.T@(d.geom_xpos[rod]-d.xpos[head]);relR=Hm.T@d.xmat[m.body('cart_roll').id].reshape(3,3);both=min(normal)>.01
 if phase in ['close','pull','hold'] and both:
  ever=True;lost=None
  if gripref is None:gripref=rel.copy();gripR=relR.copy()
 elif ever and phase in ['pull','hold'] and lost is None:lost=float(d.time)
 closure=[]
 for ei in range(m.neq):
  ids=np.where((d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id[:d.nefc]==ei))[0];closure.append({'name':m.equality(ei).name,'residual':d.efc_pos[ids].tolist(),'force':d.efc_force[ids].tolist()})
 row={'time_s':float(d.time),'phase':phase,'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'target18':env.target.tolist(),'torque18':d.actuator_force.tolist(),'drive_speed18_incoming':incoming.tolist(),'positive_power_W':float(np.maximum(d.actuator_force*incoming,0).sum()),'cart_pose':np.r_[d.xpos[cart],d.xquat[cart]].tolist(),'cargo_pose':np.r_[d.xpos[cargo],d.xquat[cargo]].tolist(),'root_pose':d.qpos[:7].tolist(),'rod_world':d.geom_xpos[rod].tolist(),'rod_rel_head':rel.tolist(),'rod_rel_R':relR.tolist(),'slip_from_capture_m':None if gripref is None else float(np.linalg.norm(rel-gripref)),'angle_from_capture_rad':None if gripR is None else float(np.arccos(np.clip((np.trace(gripR.T@relR)-1)/2,-1,1))),'jaw_normal_N':normal.tolist(),'tow_on_rod_world_N':tow.tolist(),'cart_back_m':float(initcart[0]-d.xpos[cart,0]),'feet_normal_N':feet.tolist(),'jaw_pen_m':float(penetration),'equalities':closure,'contacts':cs,'warning_counts':[int(w.number) for w in d.warning]};log.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row);qtraj.append(d.qpos.copy());vtraj.append(d.qvel.copy());ts.append(float(d.time))
 if lost is not None and d.time-lost>.3:failure='bilateral_contact_lost_over300ms'
 if phase=='hold' and not ever:failure='no_bilateral_capture'
 if not np.isfinite(d.qpos).all():failure='nonfinite'
 if max(row['warning_counts']):failure='mujoco_warning'
 if row['positive_power_W']>350+1e-7:failure='power_limit_violation'
 if int(round(d.time/.005))%200==0:print('EVENT',round(d.time,2),phase,'normal',normal.round(2).tolist(),'cart_back_mm',round(row['cart_back_m']*1000,2),'slip',row['slip_from_capture_m'],flush=True)
mj.mj_step=step;error=None;lastq=d.qpos[qi].copy()
try:
 for tick in range(650):
  if time.time()>1792331224:raise TimeoutError('stop_new_experiments deadline')
  t=float(d.time);desired=d.qpos[env.qidx].copy();desired[6:]=env.neutral[6:]
  if t<.6:phase='open';target=d.geom_xpos[rod]+[0,0,.012*min(t/.6,1)];jaw=.42
  elif t<1.8:phase='approach';target=d.geom_xpos[rod]+[0,0,.012*(1-(t-.6)/1.2)];jaw=.42
  elif t<3.0:phase='close';target=d.geom_xpos[rod].copy();jaw=0.
  elif t<5.5:
   phase='pull';jaw=0.
   if pulltarget is None:pulltarget=d.geom_xpos[rod].copy();fixed_orientation=d.xmat[m.body('cart_roll').id].reshape(3,3)@relative_initial
   u=np.clip((t-3)/2.5,0,1);target=pulltarget+[-.05*u*u*(3-2*u),0,0]
  elif t<10.7:phase='hold';target=pulltarget+[-.05,0,0];jaw=0.
  else:phase='release';target=pulltarget+[-.05,0,0];jaw=.42
  lastq,err=ik(target,lastq);desired[ni]=lastq;desired[5]=jaw;env.step((desired-env.neutral)/env.scale)
  if failure:break
except BaseException as ex:error=repr(ex)
finally:
 mj.mj_step=source_step;log.close();np.savez_compressed(R/'trajectory.npz',qpos=np.array(qtraj),qvel=np.array(vtraj),time=np.array(ts));after=hashes();dump(R/'source_after.json',after);assert H==after
 summary={'dynamic_attempts':1,'steps':len(rows),'sim_duration_s':float(d.time),'failure':failure,'error':error,'bilateral_ever':ever,'final_cart_back_m':rows[-1]['cart_back_m'] if rows else None,'max_slip_m':max([r['slip_from_capture_m'] or 0 for r in rows],default=0),'max_positive_power_W':max([r['positive_power_W'] for r in rows],default=0),'phase_durations_s':{p:sum(r['phase']==p for r in rows)*.005 for p in set(r['phase'] for r in rows)},'source_unchanged':H==after,'scope':'LOCAL_FIXED_TORSO_BENCH','NN':0,'GPU':0,'training':0,'maxRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};dump(R/'summary.json',summary);print('STATUS',json.dumps(summary),flush=True)
