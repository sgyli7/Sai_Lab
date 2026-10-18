import os,sys,time,resource,signal,datetime
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
from pathlib import Path
import json,gzip,hashlib,importlib.util,xml.etree.ElementTree as E
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
R=Path(__file__).parent;num=int(sys.argv[1]);negative=len(sys.argv)>2 and sys.argv[2]=='open';S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False);W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');P=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';plan=json.loads((R/'planning_gate.json').read_text());sel=plan['selected'];SC=S/'scene';SC.mkdir()
for n in ['robot.xml','physical_contract.json','actuator_contract.json']:(SC/n).write_bytes((R/'scene'/n).read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,a):p.write_text(json.dumps(a,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
sources={str(p):sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in str(p)};dump(S/'source_before.json',sources)
spec=importlib.util.spec_from_file_location('frozen',P/'runtime_adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad);env=ad.FrozenPhysics(SC);m,d=env.model,env.data;C=env.contract;ni=np.array([env.names.index(n) for n in plan['axes']]);qi=env.qidx[ni];lo=env.ranges[ni,0];hi=env.ranges[ni,1];head=m.body('head_roll').id;lower=m.body('beak_hinge').id;rod=m.geom('cart_bite_bar').id;cart=m.body('cart_base').id;cargo=m.body('probe_cargo').id;pitchdof=int(m.joint('cart_pitch_hinge').dofadr[0]);local=np.array(plan['cold_tcp_head']);kd=mj.MjData(m)
mj.mj_resetDataKeyframe(m,d,0);d.qpos[qi]=sel['pre_q'];d.qpos[env.qidx[5]]=.28
for j in C['passive_linkage_joints']:d.qpos[int(m.joint(j['name']).qposadr[0])]=j['mimic_multiplier']*d.qpos[int(m.joint(j['mimic_joint']).qposadr[0])]+j['mimic_offset_rad']
env.target=d.qpos[env.qidx].copy();mj.mj_forward(m,d);initial=d.qpos.copy();initialcart=d.xpos[cart].copy();pretarget=d.geom_xpos[rod].copy()+sel['off'];yaw0=d.qpos[env.qidx[0]]
initc=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'dist':float(c.dist)} for c in d.contact];bad=[c for c in initc if c['dist']<-.0005];mouthcart=[c for c in initc if ('cart_' in c['a'] and 'bill' in c['b']) or ('cart_' in c['b'] and 'bill' in c['a'])];assert not bad and not mouthcart,(bad,mouthcart)
assert d.qfrc_bias[pitchdof]*d.qfrc_passive[pitchdof]>0
meta={'attempt':num,'negative_open_jaw':negative,'selected_plan':sel,'initial_contacts':initc,'initial_mouth_cart_contacts':mouthcart,'spring_initial_bias_Nm':float(d.qfrc_bias[pitchdof]),'spring_initial_passive_Nm':float(d.qfrc_passive[pitchdof]),'scope':'LOCAL_FIXED_TORSO','source_count':len(sources),'control':'DECLARED GRAVITY POSITION TARGET BIAS(qfrc_bias/kp), 3 pitch IK at selected fixed world pitch; original yaw/roll neutral; actual rod translation followed before pull; closed-jaw target force feedback20N mean; original immutable drive','jaw_open_target':.29,'jaw_cold_angle':.28};dump(S/'declaration.json',meta);np.savez(S/'cold_state.npz',qpos=d.qpos,qvel=d.qvel);mj.mj_saveModel(m,str(S/'actual_scene.mjb'));print('EVENT admitted',num,flush=True)
# Independent planning data only; no online qpos writes to real state.
def ik(target,seed):
 kd.qpos[:]=d.qpos
 def f(q):
  kd.qpos[qi]=q;mj.mj_kinematics(m,kd);return np.r_[(kd.xpos[head]+kd.xmat[head].reshape(3,3)@local-target)[[0,2]],.08*(sum(q)-sel['pitch'])]
 sol=least_squares(f,np.clip(seed,lo+.10001,hi-.10001),bounds=(lo+.10,hi-.10),max_nfev=25,ftol=1e-8,xtol=1e-8,gtol=1e-8);return sol.x,float(np.linalg.norm(f(sol.x)))
qtraj=[d.qpos.copy()];vtraj=[d.qvel.copy()];times=[0.];records=[];commands=[];phase='open';lastq=d.qpos[qi].copy();jaw=.29;normal=np.zeros(2);jaws=[m.geom('upper_bill_envelope').id,m.geom('lower_bill_envelope').id];pulltarget=None;capture=None;failure=None;warn=None;ever=False;log=gzip.open(S/'substeps.jsonl.gz','wt',compresslevel=2);source_step=mj.mj_step;threads=0;limits_margin=100.;maxerror=0.;maxpower=0.;first_contact=None
# For every real step, run forward at the exact incoming state/control, copy solver contact+constraint values, then integrate.
def logged(model,data,nstep=1):
 global normal,capture,ever,failure,threads,limits_margin,maxpower,first_contact
 assert nstep==1 and not d.xfrc_applied.any() and not d.qfrc_applied.any()
 preq=d.qpos.copy();prev=d.qvel.copy();pret=float(d.time);mj.mj_forward(m,d)
 bodies={n:{'xpos':d.xpos[m.body(n).id].tolist(),'xmat':d.xmat[m.body(n).id].tolist(),'xquat':d.xquat[m.body(n).id].tolist()} for n in ['head_roll','beak_hinge','cart_base','cart_roll','probe_cargo','torso']};normal=np.zeros(2);tow=np.zeros(3);cs=[];groundwheel=np.zeros(3)
 for ci,c in enumerate(d.contact):
  f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);a,b=int(c.geom1),int(c.geom2);F=c.frame.reshape(3,3).T@f[:3];cs.append({'geom1':m.geom(a).name,'geom2':m.geom(b).name,'pos_world_m':c.pos.tolist(),'frame_rows_world':c.frame.tolist(),'force_contact_6D':f.tolist(),'force_on_geom2_world_N':F.tolist(),'distance_m':float(c.dist),'friction5':c.friction.tolist(),'dim':int(c.dim),'efc_address':int(c.efc_address)})
  if rod in (a,b) and (a in jaws or b in jaws):
   normal[jaws.index(b if a==rod else a)]+=max(f[0],0);tow+=(1 if b==rod else -1)*F
   if f[0]>.01 and first_contact is None:first_contact={'time':pret,'phase':phase}
  if m.geom(a).name=='ground' and m.geom(b).name.startswith('cart_wheel_'):groundwheel+=F
 eq=[]
 for ei in range(m.neq):
  ids=np.where((d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id[:d.nefc]==ei))[0];eq.append({'name':m.equality(ei).name,'residual':d.efc_pos[ids].tolist(),'force':d.efc_force[ids].tolist()})
 fr=[]
 for fi in np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF))[0]:
  dof=int(d.efc_id[fi]);jn=int(m.dof_jntid[dof]);fr.append({'joint':m.joint(jn).name,'dof':dof,'solver_force':float(d.efc_force[fi]),'velocity':float(d.qvel[dof])})
 tau=d.actuator_force.copy();velocity=d.qvel[env.drive_vidx].copy();expected=env.limit_torque(env.kp*(env.target-d.qpos[env.qidx])-env.kd*d.qvel[env.vidx],velocity);assert np.max(abs(expected-d.ctrl))<1e-9
 power=float(np.maximum(tau*velocity,0).sum());assert power<=350+1e-8;maxpower=max(maxpower,power)
 rel=d.xmat[head].reshape(3,3).T@(d.geom_xpos[rod]-d.xpos[head]);both=min(normal)>.01
 if both and phase in ['close','pull','hold']:
  ever=True
  if capture is None:capture=rel.copy()
 margin=float(np.min(np.minimum(d.qpos[env.qidx[:5]]-env.ranges[:5,0],env.ranges[:5,1]-d.qpos[env.qidx[:5]])))
 if phase in ['pull','hold']:limits_margin=min(limits_margin,margin)
 row={'pre_time_s':pret,'phase':phase,'pre_qpos':preq.tolist(),'pre_qvel':prev.tolist(),'pre_bodies':bodies,'contacts':cs,'original_equalities_and_fixture':eq,'friction_dof_reactions':fr,'actual_torque18':tau.tolist(),'drive_speed18':velocity.tolist(),'target18':env.target.tolist(),'positive_power_W':power,'jaw_normal_N':normal.tolist(),'tow_on_rod_world_N':tow.tolist(),'wheel_ground_resultant_world_N':groundwheel.tolist(),'rod_world':d.geom_xpos[rod].tolist(),'rod_rel_head':rel.tolist(),'center_migration_from_capture_m':None if capture is None else float(np.linalg.norm(rel-capture)),'bilateral':bool(both),'head_neck_min_range_margin_rad':margin,'cart_back_m':float(initialcart[0]-d.xpos[cart,0]),'pitch_spring_force_Nm':float(d.qfrc_passive[pitchdof]),'pitch_bias_Nm':float(d.qfrc_bias[pitchdof]),'warnings':[int(w.number) for w in d.warning]}
 source_step(m,d,nstep=1);row.update(post_time_s=float(d.time),post_qpos=d.qpos.tolist(),post_qvel=d.qvel.tolist());log.write(json.dumps(row,separators=(',',':'))+'\n');records.append(row);qtraj.append(d.qpos.copy());vtraj.append(d.qvel.copy());times.append(float(d.time))
 if len(records)%200==0:
  threads=max(threads,int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:'))));print('EVENT',num,round(d.time,2),phase,'Fn',normal.round(2).tolist(),'backmm',round(row['cart_back_m']*1000,2),'margin',round(margin,3),flush=True)
 if phase in ['pull','hold'] and margin<.10:failure='invalid_head_neck_limit_margin_under0p10'
 if max(row['warnings']):failure='mujoco_warning'
 if not np.isfinite(d.qpos).all():failure='nonfinite'
mj.mj_step=logged;error=None
try:
 for tick in range(600):
  assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,10,18,14,3,3,tzinfo=datetime.timezone.utc)
  t=float(d.time);desired=env.neutral.copy()
  if t<.6:phase='open';target=pretarget.copy();jaw=.29
  elif t<1.8:
   phase='approach';u=(t-.6)/1.2;target=d.geom_xpos[rod]+np.array(sel['off'])*(1-u);jaw=.29
  elif t<3.:
   phase='close';target=d.geom_xpos[rod].copy();jaw=float(np.clip(jaw+.00008*(float(np.mean(normal))-20),0,.29)) if min(normal)>.01 else max(0,jaw-.004)
  elif t<5.5:
   phase='pull'
   if pulltarget is None:pulltarget=d.geom_xpos[rod].copy()
   u=np.clip((t-3)/2.5,0,1);target=pulltarget+[-.05*u*u*(3-2*u),0,0];jaw=float(np.clip(jaw+.00006*(float(np.mean(normal))-20),0,.29))
  elif t<10.7:phase='hold';target=pulltarget+[-.05,0,0];jaw=float(np.clip(jaw+.00006*(float(np.mean(normal))-20),0,.29))
  else:phase='release';target=pulltarget+[-.05,0,0];jaw=.29
  lastq,err=ik(target,lastq);maxerror=max(maxerror,err);desired[ni]=np.clip(lastq+d.qfrc_bias[env.vidx[ni]]/env.kp[ni],lo+.10,hi-.10);desired[5]=.29 if negative else jaw;commands.append({'time_s':t,'phase':phase,'desired_q18':desired.tolist(),'IK_error':err,'target_tcp_world':target.tolist()});env.step((desired-env.neutral)/env.scale)
  if failure:break
except BaseException as ex:error=repr(ex)
finally:
 mj.mj_step=source_step;log.close();np.savez_compressed(S/'trajectory.npz',qpos=np.array(qtraj),qvel=np.array(vtraj),time=np.array(times));dump(S/'commands50hz.json',commands);dump(S/'source_after.json',{p:sha(Path(p)) for p in sources});assert all(sha(Path(p))==h for p,h in sources.items());a={'attempt':num,'negative':negative,'actual_steps':len(records),'sim_time':float(d.time),'error':error,'failure':failure,'ever_bilateral':ever,'first_rod_contact':first_contact,'pull_hold_min_limit_margin':limits_margin,'max_ik_error':maxerror,'max_power_W':maxpower,'max_observed_threads':threads,'maxRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source53_unchanged':True,'physics_GPU':0,'training':0,'scope':'LOCAL_FIXED_TORSO'};dump(S/'summary.json',a);print('STATUS',json.dumps(a),flush=True)
