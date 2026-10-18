import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable'
from pathlib import Path
import json,gzip,hashlib,subprocess
import numpy as np,mujoco as mj
R=Path(__file__).parent;P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04');C=json.loads((P/'physical_contract.json').read_text());D=json.loads((P/'actuator_contract.json').read_text());A=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n in [1,2,3]:
 S=R/f'attempt_{n:03d}';rows=[json.loads(s) for s in gzip.open(S/'substeps.jsonl.gz','rt')];summary=json.loads((S/'summary.json').read_text());tau=np.array([r['actual_torque18'] for r in rows]);vel=np.array([r['actual_drive_speed18'] for r in rows]);caps=np.zeros_like(tau)
 for i,ax in enumerate(D['axes']):
  curve=np.array(ax['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=1 if ax['reference_voltage_v']==5 else min(D['power_supply']['nominal_bus_v']/24,1);caps[:,i]=np.minimum(C['joints'][i]['continuous_design_limit_nm'],np.interp(abs(vel[:,i]),curve[:,0],curve[:,1],right=0))
 assert np.max(abs(tau)-caps)<1e-8
 H=json.loads((S/'source_before.json').read_text());assert len(H)==53 and all(sha(p)==h for p,h in H.items());hold=[r for r in rows if 5<=r['pre_time_s']<7];phases={}
 for label,t0,t1 in [('settle',0,1),('crouch',1,2.5),('body_back',2.5,5),('hold',5,7),('return_crouch',7,9.5),('return_neutral',9.5,11)]:
  rr=[r for r in rows if t0<=r['pre_time_s']<t1]
  if not rr:continue
  phases[label]={'actual_duration_s':len(rr)*.005,'root_back_start_end_m':[rr[0]['root_back_m'],rr[-1]['root_back_m']],'root_down_range_m':[min(r['root_down_m'] for r in rr),max(r['root_down_m'] for r in rr)],'foot_force_mean_world_N':np.mean([r['foot_force_world_N'] for r in rr],axis=0).tolist(),'foot_normal_min_N':np.min([r['foot_normal_N'] for r in rr],axis=0).tolist(),'dual_support_duty':np.mean([r['dual_support'] for r in rr]),'foot_material_slip_end_m':rr[-1]['foot_material_slip_integral_m'],'foot_center_max_excursion_m':np.max([r['foot_geom_center_excursion_m'] for r in rr],axis=0).tolist(),'max_tilt_deg':max(r['torso_tilt_deg'] for r in rr),'max_TCP_error_m':max(r['actual_TCP_error_m'] for r in rr),'max_penetration_m':max(r['max_penetration_m'] for r in rr),'max_closure_by_name':{rr[0]['equalities'][i]['name']:max(abs(x) for r in rr for x in r['equalities'][i]['residual']) for i in range(3)}}
 streak=0.;maxstreak=0.
 for r in rows:streak=streak+.005 if r['actual_TCP_error_m']>.005 else 0.;maxstreak=max(maxstreak,streak)
 friction=[]
 for side in ['right_actual_sole','left_actual_sole']:
  ratios=[];normalimp=0.;satimp=0.
  for r in rows:
   for c in r['contacts']:
    if side in (c['geom1'],c['geom2']) and 'ground' in (c['geom1'],c['geom2']):
     f=np.array(c['force_contact6']);N=max(f[0],0)
     if N>.01:ratio=np.linalg.norm(f[1:3])/(.65*N);ratios.append(ratio);normalimp+=N*.005;satimp+=N*.005*(ratio>.99)
  friction.append({'foot':side,'contact_friction_utilization_p95':float(np.quantile(ratios,.95)),'contact_friction_utilization_max':max(ratios),'normal_impulse_at_mu_saturation_fraction':satimp/normalimp})
 m=mj.MjModel.from_binary_path(str(S/'actual_scene.mjb'));d=mj.MjData(m);tr=np.load(S/'trajectory.npz');feet=[m.geom(x).id for x in ['right_actual_sole','left_actual_sole']];d.qpos[:]=tr['qpos'][0];mj.mj_kinematics(m,d);foot0=d.geom_xpos[feet].copy();snapshot=[]
 for t in [0,1,2.5,5,6,7,9.5,11]:
  idx=int(np.argmin(abs(tr['time']-t)))
  if abs(float(tr['time'][idx])-t)>.03:continue
  d.qpos[:]=tr['qpos'][idx];mj.mj_kinematics(m,d);snapshot.append({'time_s':float(tr['time'][idx]),'root_xyz':d.qpos[:3].tolist(),'foot_geom_signed_displacement_world_m':(d.geom_xpos[feet]-foot0).tolist(),'foot_R_world':[d.geom_xmat[g].tolist() for g in feet]})
 videos=[]
 for name in ['continuous_original_speed.mp4','bright_proxy_original_speed.mp4']:
  p=S/name
  if p.exists():
   proc=subprocess.run(['ffmpeg','-i',str(p),'-f','null','-'],capture_output=True,text=True);assert proc.returncode==0;videos.append({'path':str(p),'sha256':sha(p),'decode_returncode':proc.returncode,'readback':proc.stderr})
 out={'attempt':n,'summary':summary,'phases':phases,'caps22p2V_verified':True,'max_actual_torque18_Nm':np.max(abs(tau),axis=0).tolist(),'max_actual_drive_speed18_rad_s':np.max(abs(vel),axis=0).tolist(),'caps_saturation_duty18':np.mean(abs(tau)>=caps-1e-6,axis=0).tolist(),'foot_friction_diagnostics':friction,'signed_foot_pose_snapshots':snapshot,'TCP_abort_implementation_audit':{'runner_missing_5mm_200ms_abort':True,'actual_TCP_error_max_m':max(r['actual_TCP_error_m'] for r in rows),'actual_max_continuous_duration_over5mm_s':maxstreak,'would_original_TCP_abort_have_triggered':maxstreak>=.2},'source_robot53_unchanged':True,'only_original_equalities':[m.equality(i).name for i in range(m.neq)],'compiled_mass_kg':float(sum(m.body_mass)),'compiled_robot_geoms':m.ngeom-1,'physics_dt_s':m.opt.timestep,'videos':videos};(S/'analysis.json').write_text(json.dumps(out,indent=2));A.append(out);print('ANALYZED',n,flush=True)
t2=np.load(R/'attempt_002/trajectory.npz');t3=np.load(R/'attempt_003/trajectory.npz');prefix={'trial2_3_common_qpos_prefix_bitwise_equal':np.array_equal(t2['qpos'],t3['qpos'][:len(t2['qpos'])]),'trial2_3_common_qvel_prefix_bitwise_equal':np.array_equal(t2['qvel'],t3['qvel'][:len(t2['qvel'])]),'shared_states':len(t2['qpos'])};(R/'diagnostic_prefix_verification.json').write_text(json.dumps(prefix,indent=2));(R/'all_attempts_analysis.json').write_text(json.dumps(A,indent=2));print('PREFIX',prefix)
