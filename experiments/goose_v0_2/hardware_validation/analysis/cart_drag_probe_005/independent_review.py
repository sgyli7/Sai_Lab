from pathlib import Path
import gzip,json,hashlib,numpy as np,mujoco as mj
P=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_005/teacher_ik_mouth');O=Path(__file__).parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');C=json.load(open(W/'robots/goose_v0_2/models/physics_v0_2_20261018_04/actuator_contract.json'));rows=[]
for num,p in enumerate([P]+[P/f'attempt_{i:03}' for i in range(2,6)],1):
 m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));modelnames=[a['name'] for a in C['axes']];curves=[np.array(a['selected_torque_speed_curve_rad_s_nm'])*np.array([1 if a['reference_voltage_v']==5 else .925,1]) for a in C['axes']];caps=np.array([a['selected_continuous_design_limit_nm'] for a in C['axes']]);ph={};power=0;viol=0;closures={};contact_pairs={};count=0;first=None;last=None;limhit={};fhash={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['probe.py','trajectory.npz','summary.json','substeps.jsonl.gz']}
 with gzip.open(p/'substeps.jsonl.gz','rt') as ff:
  for line in ff:
   r=json.loads(line);count+=1;first=first or r;last=r;pp=r['phase'];z=ph.setdefault(pp,[]);z.append({k:r[k] for k in ['time_s','cart_pose','cargo_pose','root_pose','rod_world','rod_rel_head','slip_from_capture_m','angle_from_capture_rad','jaw_normal_N','tow_on_rod_world_N','jaw_pen_m']});t=np.array(r['torque18']);v=np.array(r['drive_speed18_incoming']);cc=np.minimum(caps,[np.interp(abs(vv),curve[:,0],curve[:,1],right=0) for vv,curve in zip(v,curves)]);viol+=int(np.max(abs(t)-cc)>1e-7);power=max(power,float(np.maximum(t*v,0).sum()))
   for eq in r['equalities']:
    closures[eq['name']]=max(closures.get(eq['name'],0),max([abs(x) for x in eq['residual']] or [0]))
   if pp in ['pull','hold']:
    for c in r['contacts']:
     if (c['a'].startswith('cart_'))!=(c['b'].startswith('cart_')) and c['a'] not in ['ground','probe_cargo_box'] and c['b'] not in ['ground','probe_cargo_box']:
      k=c['a']+'|'+c['b'];d=contact_pairs.setdefault(k,[0,0]);d[0]+=1;d[1]+=max(c['force_contact'][0],0)*.005
    for a in C['axes'][:5]:
     j=m.joint(a['name']);q=r['qpos'][j.qposadr[0]]
     if min(q-j.range[0],j.range[1]-q)<.005:limhit[a['name']]=limhit.get(a['name'],0)+1
 phases={}
 for phase,ss in ph.items():
  fn=np.array([x['jaw_normal_N'] for x in ss]);both=np.min(fn,axis=1)>.01;run=mx=0
  for b in both:run=run+1 if b else 0;mx=max(mx,run)
  phases[phase]=dict(duration_s=len(ss)*.005,cart_back_during_m=ss[0]['cart_pose'][0]-ss[-1]['cart_pose'][0],cargo_back_during_m=ss[0]['cargo_pose'][0]-ss[-1]['cargo_pose'][0],rod_start_end_z_m=[ss[0]['rod_world'][2],ss[-1]['rod_world'][2]],bilateral_fraction=float(both.mean()),longest_bilateral_s=mx*.005,mean_jaw_normal_N=fn.mean(0).tolist(),mean_tow_on_rod_N=np.mean([x['tow_on_rod_world_N'] for x in ss],axis=0).tolist(),capture_reference_max_slip_m=max([x['slip_from_capture_m'] or 0 for x in ss]),capture_reference_max_angle_deg=max([x['angle_from_capture_rad'] or 0 for x in ss])*180/np.pi,rod_rel_head_first_last=[ss[0]['rod_rel_head'],ss[-1]['rod_rel_head']],max_jaw_penetration_m=max(x['jaw_pen_m'] for x in ss))
 rows.append(dict(attempt=num,saved_files=fhash,phases=phases,actual_peak_positive_power_W=power,torque_speed_cap_violating_steps=viol,constraint_max_abs_residual=closures,pull_hold_neck_limit_hit_steps=limhit,pull_hold_robot_cart_contact_pairs_count_and_normal_impulse=contact_pairs,initial_rod_local=json.load(open(p/'preflight.json'))['rod_head_local'],sources_before_after_equal=json.load(open(p/'source_before.json'))==json.load(open(p/'source_after.json')),actual_pitch_stiffness=float(m.jnt_stiffness[m.joint('cart_pitch_hinge').id]),cart_mass_kg=sum(m.body_mass[m.body('cart_base').id:])-20,cargo_mass_kg=float(m.body_mass[m.body('probe_cargo').id]),equality_names=[m.equality(i).name for i in range(m.neq)]))
(O/'independent_review.json').write_text(json.dumps(dict(attempts=rows,new_integrations=0,new_LP=0,scope='Saved logs and saved binary model read only; no forward or stepping.'),indent=2)+'\n')
for r in rows:print(r['attempt'],'power',r['actual_peak_positive_power_W'],'viol',r['torque_speed_cap_violating_steps'],'hits',r['pull_hold_neck_limit_hit_steps'],'PH',r['phases'],'pairs',r['pull_hold_robot_cart_contact_pairs_count_and_normal_impulse'])
