from pathlib import Path
import gzip,json,hashlib,sys,importlib.util,collections,numpy as np,mujoco as mj
sys.dont_write_bytecode=True
P=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006');O=Path(__file__).parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');C=json.load(open(W/'robots/goose_v0_2/models/physics_v0_2_20261018_04/actuator_contract.json'));spec=importlib.util.spec_from_file_location('native_readonly',P/'native_pad_review/classify_native_pad.py');clf=importlib.util.module_from_spec(spec);spec.loader.exec_module(clf);dom=clf.load_domains();out=[];samples=[]
for num in range(1,5):
 p=P/f'attempt_{num:03}';m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));ph={};viol=0;peakpower=0.;firstshaft=None;shaftmax=0;firstrod=None;firstboth=None;firstback=None;pairimp=collections.defaultdict(float);candidate=[];closure={}
 for line in gzip.open(p/'substeps.jsonl.gz','rt'):
  r=json.loads(line);phase=r['phase'];ss=ph.setdefault(phase,[]);ss.append(r);tau=np.array(r['actual_torque18']);v=np.array(r['drive_speed18']);caps=[]
  for ax,vv in zip(C['axes'],v):
   curve=np.array(ax['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=(1 if ax['reference_voltage_v']==5 else .925);caps.append(min(ax['selected_continuous_design_limit_nm'],np.interp(abs(vv),curve[:,0],curve[:,1],right=0)))
  viol+=int(np.max(abs(tau)-caps)>1e-7);peakpower=max(peakpower,float(np.maximum(tau*v,0).sum()))
  if r['bilateral'] and firstboth is None:firstboth=r['pre_time_s'];firstback=r['cart_back_m']
  for c in r['contacts']:
   names=[c['geom1'],c['geom2']];fn=max(c['force_contact_6D'][0],0)
   if set(names)==set(['upper_bill_envelope','cart_rotating_shaft']) and fn>.01:
    firstshaft=firstshaft or dict(time=r['pre_time_s'],phase=phase,Fn_N=fn);shaftmax=max(shaftmax,fn)
   if 'cart_bite_bar'in names and any('bill_envelope'in n for n in names) and fn>.01 and firstrod is None:firstrod=dict(time=r['pre_time_s'],phase=phase)
   if phase in ['pull','hold'] and (names[0].startswith('cart_')!=names[1].startswith('cart_')) and 'ground' not in names and 'probe_cargo_box' not in names:pairimp['|'.join(names)]+=fn*.005
   if num==3 and phase in ['pull','hold'] and 'cart_bite_bar'in names and fn>.01:
    for k,n in enumerate(names):
     if n in ['upper_bill_envelope','lower_bill_envelope']:
      side='upper' if n.startswith('upper') else 'lower';body='head_roll' if side=='upper' else 'beak_hinge';nn=np.array(c['frame_rows_world'][:3]);point=np.array(c['pos_world_m'])+(-.5 if k==0 else .5)*c['distance_m']*nn;b=r['pre_bodies'][body];candidate.append((r['pre_time_s'],side,point,b,nn if k==0 else -nn))
  for eq in r['original_equalities_and_fixture']:
   closure[eq['name']]=max(closure.get(eq['name'],0),max([abs(v) for v in eq['residual']]or[0]))
 phases={}
 for phase,rr in ph.items():
  first,last=rr[0],rr[-1];F=np.array([q['jaw_normal_N'] for q in rr]);both=np.min(F,axis=1)>.01;run=longest=0
  for x in both:run=run+1 if x else 0;longest=max(longest,run)
  fr=[sum(abs(x['solver_force']) for x in q['friction_dof_reactions'] if x['joint'].startswith('cart_wheel_spin'))/.0625 for q in rr]
  phases[phase]=dict(duration_s=len(rr)*.005,cart_back_delta_m=last['cart_back_m']-first['cart_back_m'],cargo_back_delta_m=first['pre_bodies']['probe_cargo']['xpos'][0]-last['pre_bodies']['probe_cargo']['xpos'][0],jaw_normal_mean_N=F.mean(0).tolist(),bilateral_fraction=float(both.mean()),longest_bilateral_s=longest*.005,rod_z_min_max_m=[min(q['rod_world'][2] for q in rr),max(q['rod_world'][2] for q in rr)],mean_tow_on_rod_N=np.mean([q['tow_on_rod_world_N'] for q in rr],axis=0).tolist(),mean_ground_wheel_N=np.mean([q['wheel_ground_resultant_world_N'] for q in rr],axis=0).tolist(),actual_bearing_abs_over_radius_mean_N=float(np.mean(fr)),minimum_head_neck_limit_margin_rad=min(q['head_neck_min_range_margin_rad'] for q in rr),center_migration_end_m=last['center_migration_from_capture_m'])
 if num==3:
  for kk in np.linspace(0,len(candidate)-1,54).astype(int):
   t,side,point,b,normal=candidate[kk];res=[]
   for tol in [.0005,.0001]:res.append(clf.classify_point(side,point.tolist(),b['xpos'],b['xmat'],domains=dom,normal_from_pad_to_other_world=normal.tolist(),plane_tol_m=tol,edge_band_m=tol))
   samples.append(dict(pre_time_s=t,side=side,classification=res))
 out.append(dict(attempt=num,phases=phases,first_shaft_contact=firstshaft,peak_shaft_contact_N=shaftmax,first_rod_contact=firstrod,first_bilateral_s=firstboth,cart_back_at_first_bilateral_m=firstback,actual_peak_positive_power_W=peakpower,drive_torque_speed_violating_steps=viol,pull_hold_pair_normal_impulse_Ns=dict(pairimp),max_eq_residual=closure,pitch_stiffness=float(m.jnt_stiffness[m.joint('cart_pitch_hinge').id]),source_before_after_equal=json.load(open(p/'source_before.json'))==json.load(open(p/'source_after.json')),files_sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['summary.json','trajectory.npz','substeps.jsonl.gz','commands50hz.json','declaration.json']}))
result=dict(verdict='LIMITED_LOCAL_SHORT_RETENTION_AND_CONTACT_TRACTION_SUPPORTED',attempts=out,independent_54_native_prestate_contact_samples=samples,physics_integrations=0,new_LP_runs=0,new_forward_calls=0,scope='Saved binary models, exact prestate force logs, and pure native geometry classification only.')
(O/'independent_review.json').write_text(json.dumps(result,indent=2)+'\n')
for r in out:print('ATT',r['attempt'],'shaft',r['first_shaft_contact'],r['peak_shaft_contact_N'],'capture',r['first_bilateral_s'],r['cart_back_at_first_bilateral_m'],'pull',r['phases']['pull'],'hold',r['phases'].get('hold'),'pairs',r['pull_hold_pair_normal_impulse_Ns'])
print('SAMPLELABELS',collections.Counter(x['classification'][0]['classification'] for x in samples))
