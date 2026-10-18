from pathlib import Path
import json,hashlib,math,time,os
start=time.time();W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');S=W/'artifacts/goose_v0_2/low_handle_study_01';O=Path(__file__).resolve().parent
source=(S/'static_height_study.py').read_text();prefix=source[:source.index('results=[];boundsq=')]
ns={'__file__':str(S/'static_height_study.py'),'__name__':'static_library_prefix'}
# Exact previously executed model, foot support and equality/actuator LP method; matrix driver is not executed.
exec(compile(prefix,str(S/'static_height_study.py'),'exec'),ns)
sel=json.loads((S/'selected_420_interface.json').read_text());np=ns['np'];mj=ns['mj'];M=ns['M'];D=ns['D']
extra=20*math.tan(math.radians(10));rows=[]
for lab,pose,fz in [('old480_no_balance',sel['baseline_480'],-2.16),('new420_self_weight_balanced',sel['selected'],-extra),('new420_no_balance',sel['selected'],-extra-2.16)]:
 ns['BAR_X']=.3;ns['BAR_PITCH']=pose['cart_pitch_rad'];ns['setq'](np.array(pose['q_neck_mid_head_rad']),pose['leg_pose_rad'],pose['root_xz_m'],pose['target_height_m']);mj.mj_forward(M,D);tcp=ns['tcp']().copy();assert np.linalg.norm(tcp-[.3,0,pose['target_height_m']])<1e-7
 result=ns['equilibrium'](tcp,np.array([20.,0,fz]));result['continuous_static_gate']='PASS_STATIC_ONLY' if result.get('whole_min_max_util',1e9)<=1 else 'FAIL_CONTINUOUS_STATIC';rows.append({'case':lab,'tcp_world_m':tcp.tolist(),'result':result})
base=sel['baseline_480']['force_cases'][2]
rows.insert(0,{'case':'old480_self_weight_balanced','source':'existing exactly Fx20/Fz0, no recomputation','tcp_world_m':sel['baseline_480']['tcp_world_m'],'result':base})
receipt={'status':'SPECIFIC_STATIC_TONGUE_LOAD_HYPOTHESIS_NOT_MEASURED','assumptions':{'Fx_world_N':20,'new_pitch_deg':10,'bar_minus_pitch_axis_height_m':.04428028530506725,'extra_pitch_moment_Nm':.04428028530506725*20,'bar_pitch_axis_horizontal_distance_m':.255*math.cos(math.radians(10)),'extra_downward_mouth_load_N':extra,'self_weight_load_without_balance_N':2.16,'no_spring_note':'Specified common2.16N rounded self-weight hypothesis, not measured or full exact new02 spring/COM reconstruction.'},'comparisons':rows,'physics_integrations':0,'new_forward_calls':3,'setup_position_forward_calls':2,'new_LP_solves':3,'training_updates':0,'GPU':0,'wall_s':time.time()-start,'original_static_source_sha256':hashlib.sha256((S/'static_height_study.py').read_bytes()).hexdigest(),'reused_library_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'immutable_note':'No existing static-study files, CAD, collision002/003 or04 inputs written; this new supplement only.','scope':'Nominal zero-speed continuous static allocation only. Not spring dynamics, grab retention, cart rolling/traction, gait or turn qualification.'}
(O/'supplement.json').write_text(json.dumps(receipt,indent=2)+'\n')
for r in rows:
 x=r['result'];print(r['case'],'F',x['F_world_N'],'util',x.get('whole_min_max_util'),'worst',x.get('worst_axis'),'upper',x.get('upper_chain_utilisation'),'CoPmargin',x.get('support_fore_aft_margin_m'))
