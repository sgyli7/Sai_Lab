from pathlib import Path
import json,importlib.util
R=Path(__file__).parent
src=R/'narrow_stance_readiness/screen_narrow.py'
ns={'__file__':str(R/'build_narrow_cycle.py')};exec(compile(src.read_text().split('# Every call to pose')[0],str(src),'exec'),ns)
np,mj,m,d=ns['np'],ns['mj'],ns['m'],ns['d'];Rotation=ns['Rotation'];qidx=ns['qidx'];cold=ns['cold'];footref=ns['footref']
spec=importlib.util.spec_from_file_location('foot_ik_reference',R/'narrow_stance_readiness/foot_ik_reference.py');ik=importlib.util.module_from_spec(spec);spec.loader.exec_module(ik)
p=json.loads((R/'program_toein_period4.json').read_text());checks=[]
for fi,f in enumerate(p['frames']):
 d.qpos[:]=cold;d.qpos[:3]+=np.array(f['expected_root_delta_xyz']);errs={}
 for side in ['left','right']:
  pos,rot=footref[side];off=np.array(f['foot_target_world_relative'][side],float);target=pos+off;target[1]=.07 if side=='left' else -.07;rot=Rotation.from_euler('z',-.25 if side=='left' else .25).as_matrix()@rot
  names=[side+'_'+s for s in ik.LEG_JOINT_SUFFIXES];ix=[int(m.joint(n).qposadr[0]) for n in names];seed=np.array(f['q_target18_rad'])[[ns['names'].index(n) for n in names]]
  errs[side]=ik.solve_foot_ik(m,d,side,target,rot,seed)
 neckerr=ns['neck'](np.array([.30+f['expected_root_delta_xyz'][0],0,.42]));mj.mj_forward(m,d)
 bad=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'distance_m':float(c.dist)} for c in d.contact if c.dist<-1e-6]
 lp=ns['static'](f['stance'],8) if not bad and max(e['equivalent_IK_error_m'] for e in errs.values())<.0005 else {'success':False,'tau18':None,'reason':'geometry_or_IK_not_admitted'}
 margins=[float(min(d.qpos[int(m.joint(n).qposadr[0])]-m.jnt_range[m.joint(n).id,0],m.jnt_range[m.joint(n).id,1]-d.qpos[int(m.joint(n).qposadr[0])])) for n in ns['names']]
 check={'frame':fi,'u':f['u'],'leg_IK':errs,'neckerr':neckerr,'bad_contacts':bad,'static8N':lp,'min_public_ROM_margin_rad':min(margins),'qpos_reference_only':d.qpos.tolist()};checks.append(check);f['q_target18_rad']=d.qpos[qidx].tolist();f['static_tau18_Nm']=lp.get('tau18');f['narrow_recomputed_check']=check;f.pop('full_check',None)
 print('FRAME',fi,'bad',bad,'LP',lp.get('success'),'ROM',min(margins),flush=True)
p.update(variant='narrow70_recomputed_original_cycle',cold_start_from_first_frame=True,period_s=4.,offline_only=True,physics_steps=0,geometry_scope='original_proxy_not_native',changes='foot center Y±70mm; original rootdelta/toein/Xstep/lift/period4 unchanged',single_static_not_admitted=True)
(R/'program_narrow70_period4.json').write_text(json.dumps(p,indent=2));(R/'narrow_cycle_build_checks.json').write_text(json.dumps({'frames':checks,'physics_steps':0,'poses':len(checks)},indent=2))
