from pathlib import Path
import sys,json,gzip,hashlib,collections
import numpy as np
R=Path(__file__).parent;sys.path.insert(0,str(R/'native_pad_review'));from classify_native_pad import load_domains,classify_point
D=load_domains();allA=[]
for S in sorted(R.glob('attempt_*')):
 if not (S/'summary.json').exists():continue
 rows=[json.loads(x) for x in gzip.open(S/'substeps.jsonl.gz','rt')];classified=[];phase={};firstboth=next((r for r in rows if r['bilateral']),None)
 for p in ['open','approach','close','pull','hold','release']:
  rr=[r for r in rows if r['phase']==p]
  if not rr:continue
  index=rows.index(rr[0]);prev=rows[max(index-1,0)];bins=collections.defaultdict(float);counts=collections.Counter();pen=0;eqmax=0;wheel=[]
  for r in rr:
   for c in r['contacts']:
    names=[c['geom1'],c['geom2']];n=np.array(c['frame_rows_world'][:3]);mid=np.array(c['pos_world_m']);dist=c['distance_m']
    if 'cart_bite_bar' in names and ('upper_bill_envelope' in names or 'lower_bill_envelope' in names):pen=max(pen,-dist)
    for k,name in enumerate(names):
     side='upper' if name=='upper_bill_envelope' else ('lower' if name=='lower_bill_envelope' else None)
     if side is None:continue
     owner='head_roll' if side=='upper' else 'beak_hinge';pose=r['pre_bodies'][owner];point=mid+(-.5 if k==0 else .5)*dist*n
     for tol in [.0005,.0001]:
      result=classify_point(side,point.tolist(),pose['xpos'],pose['xmat'],domains=D,normal_from_pad_to_other_world=(n if k==0 else -n).tolist(),plane_tol_m=tol,edge_band_m=tol)
      row={'time_s':r['pre_time_s'],'phase':p,'side':side,'pad_geom':name,'other_geom':names[1-k],'force_normal_N':c['force_contact_6D'][0],'contact_distance_m':dist,**result};classified.append(row)
      if names[1-k]=='cart_bite_bar':bins[f'{side}:{tol}:{result["classification"]}']+=max(c['force_contact_6D'][0],0)*.005;counts[f'{side}:{tol}:{result["classification"]}']+=1
   eqmax=max(eqmax,max(abs(x) for x in r['original_equalities_and_fixture'][0]['residual']))
   wf=[x for x in r['friction_dof_reactions'] if x['joint'].startswith('cart_wheel_spin')];wheel.append(sum(abs(x['solver_force'])/.0625 for x in wf))
  phase[p]={'duration_s':len(rr)*.005,'cart_back_delta_m':rr[-1]['cart_back_m']-prev['cart_back_m'],'cart_back_start_m':prev['cart_back_m'],'cart_back_end_m':rr[-1]['cart_back_m'],'jaw_normal_mean_N':np.mean([r['jaw_normal_N'] for r in rr],axis=0).tolist(),'tow_mean_world_N':np.mean([r['tow_on_rod_world_N'] for r in rr],axis=0).tolist(),'wheel_ground_resultant_mean_world_N':np.mean([r['wheel_ground_resultant_world_N'] for r in rr],axis=0).tolist(),'wheel_friction_abs_over_R_mean_N':float(np.mean(wheel)),'bilateral_duty':np.mean([r['bilateral'] for r in rr]),'rod_z_min_max_m':[min(r['rod_world'][2] for r in rr),max(r['rod_world'][2] for r in rr)],'center_migration_end_m':rr[-1]['center_migration_from_capture_m'],'head_neck_min_limit_margin':min(r['head_neck_min_range_margin_rad'] for r in rr),'beak_pin_max_abs_residual_m':eqmax,'mouth_rod_max_penetration_m':pen,'native_contact_impulse_bins_Ns':dict(bins),'native_contact_counts':dict(counts)}
 with gzip.open(S/'classified_contacts.jsonl.gz','wt',compresslevel=2) as f:
  for r in classified:f.write(json.dumps(r,separators=(',',':'))+'\n')
 a={'directory':str(S),'summary':json.loads((S/'summary.json').read_text()),'phases':phase,'first_bilateral_time_s':None if firstboth is None else firstboth['pre_time_s'],'cart_back_before_bilateral_m':None if firstboth is None else firstboth['cart_back_m'],'classifier':'native BRep8 body-local footprint/plane/normal,0.5mm and0.1mm sensitivity; not material friction certification','center_migration_is_not_material_tangential_slip':True};(S/'analysis.json').write_text(json.dumps(a,indent=2));allA.append(a);print('ANALYZED',S.name,flush=True)
(R/'all_attempts_analysis.json').write_text(json.dumps(allA,indent=2))
