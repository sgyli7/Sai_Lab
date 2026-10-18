from pathlib import Path
import json
F=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/full_body_readiness');R=Path(__file__).parent
source=(F/'prepare_full_body.py').read_text();source=source[:source.index('configs=[')];source=source.replace('R=Path(__file__).parent',f'R=Path({str(F)!r})')
needle="return {'tow_reaction_world_X_N':tow"
insert="""primary_t=float(result.x[-1]) if result.success else None
 if result.success:
  oldn=A.shape[1];bounds[-1]=(0,primary_t+1e-7);A=np.c_[A,np.zeros((m.nv,na))];Aub=[np.r_[row,np.zeros(na)] for row in Aub]
  for i in range(na):
   for sign in [-1,1]:
    row=np.zeros(oldn+na);row[nf+i]=sign;row[oldn+i]=-1;Aub.append(row);bub.append(0.)
  bounds.extend([(0,None)]*na);obj=np.r_[np.zeros(oldn),np.ones(na)];result=linprog(obj,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=A,b_eq=b,bounds=bounds,method='highs',options={'time_limit':5.})
 """
source=source.replace(needle,insert+needle);scope={'__file__':str(F/'prepare_full_body.py')};exec(compile(source,str(R/'copied_static_definitions'), 'exec'),scope)
m=scope['m'];d=scope['d'];A=json.loads((F/'keyframes_static_checks.json').read_text());result=[]
for row in A['rows']:
 if row['name'] not in ['double_neutral','double_crouch','double_back_020']:continue
 d.qpos[:]=row['qpos_offline_only'];new=scope['static'](['left','right'],0.);old=row['static_checks'][0];result.append({'name':row['name'],'old_tau18':old['tau18_Nm'],'new_tau18':new['tau18_Nm'],'old_max_utilization':old['max_continuous_utilization'],'new_max_utilization':new['max_continuous_utilization'],'new_solver_success':new['solver_success'],'new_balance_residual':new['max_force_balance_residual'],'same_qpos':True,'same_original_caps':True,'same_support_mu':.65,'secondary_objective':'min sumabs actuator tau after retaining minimax-optimal utilization bound+1e-7'})
(R/'l1_bias_proposal.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
