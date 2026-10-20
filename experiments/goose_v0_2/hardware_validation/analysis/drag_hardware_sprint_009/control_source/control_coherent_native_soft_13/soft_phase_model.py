"""009 bounded 49DOF soft-reference LP using existing native constraint algebra.
No integration, model mutation, state write or external force. One configuration.
Unknown FF is added to original200Hz physical PD. All physical relations are hard;
reference tracking/unload preference are soft and explicitly normalized.
"""
import numpy as np
import mujoco as mj
from scipy.optimize import linprog

def smooth(u):
 u=np.clip(u,0.,1.);return 10*u**3-15*u**4+6*u**5

def reference_derivatives(model,qminus,qzero,qplus,h):
 vm=np.zeros(model.nv);vp=vm.copy();mj.mj_differentiatePos(model,vm,h,qminus,qzero);mj.mj_differentiatePos(model,vp,h,qzero,qplus);return .5*(vm+vp),(vp-vm)/h

def caps(env,d):
 c=np.asarray(env.cont,float).copy()
 for i,row in enumerate(env.entries):
  curve=np.asarray(row['selected_torque_speed_curve_rad_s_nm'],float).copy()
  if row['reference_voltage_v']!=5:curve[:,0]*=env.voltage_speed_factor
  c[i]=min(c[i],float(np.interp(abs(d.qvel[env.drive_vidx[i]]),curve[:,0],curve[:,1],right=0.)))
 return c

def native_feedback_response(env,d,pd18,baseline_ff18,return_native_data=False):
 """Native constrained acceleration contribution of physical leg PD (no step).
 Source limiter is included, so unavailable saturated feedback is not invented.
 Two scratch forwards do not touch caller state. Front q-only remains fixed.
 """
 results=[];controls=[]
 for with_pd in [False,True]:
  x=mj.MjData(env.model);x.qpos[:]=d.qpos;x.qvel[:]=d.qvel
  request=np.asarray(baseline_ff18,float).copy();request[:6]=pd18[:6]
  if with_pd:request[6:]+=pd18[6:]
  x.ctrl[:]=env.limit_torque(request,d.qvel[env.drive_vidx]);mj.mj_forward(env.model,x);results.append(x.qacc.copy());controls.append(x.ctrl.copy())
 meta=dict(no_PD_total_tau=controls[0].tolist(),with_PD_total_tau=controls[1].tolist(),native_acceleration_feedback=(results[1]-results[0]).tolist(),native_qacc_without_leg_PD=results[0].tolist(),native_qacc_with_leg_PD=results[1].tolist(),definition='native acceleration(with original physical leg PD)-native acceleration(without leg PD), same q/v/source limits/front; zero integrations')
 return (results[1]-results[0],meta,x) if return_native_data else (results[1]-results[0],meta)

def solve(env,d,qdd_curve,pd18,baseline_ff18,left_normal_target_N,unload_weight,oracle=None,root_reference=None,unload_side="left"):
 assert unload_side in ("left","right")
 # Modes must correspond to THIS command's originalPD+previousactualFF, not
 # stale heldctrl. Reuse the already required with-PD scratch nativeforward.
 feedback,feedback_meta,fresh=native_feedback_response(env,d,np.asarray(pd18,float),np.asarray(baseline_ff18),return_native_data=True)
 d=fresh
 m=env.model;nv=m.nv;ne=d.nefc;idx=np.arange(6,18);na=12;J=d.efc_J.reshape(ne,nv).copy();RR=d.efc_R[:ne].copy();ar=d.efc_aref[:ne].copy();M=np.zeros((nv,nv));mj.mj_fullM(m,d,M);B=np.zeros((nv,18))
 for i in range(18):B[m.jnt_dofadr[m.actuator_trnid[i,0]],i]=m.actuator_gear[i,0]
 cap=caps(env,d);pd=np.asarray(pd18,float);sat=np.zeros(18,dtype=bool);sat[6:]=abs(pd[6:]+np.asarray(baseline_ff18)[6:])>cap[6:]+1e-9;sat_sign=np.sign(pd+np.asarray(baseline_ff18));known_tau=pd.copy();known_tau[sat]=sat_sign[sat]*cap[sat];Bleg=B[:,idx].copy();Bleg[:,sat[idx]]=0.;reference=np.asarray(qdd_curve,float)+feedback
 if root_reference is not None:
  from root_pose_feedback import root_acceleration_goal
  root_task=root_acceleration_goal(m,d.qpos,d.qvel,*root_reference);reference[:6]=root_task['goal']
 # Units/relative priorities: base vertical acceleration100/.2, base horizontal20/.5,
 # base roll/pitch50/2, yaw10/4; front inertial resource suppression30/5;
 # 12leg reference acceleration5/10. Passive qdd has no tracking objective.
 task_ids=np.arange(28);normalizer=np.r_[.5,.5,.2,2.,2.,4.,np.full(8,5.),np.full(14,10.)];priority=np.r_[20.,20.,100.,50.,50.,10.,np.full(8,30.),np.full(14,5.)]
 # Front is q-only. Its actual baseline acceleration, not artificial zero, is tracked.
 reference[6:14]=np.array(feedback_meta['native_qacc_with_leg_PD'])[6:14]
 # Variables:49qdd,12FF,all native constraint forces, task abs,12FF abs,
 #18positive total power, one left-normal target abs.
 it=nv;il=it+na;ia=il+ne;iF=ia+len(task_ids);ip=iF+na;iu=ip+18;N=iu+1
 A=[];be=[];U=[];bu=[];bounds=[(None,None)]*nv+[(-float(env.cont[i]),float(env.cont[i])) for i in idx]+[(None,None)]*ne+[(0,None)]*(len(task_ids)+na+18+1)
 dyn=np.zeros((nv,N));dyn[:,:nv]=M;dyn[:,it:il]=-Bleg;
 dyn[:,il:ia]=-J.T
 A.extend(dyn);be.extend(d.qfrc_passive+B@known_tau-d.qfrc_bias)
 g0=J@d.qacc+RR*d.efc_force[:ne]-ar;contact_rows=set();leftrow=np.zeros(N);mode_records=[]
 for k,c in enumerate(d.contact):
  if c.efc_address<0:continue # native reports no constraint rows for this contact
  assert c.dim==3 and int(m.opt.cone)==int(mj.mjtCone.mjCONE_ELLIPTIC)
  ids=np.arange(c.efc_address,c.efc_address+3);contact_rows.update(ids.tolist());mu=float(c.friction[0]);assert abs(mu-float(c.friction[1]))<1e-10
  F=d.efc_force[ids];gg=g0[ids];fn=float(F[0]);tn=float(np.linalg.norm(F[1:]));G=np.zeros((3,N));G[:,:nv]=J[ids];G[:,il+ids]=np.diag(RR[ids]);rhs=ar[ids];names=[m.geom(c.geom1).name,m.geom(c.geom2).name]
  bounds[il+ids[0]]=(0,None)
  if unload_side+'_actual_sole' in names and any(n in names for n in ['ground','sole_ground_box']):leftrow[il+ids[0]]=1
  # Actual physical branch; no planned swing tag forces premature Fn0.
  if fn<=1e-8:
   mode='released';
   for i in ids:bounds[il+i]=(0.,0.)
   # Fn0=>Ft0 EXACT. Safe linear sufficient dual circle approximation,
   # rotated so the actual dual-gradient direction is represented exactly.
   theta=np.arctan2(gg[2],gg[1]) if np.linalg.norm(gg[1:])>1e-12 else 0.
   for a in theta+np.pi/8+np.arange(8)*np.pi/4:
    v=np.r_[-1.,mu/np.cos(np.pi/8)*np.cos(a),mu/np.cos(np.pi/8)*np.sin(a)];U.append(v@G);bu.append(float(v@rhs))
  elif np.linalg.norm(gg)>1e-6 and tn>1e-8:
   mode='sliding';u=F[1:]/tn
   # Exact source circular boundary/sliding law, never an unlimited Ft branch.
   for j in range(2):
    row=np.zeros(N);row[il+ids[1+j]]=1;row[il+ids[0]]=-mu*u[j];A.append(row);be.append(0.)
   for v in [np.r_[1.,mu*u],np.r_[0.,-u[1],u[0]]]:A.append(v@G);be.append(float(v@rhs))
   v=np.r_[0.,u];U.append(v@G);bu.append(float(v@rhs))
  else:
   mode='sticking';A.extend(G);be.extend(rhs)
   # Physical-safe inner8gon, actual-force aligned vertex contains oracle.
   theta=np.arctan2(F[2],F[1]) if tn>1e-12 else 0.
   for a in theta+np.pi/8+np.arange(8)*np.pi/4:
    row=np.zeros(N);row[il+ids[0]]=-mu*np.cos(np.pi/8);row[il+ids[1:]]=np.r_[np.cos(a),np.sin(a)];U.append(row);bu.append(0.)
  mode_records.append(dict(original_contact_index=k,geoms=names,ids=ids.tolist(),mode=mode))
 for k in range(ne):
  if k in contact_rows:continue
  row=np.zeros(N);row[:nv]=J[k];row[il+k]=RR[k]
  typ=int(d.efc_type[k])
  if typ==int(mj.mjtConstraint.mjCNSTR_EQUALITY):A.append(row);be.append(float(ar[k]))
  elif typ==int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF):
   loss=float(m.dof_frictionloss[int(d.efc_id[k])]);force=float(d.efc_force[k]);bounds[il+k]=(-loss,loss)
   if abs(force)<loss-1e-7:A.append(row);be.append(float(ar[k]))
   else:
    sign=float(np.sign(force));bounds[il+k]=(sign*loss,sign*loss);U.append(sign*row);bu.append(float(sign*ar[k]))
  else:
   bounds[il+k]=(0,None)
   if d.efc_force[k]>1e-8:A.append(row);be.append(float(ar[k]))
   else:bounds[il+k]=(0,0);U.append(-row);bu.append(float(-ar[k]))
 for j,i in enumerate(task_ids):
  for sg in [-1,1]:
   row=np.zeros(N);row[i]=sg;row[ia+j]=-1;U.append(row);bu.append(float(sg*reference[i]))
 for k,i in enumerate(idx):
  # Exact active source clip branch. Applied torque, not raw PD+FF, enters dynamics.
  row=np.zeros(N);row[it+k]=1
  if sat[i]:
   if sat_sign[i]>0:U.append(-row);bu.append(float(pd[i]-cap[i]))
   else:U.append(row);bu.append(float(-cap[i]-pd[i]))
  else:U.append(row);bu.append(float(cap[i]-pd[i]));U.append(-row);bu.append(float(cap[i]+pd[i]))
  for sg in [-1,1]:row=np.zeros(N);row[it+k]=sg;row[iF+k]=-1;U.append(row);bu.append(0.)
 for i in range(18):
  row=np.zeros(N);row[ip+i]=-1
  if i>=6 and not sat[i]:row[it+i-6]=d.qvel[env.drive_vidx[i]]
  U.append(row);bu.append(float(-known_tau[i]*d.qvel[env.drive_vidx[i]]))
 row=np.zeros(N);row[ip:iu]=1;U.append(row);bu.append(350.)
 for sg in [-1,1]:row=sg*leftrow.copy();row[iu]=-1;U.append(row);bu.append(float(sg*left_normal_target_N))
 if oracle is not None:
  for k,v in enumerate(oracle['qacc']):row=np.zeros(N);row[k]=1;A.append(row);be.append(float(v))
  for k,v in enumerate(oracle['force']):row=np.zeros(N);row[il+k]=1;A.append(row);be.append(float(v))
  for k,i in enumerate(idx):row=np.zeros(N);row[it+k]=1;A.append(row);be.append(float(oracle['tau'][i]-pd[i]))
 obj=np.zeros(N);obj[ia:iF]=priority/normalizer;obj[iF:ip]=.02/np.asarray(env.cont)[idx];obj[iu]=float(unload_weight)/100.
 # Known actual point must satisfy hard algebra unless the NEW PD decomposition
 # exceeds the conservative FF software bound. Diagnose instead of blaming hardware.
 xx=np.zeros(N);xx[:nv]=d.qacc;xx[it:il]=np.asarray(baseline_ff18)[idx];xx[il:ia]=d.efc_force[:ne];xx[ia:iF]=abs(d.qacc[:28]-reference[:28]);xx[iF:ip]=abs(xx[it:il]);xx[ip:iu]=np.maximum(d.actuator_force*d.qvel[env.drive_vidx],0.);xx[iu]=abs(leftrow@xx-left_normal_target_N)
 eqv=np.asarray(A)@xx-np.asarray(be);uv=np.asarray(U)@xx-np.asarray(bu);bv=[max(0.,(b[0]-xx[k]) if b[0] is not None else 0.,(xx[k]-b[1]) if b[1] is not None else 0.) for k,b in enumerate(bounds)];known=dict(max_eq=float(max(abs(eqv))),eq_row=int(np.argmax(abs(eqv))),max_ineq=float(max(uv)),ineq_row=int(np.argmax(uv)),max_bound=float(max(bv)),bound_index=int(np.argmax(bv)),FF_needed_actual_total=xx[it:il].tolist())
 result=linprog(obj,A_eq=np.asarray(A),b_eq=np.asarray(be),A_ub=np.asarray(U),b_ub=np.asarray(bu),bounds=bounds,method='highs',options={'time_limit':.8,'threads':1,'primal_feasibility_tolerance':1e-6,'presolve':False})
 out=dict(contact_active_mask=[bool(c.efc_address>=0) for c in d.contact],inactive_contacts_without_native_rows=[k for k,c in enumerate(d.contact) if c.efc_address<0],unload_side=unload_side,root_task=root_task if root_reference is not None else None,known_actual_point=known,success=bool(result.success),status=result.message,physics_steps=0,source_hard_relations='native full49 dynamics + same-command fresh native equality/contact/friction branches, source actuator total caps/power',mode_initialization='same q/v THIS originalPD+previousactualFF source-limited nativeforward; reuse feedback scratch; no stale heldctrl modes',task_acceleration_goal=reference.tolist(),task_ids=task_ids.tolist(),task_normalizers=normalizer.tolist(),task_priorities=priority.tolist(),left_normal_target_N=float(left_normal_target_N),left_normal_soft_weight=float(unload_weight),contact_modes=mode_records,feedback=feedback_meta,FF_semantics='physical PD appears once in hard total dynamics; root acceleration uses explicit planned pose/velocity task when root_reference supplied, never native jointPD root delta; joint soft recovery retains nativePD response; heldFF retains original200Hz PD, not a stability proof',unload_normal_target_N=float(left_normal_target_N),legacy_left_field_names_follow_unload_side=True,physical_PD18=pd.tolist(),source_caps18=cap.tolist(),source_saturation_axes=np.flatnonzero(sat).tolist(),source_applied_known_tau=known_tau.tolist())
 # A numerical failure may continue ONE held-FF20ms command only when this
 # exact fresh-native baseline is a finite, source-legal hard witness. This is
 # not an optimal solution and does not bypass any physical/reference guard.
 finite=all(np.isfinite(np.asarray(a,float)).all() for a in [np.asarray(A),be,np.asarray(U),bu,obj,xx,reference,pd,np.asarray(baseline_ff18)]) and all(np.isfinite(v) for b in bounds for v in b if v is not None)
 witness_total=env.limit_torque(pd+np.asarray(baseline_ff18),d.qvel[env.drive_vidx]);witness_power=float(np.maximum(witness_total*d.qvel[env.drive_vidx],0.).sum());source_error=float(max(abs(witness_total-d.actuator_force)));FFlegal=bool(max(abs(np.asarray(baseline_ff18))-env.cont)<=0. and max(abs(np.asarray(baseline_ff18)[:6]))==0.)
 source_legal=bool(np.isfinite(witness_total).all() and max(abs(witness_total)-cap)<=1e-12 and witness_power<=350.+1e-12 and source_error<=1e-6)
 valid=bool(finite and FFlegal and source_legal and known['max_eq']<=1e-6 and known['max_ineq']<=1e-6 and known['max_bound']==0.)
 out['validated_witness']=dict(valid=valid,all_coefficients_targets_witness_and_specified_bounds_finite=bool(finite),hard_equality_max=known['max_eq'],hard_inequality_max=known['max_ineq'],hard_bound_max=known['max_bound'],FF_legal=FFlegal,source_legal=source_legal,source_applied_model_error=source_error,positive_power_W=witness_power,previous_actual_FF18=np.asarray(baseline_ff18).tolist(),source_limited_total_tau18=witness_total.tolist(),scope='fresh same-qv actual native baseline only; not optimum or future20ms feasibility certificate')
 if result.success:
  x=result.x;ff=np.zeros(18);ff[idx]=x[it:il];total=pd+ff;actual=env.limit_torque(total,d.qvel[env.drive_vidx]);modeled_applied=known_tau.copy();modeled_applied[6:]+=ff[6:]*(~sat[6:]);force=x[il:ia];grad=J@x[:nv]+RR*force-ar;out.update(FF18=ff.tolist(),requested_total_tau18=total.tolist(),source_limited_total_tau18=actual.tolist(),source_clip_max=float(max(abs(total-actual))),source_applied_model_error=float(max(abs(actual-modeled_applied))),predicted_full_qacc=x[:nv].tolist(),native_force=force.tolist(),unload_normal_predicted_N=float(leftrow@x),left_normal_predicted_N=float(leftrow@x),task_error=(x[:28]-reference[:28]).tolist(),hard_equality_residual=float(max(abs(np.asarray(A)@x-np.asarray(be)))),hard_inequality_residual=float(max(np.asarray(U)@x-np.asarray(bu))),front_acceleration=x[6:14].tolist(),maximum_contact_circle_excess=max([float(max(0,np.linalg.norm(force[np.array(c['ids'])[1:]])-d.contact[c['original_contact_index']].friction[0]*force[c['ids'][0]])) for k,c in enumerate(mode_records)]),default_FF_component_guard_not_hardware=True)
 return out


def dispatch_or_validated_witness(result,previous_actual_FF18,consecutive,total):
 """Bounded drive continuation, no solver/model/clock change."""
 if result['success']:
  return dict(accepted=True,witness_hold=False,consecutive=0,total=int(total),FF18=result['FF18'],reason=None,optimizer_status=result['status'])
 valid=result.get('validated_witness',{}).get('valid',False)
 if valid and consecutive<2:
  return dict(accepted=True,witness_hold=True,consecutive=int(consecutive)+1,total=int(total)+1,FF18=np.asarray(previous_actual_FF18).tolist(),reason=None,optimizer_status=result['status'])
 return dict(accepted=False,witness_hold=False,consecutive=int(consecutive),total=int(total),FF18=None,reason='validated_witness_hold_consecutive_limit2' if valid else 'unsuccessful_solver_without_validated_finite_hard_witness',optimizer_status=result['status'])
