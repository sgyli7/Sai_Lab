"""14 engineering numerical budget; no integration, state/model/solver edits.
G frozen source derivation be936e68…: native fixed-q/v convex objective Hessian>=M
implies ||accel error||_M <= rho=sqrt(r'M^-1r). Frozen-state local Taylor only:
.5*h²*rho*sqrt(j*M^-1*j') (coordinate) or sqrt(lambda_max(Jp*M^-1*Jp')).
This is not a finite-horizon or hardware safety certificate.
"""
import numpy as np,mujoco as mj
from scipy.linalg import cho_solve
H=.005;TRANSLATION_BUDGET_M=1e-6;ROTATION_BUDGET_RAD=1e-5;POINT_BUDGET_M=1e-6

def certificate(m,d,M,r):
 out=dict(valid=False,kind='instantaneous_frozen_state_native_solver_error_budget',h_gate_s=H,h_diagnostic_s=.001,translation_budget_m=TRANSLATION_BUDGET_M,rotation_budget_rad=ROTATION_BUDGET_RAD,point_budget_m=POINT_BUDGET_M,not_full_trajectory_guarantee=True,not_hardware_safety_threshold=True)
 try:
  M=np.asarray(M,float);r=np.asarray(r,float)
  if M.shape!=(m.nv,m.nv) or r.shape!=(m.nv,) or m.nv!=49:raise ValueError('unexpected_full_native_dimensions')
  if not np.isfinite(np.r_[M.ravel(),r,d.qpos,d.qvel]).all():raise ValueError('nonfinite_mass_residual_or_state')
  if max(abs(M-M.T).ravel())>1e-12:raise ValueError('nonsymmetric_mass_matrix')
  L=np.linalg.cholesky(M);Mi=cho_solve((L,True),np.eye(m.nv));rho=float(np.linalg.norm(np.linalg.solve(L,r)))
  if not np.isfinite(Mi).all() or not np.isfinite(rho) or np.any(np.diag(Mi)<=0):raise ValueError('invalid_SPD_inverse_or_rho')
  diag=np.diag(Mi);acc=rho*np.sqrt(diag);b5=.5*H**2*acc;b1=.5*.001**2*acc;types=[];names=[]
  for k in range(m.nv):
   j=int(m.dof_jntid[k]);jt=int(m.jnt_type[j]);off=k-int(m.jnt_dofadr[j]);translation=(jt==int(mj.mjtJoint.mjJNT_FREE) and off<3) or jt==int(mj.mjtJoint.mjJNT_SLIDE);types.append('translation_m' if translation else 'rotation_rad');suffix='/'+(['tx','ty','tz','rx','ry','rz'][off]) if jt==int(mj.mjtJoint.mjJNT_FREE) else '';names.append(m.joint(j).name+suffix)
  if types.count('translation_m')!=9 or types.count('rotation_rad')!=40:raise ValueError('unexpected_native_tangent_classification')
  limits=np.array([TRANSLATION_BUDGET_M if t=='translation_m' else ROTATION_BUDGET_RAD for t in types]);points=[]
  def point(label,b,p):
   J=np.zeros((3,m.nv));mj.mj_jac(m,d,J,None,np.asarray(p),int(b));Q=J@Mi@J.T;ev=np.linalg.eigvalsh((Q+Q.T)*.5)
   if not np.isfinite(np.r_[J.ravel(),ev]).all() or ev[0]<0:raise ValueError('nonfinite_or_negative_point_quadratic_form')
   a=float(rho*np.sqrt(ev[-1]));return dict(label=label,world_point_m=np.asarray(p).tolist(),acceleration_bound_m_s2=a,local_1ms_m=.5*.001**2*a,local_5ms_m=.5*H**2*a)
  for side in ['left','right']:
   g=m.geom(side+'_actual_sole').id;b=int(m.geom_bodyid[g]);mesh=int(m.geom_dataid[g]);n=int(m.mesh_vertnum[mesh]);adr=int(m.mesh_vertadr[mesh])
   if n!=63:raise ValueError('not_all_63_source_sole_vertices')
   ps=m.mesh_vert[adr:adr+n].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];rs=[point(side+'_sole_vertex_'+str(i),b,p) for i,p in enumerate(ps)];worst=max(rs,key=lambda z:z['local_5ms_m']);points.append(dict(label=side+'_all_source_sole_vertices',count=n,bounds_5ms_m=[p['local_5ms_m'] for p in rs],bounds_1ms_m=[p['local_1ms_m'] for p in rs],worst=worst,local_5ms_m=worst['local_5ms_m']))
  b=m.body('head_roll').id;p=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.13750001972253326,0,-.03849996489687674]);points.append(point('true_mouth_TCP',b,p));b=m.body('beak_coupler_link').id;eq=m.equality('beak_distal_pin').id
  if int(m.eq_obj2id[eq])!=b:raise ValueError('fourbar_pin_owner_mismatch')
  for label,local in [('coupler_proximal_pin',np.zeros(3)),('coupler_distal_pin',m.eq_data[eq,3:6]),('coupler_COM',m.body_ipos[b])]:points.append(point(label,b,d.xpos[b]+d.xmat[b].reshape(3,3)@local))
  pointpass=all(p['local_5ms_m']<=POINT_BUDGET_M for p in points);dofpass=bool(np.all(b5<=limits));out.update(valid=bool(dofpass and pointpass),rho_mass_dual_norm=rho,SPD_checked_by_Cholesky=True,mass_cholesky_min_diagonal=float(min(np.diag(L))),all_finite=True,dof_names=names,dof_units=types,dof_acceleration_bounds=acc.tolist(),dof_local_1ms_bounds=b1.tolist(),dof_local_5ms_bounds=b5.tolist(),dof_budget_ratios=(b5/limits).tolist(),all_49_dofs_pass=dofpass,points=points,all_critical_points_pass=pointpass,max_point_5ms_m=max(p['local_5ms_m'] for p in points),max_translation_5ms_m=max(b5[i] for i,t in enumerate(types) if t=='translation_m'),max_rotation_5ms_rad=max(b5[i] for i,t in enumerate(types) if t=='rotation_rad'),reason=None if dofpass and pointpass else 'local_numerical_motion_budget_exceeded')
 except (ValueError,np.linalg.LinAlgError,FloatingPointError) as ex:out.update(valid=False,reason=str(ex),all_finite=False)
 return out
