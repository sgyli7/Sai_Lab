"""Read-only derived actual-state centroidal verification; never a future prescribed mouth load."""
import numpy as np,mujoco as mj
def measure(m,d):
 torso=m.body('torso').id;cart=m.body('cart_base').id;C=d.subtree_com[torso];mass=m.body_subtreemass[torso];Fg=np.zeros(3);Tg=np.zeros(3);Fm=np.zeros(3);Tm=np.zeros(3);Fo=np.zeros(3);To=np.zeros(3)
 for i,c in enumerate(d.contact):
  ids=[int(c.geom1),int(c.geom2)];names=[m.geom(g).name for g in ids];bodies=[int(m.geom_bodyid[g]) for g in ids];robot=[torso<=b<cart for b in bodies]
  if robot[0]==robot[1]:continue
  sg=1 if robot[1] else -1;f=np.zeros(6);mj.mj_contactForce(m,d,i,f);F=sg*c.frame.reshape(3,3).T@f[:3];T=sg*c.frame.reshape(3,3).T@f[3:];M=np.cross(c.pos-C,F)+T
  if any(n in names for n in ['ground','sole_ground_box']):Fg+=F;Tg+=M
  elif 'cart_bite_bar' in names:Fm+=F;Tm+=M
  else:Fo+=F;To+=M
 ac=np.zeros(3);hd=np.zeros(3)
 for b in range(torso,cart):
  if not m.body_mass[b]:continue
  Jp=np.zeros((3,m.nv));Jr=np.zeros_like(Jp);Dp=np.zeros_like(Jp);Dr=np.zeros_like(Jp);mj.mj_jac(m,d,Jp,Jr,d.xipos[b],b);mj.mj_jacDot(m,d,Dp,Dr,d.xipos[b],b);a=Jp@d.qacc+Dp@d.qvel;w=Jr@d.qvel;alpha=Jr@d.qacc+Dr@d.qvel;R=d.ximat[b].reshape(3,3);I=R@np.diag(m.body_inertia[b])@R.T;ac+=m.body_mass[b]*a;hd+=I@alpha+np.cross(w,I@w)+np.cross(d.xipos[b]-C,m.body_mass[b]*a)
 ac/=mass;return {'scope':'same actualPRE state only, not fixed future mouthwrench','COM_acceleration_world_m_s2':ac.tolist(),'ground_force_world_N':Fg.tolist(),'ground_torque_about_COM_Nm':Tg.tolist(),'mouth_force_on_robot_world_N':Fm.tolist(),'mouth_torque_about_COM_Nm':Tm.tolist(),'other_external_contact_force_world_N':Fo.tolist(),'other_external_contact_torque_COM_Nm':To.tolist(),'gravity_world_N':(mass*m.opt.gravity).tolist(),'actual_Hdot_world_Nm':hd.tolist(),'force_balance_residual_N':(Fg+Fm+Fo+mass*m.opt.gravity-mass*ac).tolist(),'moment_balance_residual_Nm':(Tg+Tm+To-hd).tolist()}
