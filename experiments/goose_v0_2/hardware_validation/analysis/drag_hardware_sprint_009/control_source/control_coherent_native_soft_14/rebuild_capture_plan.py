"""Single same-plan geometric rebuild to observed capture soleXY anchors.
No new gait parameters/search/integration. Reference scratch only. Original
flat native sole rotations+yaw and originalZ, same relative foot paths/times.
"""
from pathlib import Path
import json,numpy as np,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import complete_cycle_plan as plan
import foot_ik_reference as ik
LOCAL=np.array([.13750001972253326,0,-.03849996489687674])
def rebuild(env,base,alignment,capture_context,output):
 m=env.model;d=mj.MjData(m);d.qpos[:]=base.sample(0)['qpos'];mj.mj_forward(m,d);sides=['right','left'];coldfeet={s:d.geom_xpos[m.geom(s+'_actual_sole').id].copy() for s in sides};flat={s:alignment.R.as_matrix()@d.geom_xmat[m.geom(s+'_actual_sole').id].reshape(3,3) for s in sides};actual=mj.MjData(m);actual.qpos[:]=capture_context['actual_qpos'];mj.mj_forward(m,actual);observed={s:actual.geom_xpos[m.geom(s+'_actual_sole').id].copy() for s in sides};seed=base.sample(0)['qpos'][env.qidx].copy();qs=[];qrefs=[];checks=[]
 for t in base.time:
  r=alignment.sample(float(t));d.qpos[:]=r['qpos'];targets,_=plan.feet_targets(float(t),coldfeet);errors={}
  for side in sides:
   delta=targets[side]-coldfeet[side];world=observed[side].copy()+alignment.R.apply(delta);world[2]=coldfeet[side][2]+delta[2];ii=np.array([env.names.index(side+'_'+n) for n in ik.LEG_JOINT_SUFFIXES]);sol=ik.solve_foot_ik(m,d,side,world,flat[side],seed[ii],max_nfev=40);seed[ii]=sol['q_target6_rad'];errors[side]=sol['equivalent_IK_error_m']
  worldrod=alignment.rod_anchor+alignment.R.apply([-plan.travel(float(t))[0],0,0]);head=m.body('head_roll').id
  def res(z):
   d.qpos[env.qidx[:5]]=z;mj.mj_kinematics(m,d);H=d.xmat[head].reshape(3,3);ang=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[d.xpos[head]+H@LOCAL-worldrod,.08*(ang[1]-.174532925),.08*ang[2]]
  so=least_squares(res,np.clip(seed[:5],env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),bounds=(env.ranges[:5,0]+.05,env.ranges[:5,1]-.05),max_nfev=40,ftol=1e-9,gtol=1e-9,xtol=1e-9);seed[:5]=so.x;d.qpos[env.qidx[:5]]=so.x;errors['neck']=float(np.linalg.norm(res(so.x)));q=d.qpos.copy();qr=q[env.qidx].copy();qr[:5]+=alignment.front_bias;qr[5]=0.;qs.append(q);qrefs.append(qr);checks.append(dict(t=float(t),IK=errors,minimum_leg_ROM_margin=float(min(np.minimum(qr[6:]-env.ranges[6:,0],env.ranges[6:,1]-qr[6:])))))
 qs=np.array(qs);qrefs=np.array(qrefs);vi=[]
 for i,dt in enumerate(np.diff(base.time)):
  vv=np.zeros(m.nv);mj.mj_differentiatePos(m,vv,float(dt),qs[i],qs[i+1]);vi.append(vv)
 vi=np.array(vi);vel=np.vstack([vi[0],.5*(vi[:-1]+vi[1:]),vi[-1]]);acc=np.gradient(vel,base.time,axis=0,edge_order=2);rates=np.diff(qrefs,axis=0)/np.diff(base.time)[:,None];rat=abs(rates[:,6:])/env.slew_speed[6:];worst=np.unravel_index(np.argmax(rat),rat.shape)
 np.savez_compressed(output,time=base.time,qpos=qs,qref18=qrefs,qvel_interval=vi,qvel=vel,qacc=acc,reference_qref_velocity=rates)
 report=dict(physics_steps=0,samples=len(qs),same_original_plan_parameters=True,actual_anchor_XY_rebuild=True,flat_native_sole_R_preserved_with_frame_yaw=True,sole_Z_compression_difference_not_hidden={s:float(coldfeet[s][2]-observed[s][2]) for s in sides},max_IK=max(max(x['IK'].values()) for x in checks),minimum_leg_ROM_margin=min(x['minimum_leg_ROM_margin'] for x in checks),max_leg_reference_source_slew_ratio=float(rat[worst]),limiting_axis=env.names[worst[1]+6],limiting_time=float(base.time[worst[0]]),checks=checks,observed_anchors={s:x.tolist() for s,x in observed.items()})
 return report
