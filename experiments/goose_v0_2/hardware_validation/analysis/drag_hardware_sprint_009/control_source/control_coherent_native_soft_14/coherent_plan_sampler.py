"""009 bounded C2 sampler: existing492 nodes, same times, no dynamics/IK.
Freejoint rotations MUST be fixed-axis about their first orientation (this
reference is root-roll and constant cart/cargo). Native local angular tangent
is preserved; quaternion components are never splined. Reject other paths.
"""
from pathlib import Path
import hashlib
import numpy as np
import mujoco as mj
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
class CoherentPlan:
 def __init__(self,model,reference_npz,qidx=None):
  self.model=model;self.path=Path(reference_npz);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest();z=np.load(self.path);self.time=z['time'].copy();self.nodes=z['qpos'].copy();v=z['qvel'].copy();a=z['qacc'].copy();self.qidx=None if qidx is None else np.asarray(qidx,int);self.bias18=None if qidx is None else z['qref18'][0]-self.nodes[0,self.qidx];self.rotation_charts=[];chart=np.zeros((len(self.time),model.nv))
  for j in range(model.njnt):
   q=int(model.jnt_qposadr[j]);d=int(model.jnt_dofadr[j]);typ=int(model.jnt_type[j])
   if typ==int(mj.mjtJoint.mjJNT_FREE):
    chart[:,d:d+3]=self.nodes[:,q:q+3];quats=self.nodes[:,q+3:q+7];R=Rotation.from_quat(quats[:,[1,2,3,0]]);R0=R[0];rv=(R0.inv()*R).as_rotvec();norm=np.linalg.norm(rv,axis=1);imax=int(np.argmax(norm));axis=np.array([1.,0.,0.]) if norm[imax]<1e-12 else rv[imax]/norm[imax];angle=rv@axis;res=float(np.max(np.linalg.norm(rv-angle[:,None]*axis,axis=1)));vperp=float(np.max(np.linalg.norm(v[:,d+3:d+6]-(v[:,d+3:d+6]@axis)[:,None]*axis,axis=1)));aperp=float(np.max(np.linalg.norm(a[:,d+3:d+6]-(a[:,d+3:d+6]@axis)[:,None]*axis,axis=1)))
    if max(res,vperp,aperp)>1e-8 or max(abs(angle))>np.pi/2:raise ValueError(('reference outside bounded fixed-axis quaternion chart',j,res,vperp,aperp))
    chart[:,d+3:d+6]=angle[:,None]*axis;self.rotation_charts.append(dict(q=q,d=d,R0=R0,axis=axis,maximum_nonaxis_orientation_rad=res,maximum_nonaxis_velocity=vperp,maximum_nonaxis_acceleration=aperp))
   elif typ in [int(mj.mjtJoint.mjJNT_HINGE),int(mj.mjtJoint.mjJNT_SLIDE)]:chart[:,d]=self.nodes[:,q]
   else:raise ValueError('ball joints not in admitted reference')
  self.poly=BPoly.from_derivatives(self.time,[[p,vi,ai] for p,vi,ai in zip(chart,v,a)],extrapolate=False);self.derivative=self.poly.derivative(1);self.second=self.poly.derivative(2)
  self.qref_bias_varies=0. if qidx is None else float(np.max(abs(z['qref18']-self.nodes[:,self.qidx]-self.bias18)))
  if self.qref_bias_varies>1e-10:raise ValueError('nonconstant command bias requires separately admitted reference')
 def sample(self,t):
  if not self.time[0]-1e-12<=t<=self.time[-1]+1e-12:raise ValueError('outside certified plan; no extrapolation')
  t=float(np.clip(t,self.time[0],self.time[-1]));p=np.asarray(self.poly(t));v=np.asarray(self.derivative(t));a=np.asarray(self.second(t));q=np.zeros(self.model.nq)
  for j in range(self.model.njnt):
   qi=int(self.model.jnt_qposadr[j]);di=int(self.model.jnt_dofadr[j]);typ=int(self.model.jnt_type[j])
   if typ==int(mj.mjtJoint.mjJNT_FREE):q[qi:qi+3]=p[di:di+3]
   else:q[qi]=p[di]
  for c in self.rotation_charts:
   quat=(c['R0']*Rotation.from_rotvec(p[c['d']+3:c['d']+6])).as_quat();q[c['q']+3:c['q']+7]=quat[[3,0,1,2]]
  return dict(time=t,qpos=q,qvel=v,qacc=a,qref18=None if self.qidx is None else q[self.qidx]+self.bias18,reference_npz_sha256=self.sha256,semantics='same fixed-axis native C2 quintic Hermite q(t), analytic native qdot/qddot; no actual-root mixture; free object reference never written to live data')
 def command(self,t):
  x=self.sample(t);qd=x['qvel'][self._vidx()].copy();qd[:6]=0.;return x['qref18'],qd,x
 def _vidx(self):
  if self.qidx is None:raise ValueError('qidx required for command')
  result=[]
  for qi in self.qidx:
   j=int(np.where(self.model.jnt_qposadr==qi)[0][0]);result.append(int(self.model.jnt_dofadr[j]))
  return np.array(result)
