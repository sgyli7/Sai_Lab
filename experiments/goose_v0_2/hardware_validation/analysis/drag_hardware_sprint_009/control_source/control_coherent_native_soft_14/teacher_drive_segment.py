"""Private Teacher50Hz packet /200Hz evaluation, original drive limits.
No integration. Published policy/runtime untouched. Reference-only packet
contains this known20ms plan; no future actual observations are used.
"""
import numpy as np
from scipy.interpolate import BPoly

def reference_command_derivatives(curve,t):
 base=getattr(curve,'plan',curve);end=float(base.time[-1]);t=float(np.clip(t,0,end))
 if not hasattr(curve,'duration') or t>=curve.duration+2e-5:
  r=base.sample(t);return r['qref18'],r['qvel'][base._vidx()],r['qacc'][base._vidx()]
 def pos(x):
  x=float(np.clip(x,0,end));s=curve.sample(x);return s['qref18']
 q=pos(t)
 if t<=1e-12:return q,np.zeros(18),np.zeros(18)
 h=min(1e-5,t/2,(end-t)/2 if end>t else 1e-5);vm=(q-pos(t-h))/h;vp=(pos(t+h)-q)/h
 return q,.5*(vp+vm),(vp-vm)/h

def make_packet(curve,command_time):
 base=getattr(curve,'plan',curve);end=float(base.time[-1]);knots=sorted(set([float(command_time),float(command_time+.01),float(command_time+.02)]+([end] if command_time<end<command_time+.02 else [])));segments=[]
 for lo,hi in zip(knots[:-1],knots[1:]):
  vals=[]
  for t in [lo,hi]:
   if t>end+1e-12:q=base.sample(end)['qref18'];qd=np.zeros(18);qdd=np.zeros(18)
   else:q,qd,qdd=reference_command_derivatives(curve,t)
   vals.append([q,qd,qdd])
  # Terminal input derivatives are tiny finite-difference remnants; after the
  # declaredEND stop holds position, no extrapolation or motion-time extension.
  if lo>=end-1e-12:vals=[[vals[0][0],np.zeros(18),np.zeros(18)],[vals[0][0],np.zeros(18),np.zeros(18)]]
  p=BPoly.from_derivatives([lo,hi],vals);segments.append(dict(start_s=lo,end_s=hi,bernstein_q18=p.c[:,0,:].tolist()))
 return dict(command_time_s=float(command_time),valid_until_s=float(command_time+.02),segments=segments,front6_qd0_FF0=True,FF_hold_Hz=50,drive_Hz=200,physics_Hz=1000,known_reference_only=True)

def sample_packet(packet,drive_time):
 if not packet['command_time_s']-1e-10<=drive_time<packet['valid_until_s']+1e-10:raise ValueError('drive outside current50Hzpacket')
 ss=next((s for s in packet['segments'] if s['start_s']-1e-10<=drive_time<s['end_s']-1e-10),packet['segments'][-1]);p=BPoly(np.array(ss['bernstein_q18'])[:,None,:],[ss['start_s'],ss['end_s']]);q=np.array(p(drive_time));qd=np.array(p.derivative()(drive_time));rawderivative=qd.copy();qd[:6]=0.
 return q,qd,dict(q_reference_derivative18=rawderivative.tolist(),packet_start_s=packet['command_time_s'],drive_time_s=float(drive_time),front6_q_only=True)

def compute_drive_tick(env,packet,drive_time,current_target,actual_qpos,actual_qvel,held_FF18):
 """Return target/ctrl only; caller owns originalfine guards/integration.
 This function does not mutate target,data,q/v or model. Exactly one original
 source total limiter after qtarget slew+PD+heldFF.
 """
 qr,qd,meta=sample_packet(packet,drive_time);qr=np.clip(qr,env.ranges[:,0],env.ranges[:,1]);qd=np.clip(qd,-env.slew_speed,env.slew_speed);ff=np.clip(np.asarray(held_FF18,float),-env.cont,env.cont)
 if np.max(abs(ff[:6]))>1e-12:raise ValueError('front6qonlyFF0 for this candidate')
 target=np.asarray(current_target,float)+np.clip(qr-current_target,-env.slew_speed*.005,env.slew_speed*.005);pd=env.kp*(target-np.asarray(actual_qpos)[env.qidx])+env.kd*(qd-np.asarray(actual_qvel)[env.vidx]);raw=pd+ff;total=env.limit_torque(raw,np.asarray(actual_qvel)[env.drive_vidx]);meta.update(raw_q_ref18=qr.tolist(),dispatch_qd18=qd.tolist(),actual_slewed_target18=target.tolist(),PD18=pd.tolist(),held_FF18=ff.tolist(),requested_total18=raw.tolist(),limited_total18=total.tolist(),source_limiter_calls=1,no_live_state_write=True)
 return target,total,meta
