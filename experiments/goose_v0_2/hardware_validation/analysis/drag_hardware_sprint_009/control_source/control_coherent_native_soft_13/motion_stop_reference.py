"""Same command clock: .2s join INSIDE4.90625motion, then3s hold.
Reference-only nominal static stop requires its own physical inverse certificate.
"""
import hashlib
from pathlib import Path
import numpy as np
class MotionStopReference:
 def __init__(self,entry,hold_s=3.):
  self.entry=entry;self.plan=entry.plan;self.model=entry.model;self.duration=entry.duration;self.motion_end=float(self.plan.time[-1]);self.hold_s=float(hold_s)
 def sample(self,t):
  if t<0 or t>self.motion_end+self.hold_s+1e-10:raise ValueError('outside complete command/hold clock')
  if t<=self.motion_end:return self.entry.sample(t)
  r=self.plan.sample(self.motion_end);r.update(time=float(t),qvel=np.zeros(self.model.nv),qacc=np.zeros(self.model.nv),qd_ref18=np.zeros(18),phase='nominal_stop_hold',stop_hold_inverse_certificate_required=True);return r
 def certificate_sampler_sha256(self,t):
  if t<=self.motion_end:return self.entry.certificate_sampler_sha256(t)
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 def join_FF(self,t,ff):return self.entry.join_FF(t,ff)
