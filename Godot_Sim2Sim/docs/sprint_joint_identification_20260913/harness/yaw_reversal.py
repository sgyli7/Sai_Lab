"""Input-layer reference; preserve original ramps except sprint reversal."""
import numpy as np
from sim2sim.play_input import TwistRamp

class YawReversalRamp(TwistRamp):
    def __init__(self,brain,duration):
        super().__init__(accel=brain.lim.accel,decel=brain.lim.decel)
        self.brain=brain;self.duration=float(duration);self.transitioning=False

    def reset(self):
        super().reset();self.transitioning=False

    def step(self,target,dt):
        current=float(self.vel[2]);requested=float(np.asarray(target,np.float32)[2])
        out=super().step(target,dt)
        if not self.brain.sprinting or self.duration<=0 or abs(requested)<=.05:
            self.transitioning=False
            return out
        if current*requested<0 and abs(current)>.05:self.transitioning=True
        if self.transitioning:
            rate=2*float(self.brain.lim.sprint_vmax_ang)/self.duration
            difference=requested-current
            self.vel[2]=current+np.sign(difference)*min(abs(difference),rate*dt)
            if abs(float(self.vel[2])-requested)<1e-7:self.transitioning=False
        return self.vel
