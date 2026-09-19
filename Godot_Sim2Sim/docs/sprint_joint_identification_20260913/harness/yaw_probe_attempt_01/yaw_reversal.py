"""Python reference for the diagnostic-only sprint turn reversal contract."""
import numpy as np
from sim2sim.motion_control import MotionControl

class YawReversalControl(MotionControl):
    def reset(self):
        super().reset()
        self.previous_request=0.;self.filtered_turn=0.;self.turn_step=0.;self.transitioning=False

    def command(self,command,state,skill,dt=.02):
        out=super().command(command,state,skill,dt)
        duration=float(self.settings.get('sprint_yaw_reversal_s',0.));requested=float(np.asarray(command,np.float32)[2])
        if skill!='sprint' or duration<=0 or abs(requested)<=.05:
            self.previous_request=requested if skill=='sprint' else 0.;self.filtered_turn=requested;self.transitioning=False
            return out
        if self.previous_request*requested<0 and abs(self.previous_request)>.05:
            self.transitioning=True;self.turn_step=abs(requested-self.filtered_turn)*dt/duration
        if self.transitioning:
            difference=requested-self.filtered_turn
            self.filtered_turn+=np.sign(difference)*min(abs(difference),self.turn_step)
            if abs(self.filtered_turn-requested)<1e-9:self.transitioning=False
            out[2]=self.filtered_turn
        else:self.filtered_turn=requested
        self.previous_request=requested
        return out
