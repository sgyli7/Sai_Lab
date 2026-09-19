"""Declared yaw-drift memory computed only from existing IMU observations.

The extra feature is an integral of world-vertical angular velocity during
straight motion/idle. Commanded turns clear the reference. It never changes
actions, requested commands, or simulator state; the neural actor decides
whether and how to react. obs[55] is repurposed and must be documented.
"""
import numpy as np


def has_yaw_memory(metadata):
    mode=metadata.get('sim2sim_yaw_memory','')
    if mode not in ('','gyro_vertical_integral_v1'):raise ValueError('Unknown yaw-memory contract')
    return bool(mode)


class YawDriftMemory:
    def __init__(self):self.reset()

    def reset(self):
        self.error=0.;self.previous_turn=False;self.started=False;self.stamp=None

    def observe(self,obs,stamp=None,dt=.02):
        x=np.asarray(obs,np.float32).copy()
        if x.shape!=(61,):raise ValueError('Yaw memory requires one 61D observation')
        turn=abs(float(x[50]))>.05
        if not self.started:
            self.started=True;self.previous_turn=turn
        elif stamp is None or stamp!=self.stamp:
            if turn or self.previous_turn:self.error=0.
            else:
                # gravity_local = -R_world_from_body[2,:]. This dot product
                # uses the same gyro/gravity already present in obs[0:6].
                vertical_rate=-float(x[3:6]@x[0:3])
                self.error=float(np.clip(self.error-vertical_rate*dt,-.5,.5))
            self.previous_turn=turn
        self.stamp=stamp
        x[55]=self.error
        return x
