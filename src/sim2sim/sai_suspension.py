"""Bounded terrain-relative leg targets. Wheels and stair gait remain separate."""
import numpy as np
from sai_agent.control import FRONTS, SIDES, CROUCH_DROP


def leg_angles(down):
    beta = -FRONTS*np.arccos(np.clip((down**2-.09**2-.11**2)/(2*.09*.11),-1.,1.))
    theta = -np.arctan2(.11*np.sin(beta), .09+.11*np.cos(beta))
    return np.stack([-SIDES*theta,-SIDES*beta],axis=-1)


class Suspension:
    def __init__(self, parameters, apply_on_stairs=False):
        p = np.asarray(parameters, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < [0.,0.,.02]) or np.any(p > [1.2,1.2,.30]):
            raise ValueError('Invalid suspension gain/attitude/time-constant parameters')
        self.ground_gain, self.attitude_gain, self.tau = p
        self.apply_on_stairs = bool(apply_on_stairs)
        self.offset = np.zeros(4)

    def apply(self, result, state):
        if 'wheel_ground_heights' not in state or (result['stage']=='stairs' and not self.apply_on_stairs):
            self.offset.fill(0.)
            return result
        heights = np.asarray(state['wheel_ground_heights'],dtype=float)
        if heights.shape != (4,) or not np.isfinite(heights).all():
            raise ValueError('Invalid wheel ground heights')
        rotation = np.asarray(state['base_rotation_columns'],dtype=float).T
        # The released standing reference is retained; only differential support
        # correction is applied. Common ground height cancels, including spawn height.
        xy = np.array([[.115,.146],[.115,-.146],[-.115,.146],[-.115,-.146]])
        tilt = xy@rotation[2,:2]
        deviation = heights-heights.mean()
        desired = -self.ground_gain*deviation + self.attitude_gain*tilt
        # Do not invent a correction on perfectly flat terrain. A rolling incline
        # still has differential elevations; attitude feedback may track that plane.
        blend = np.clip(np.ptp(heights)/.004,0.,1.)
        desired *= blend
        desired = np.clip(desired,-.025,.020)
        dt=.02
        self.offset += np.clip((desired-self.offset)*(dt/(self.tau+dt)),-.003,.003)
        nominal = .172812737-CROUCH_DROP*float(result['effective_crouch'])
        down = np.clip(nominal+self.offset,.12,.195)
        delta = leg_angles(down)-leg_angles(np.full(4,nominal))
        target = np.asarray(result['target_leg']).copy()
        target[1::4] += delta[:,0]
        target[2::4] += delta[:,1]
        result = dict(result,target_leg=target.tolist(),suspension_offset_m=self.offset.tolist(),
                      suspension_parameters=[float(self.ground_gain),float(self.attitude_gain),float(self.tau)])
        return result
