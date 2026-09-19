"""Batched training-side version of the declared walking motion contract.

Only the all-phase heading/path contract is supported. Real native trajectories
must pass the accompanying differential check before using it for learning.
"""
import torch


def rotate(q, v):
    t = 2. * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def inverse_rotate(q, v):
    return rotate(torch.cat((q[..., :1], -q[..., 1:]), dim=-1), v)


class WalkingControl:
    def __init__(self, count, settings, device):
        if settings.get('walk_heading_scope', 'all') != 'all':
            raise ValueError('GPU walking supports only the all-phase heading contract')
        self.settings = settings
        self.started = torch.zeros(count, device=device, dtype=torch.bool)
        self.moving = self.started.clone()
        self.path_started = self.started.clone()
        self.yaw_target = torch.zeros(count, device=device, dtype=torch.float64)
        self.path_yaw = self.yaw_target.clone()
        self.path_speed = self.yaw_target.clone()
        self.origin = torch.zeros((count, 2), device=device, dtype=torch.float64)

    def reset(self, ids):
        self.started[ids] = False
        self.moving[ids] = False
        self.path_started[ids] = False
        self.path_speed[ids] = 0.

    def command(self, requested, pos, quat, velocity, sprint):
        s = self.settings
        q = quat.double()
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        moving = (torch.linalg.vector_norm(requested[:, :2].double(), dim=1) > .01) | (requested[:, 2].abs() > .05)
        anchor = ~self.started
        if s.get('walk_reanchor_on_start', False):
            anchor |= moving & ~self.moving
        self.yaw_target = torch.where(anchor, yaw, self.yaw_target)
        self.started[:] = True
        self.moving = moving
        out = requested.clone()
        gain = float(s.get('walk_heading_gain', 0.))
        limit = float(s.get('walk_heading_limit', .3))
        if gain > 0:
            turning = requested[:, 2].abs() > .05
            self.yaw_target = torch.where(turning, yaw, self.yaw_target)
            error = self.yaw_target - yaw
            correction = (gain * torch.atan2(error.sin(), error.cos())).clamp(-limit, limit)
            out[:, 2] = torch.where(turning, requested[:, 2].double(), correction).float()
        pg = torch.where(sprint, float(s.get('walk_path_gain', 0.)),
                         float(s.get('walk_ordinary_path_gain', s.get('walk_path_gain', 0.))))
        straight = (requested[:, 0] > .01) & (requested[:, 1].abs() < .01) & (requested[:, 2].abs() < .05) & (pg > 0)
        begin = straight & ~self.path_started
        self.origin = torch.where(begin[:, None], pos[:, :2].double(), self.origin)
        self.path_yaw = torch.where(begin, yaw, self.path_yaw)
        self.path_speed = torch.where(begin, 0., self.path_speed)
        self.path_started = straight
        delta = pos[:, :2].double() - self.origin
        error = -self.path_yaw.sin() * delta[:, 0] + self.path_yaw.cos() * delta[:, 1]
        damping = float(s.get('walk_path_damping', 0.))
        if damping > 0:
            speed = -self.path_yaw.sin() * velocity[:, 0] + self.path_yaw.cos() * velocity[:, 1]
            self.path_speed = torch.where(straight, .8 * self.path_speed + .2 * speed, self.path_speed)
        pl = float(s.get('walk_path_limit', .15))
        correction = (-pg * error - damping * self.path_speed).clamp(-pl, pl)
        out[:, 1] = torch.where(straight, correction, out[:, 1].double()).float()
        ahead = float(s.get('walk_path_lookahead', 0.))
        if ahead > 0 and gain > 0:
            target = self.path_yaw - torch.atan2(error, torch.full_like(error, ahead))
            self.yaw_target = torch.where(straight, target, self.yaw_target)
            difference = target - yaw
            correction = (gain * torch.atan2(difference.sin(), difference.cos())).clamp(-limit, limit)
            out[:, 2] = torch.where(straight, correction, out[:, 2].double()).float()
        out[:, :2] = (out[:, :2].double() * float(s.get('walk_translation_scale', 1.))).float()
        return out
