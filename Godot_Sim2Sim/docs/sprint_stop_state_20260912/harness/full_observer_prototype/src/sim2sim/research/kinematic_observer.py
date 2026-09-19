"""Training-side counterpart of Jolt's pose-difference velocity observer.

The game reads this filtered velocity for both policy observations and motor
damping. Solver velocities remain separate physical state.
"""
import math
import torch
from .torch_walking import inverse_rotate


class KinematicObserver:
    def __init__(self, count, device, *, dtype=torch.float32, dt=.005, tau=.005):
        self.dt = dt
        self.alpha = 1. - math.exp(-dt / tau)
        self.q = torch.zeros((count, 14), device=device, dtype=dtype)
        self.quat = torch.zeros((count, 4), device=device, dtype=dtype)
        self.quat[:, 0] = 1.
        self.qd = torch.zeros_like(self.q)
        self.omega_world = torch.zeros((count, 3), device=device, dtype=dtype)

    def reset(self, ids, q, quat):
        self.q[ids] = q[ids]
        self.quat[ids] = quat[ids]
        self.qd[ids] = 0.
        self.omega_world[ids] = 0.

    def update(self, q, quat):
        delta = torch.atan2((q - self.q).sin(), (q - self.q).cos())
        self.qd.lerp_(delta / self.dt, self.alpha)
        # new * conjugate(old): shortest rotation in world coordinates.
        w = (quat * self.quat).sum(-1, keepdim=True)
        v = (-quat[:, :1] * self.quat[:, 1:]
             + self.quat[:, :1] * quat[:, 1:]
             - torch.cross(quat[:, 1:], self.quat[:, 1:], dim=-1))
        sign = torch.where(w < 0., -1., 1.)
        w = w * sign
        v = v * sign
        length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        rate = v * (2. * torch.atan2(length, w) / (length.clamp_min(1e-10) * self.dt))
        rate = torch.where(length < 1e-10, 0., rate)
        self.omega_world.lerp_(rate, self.alpha)
        self.q.copy_(q)
        self.quat.copy_(quat)

    def angular_local(self, quat):
        return inverse_rotate(quat, self.omega_world)
