"""GPU training surrogates for the frozen keyboard acceptance constraints.

These per-step constraints are stricter/local approximations of trajectory
scoring. The independent native scorer remains the acceptance authority.
"""
import math
import torch


def yaw(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2].square() + q[:, 3].square()))


class SprintConstraints:
    def __init__(self, count, device):
        self.command = torch.full((count, 3), float('nan'), device=device)
        self.age = torch.zeros(count, device=device, dtype=torch.long)
        self.origin = torch.zeros((count, 2), device=device)
        self.heading = torch.zeros(count, device=device)
        self.forward_max = torch.zeros(count, device=device)
        self.rates = torch.zeros((10, count), device=device)
        self.speeds = torch.zeros((10, count), device=device)
        self.cursor = 0

    def reset(self, ids):
        self.command[ids] = float('nan')
        self.age[ids] = 0
        self.rates[:, ids] = 0.
        self.speeds[:, ids] = 0.

    def observe(self, requested, previous_position, previous_yaw, position, quaternion, velocity):
        current_yaw = yaw(quaternion)
        changed = (requested[:, :3] != self.command).any(-1)
        self.origin = torch.where(changed[:, None], previous_position[:, :2], self.origin)
        self.heading = torch.where(changed, previous_yaw, self.heading)
        self.forward_max = torch.where(changed, 0., self.forward_max)
        self.age = torch.where(changed, 1, self.age + 1)
        self.command.copy_(requested[:, :3])
        difference = current_yaw - previous_yaw
        self.rates[self.cursor] = torch.atan2(difference.sin(), difference.cos()) / .02
        self.speeds[self.cursor] = torch.linalg.vector_norm(velocity[:, :2], dim=-1)
        self.cursor = (self.cursor + 1) % 10
        moving = requested[:, 0] > .01
        turning = requested[:, 2].abs() > .05
        straight = moving & ~turning & (requested[:, 1].abs() < .01)
        turn = torch.where(moving & turning & (self.age >= 50),
                           .6 * requested[:, 2].abs() - requested[:, 2].sign() * self.rates.mean(0), 0.)
        delta = position[:, :2] - self.origin
        forward = self.heading.cos() * delta[:, 0] + self.heading.sin() * delta[:, 1]
        self.forward_max = torch.maximum(self.forward_max, forward)
        cross = -self.heading.sin() * delta[:, 0] + self.heading.cos() * delta[:, 1]
        path = torch.where(straight, cross.abs() - (.1 * self.forward_max).clamp_min(.05), 0.)
        angle = current_yaw - self.heading
        heading = torch.where(straight, torch.atan2(angle.sin(), angle.cos()).abs() - math.pi / 12, 0.)
        idle = (requested[:, :3].abs().amax(-1) < .01) & (self.age >= 100)
        stop = torch.where(idle, self.speeds.mean(0) - .05, 0.)
        return torch.stack((turn, path, heading, stop), dim=-1)
