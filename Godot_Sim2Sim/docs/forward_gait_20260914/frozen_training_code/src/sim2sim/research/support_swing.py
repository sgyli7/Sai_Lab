"""Contact-driven swing objective, with no external phase clock in the actor.

The 0.125--0.300 s swing window follows MicroDuck's upstream velocity recipe.
This experiment additionally requires one stance foot, an upright body and
forward progress: hovering/hopping in place and flight do not earn this term.
It is a gait-discovery hypothesis, not an independent acceptance metric.
"""
import torch


class SupportSwing:
    def __init__(self, count, device):
        self.air_time = torch.zeros((count, 2), device=device)
        self.contact = torch.ones((count, 2), device=device, dtype=torch.bool)
        self.clearance = torch.zeros((count, 2), device=device)

    def reset(self, ids):
        self.air_time[ids] = 0.
        self.contact[ids] = True
        self.clearance[ids] = 0.

    def update(self, contact, clearance=None, dt=.005):
        self.contact.copy_(contact)
        if clearance is not None:self.clearance.copy_(clearance)
        self.air_time.copy_(torch.where(contact, 0., self.air_time + dt))

    def reward(self, requested, forward_speed, tilt, height, sprint):
        window = (self.air_time > .125) & (self.air_time < .300) & (self.clearance > .005)
        single_support = self.contact.sum(-1) == 1
        valid = sprint & (requested[:, 0] > .01) & (tilt < .35) & (height > .08) & single_support
        progress = (forward_speed / requested[:, 0].clamp_min(.01)).clamp(0., 1.)
        return window.float().sum(-1) * valid.float() * progress
