"""Training-only CaT return weighting, independent of physical episode resets.

Reference: Chane-Sane et al., arXiv:2403.18765, equations 6/7 and algorithm 1.
The supplied constraints are project-specific; this is not a safety guarantee.
"""
import torch


class ConstraintReturns:
    def __init__(self, count, device):
        self.scale = torch.zeros(count, device=device)

    def probability(self, violations, maximum):
        positive = violations.clamp_min(0.)
        peak = positive.flatten(0, -2).amax(dim=0)
        self.scale = .95 * self.scale + .05 * peak
        return (positive / self.scale.clamp_min(1e-6)).clamp(0., 1.).amax(-1) * maximum


def advantages(rewards, values, next_value, done, probability, gamma=.99, lam=.95):
    """Constraint probability discounts rewards/returns but never resets physics."""
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in reversed(range(len(rewards))):
        survival = 1. - probability[step]
        alive = (~done[step]).to(rewards.dtype) * survival
        delta = survival * rewards[step].clamp_min(0.) + gamma * next_value * alive - values[step]
        gae = delta + gamma * lam * alive * gae
        advantage[step] = gae
        next_value = values[step]
    return advantage, advantage + values
