"""Bounded command tracking for the speed curriculum's inner policy learner."""
import torch


def command_tracking(command, local_velocity, yaw_rate):
    """CaT Table I option A; translation uses the actor's body-frame command."""
    linear_error = (command[:, :2] - local_velocity[:, :2]).square().sum(-1)
    angular_error = (command[:, 2] - yaw_rate).square()
    return torch.exp(-linear_error / .25) + .5 * torch.exp(-angular_error / .25)
