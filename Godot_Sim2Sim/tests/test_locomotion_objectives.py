import unittest
import torch
from sim2sim.research.locomotion_objectives import command_tracking


class CommandTracking(unittest.TestCase):
    def test_overspeed_cannot_buy_a_missed_turn(self):
        command = torch.tensor([[.3, 0., .8]]).repeat(3, 1)
        velocity = torch.tensor([[.3, 0., 0.], [.6, 0., 0.], [.6, 0., 0.]])
        score = command_tracking(command, velocity, torch.tensor([.8, .4, .8]))
        self.assertAlmostEqual(float(score[0]), 1.5)
        self.assertGreater(float(score[0]), float(score[2]))
        self.assertGreater(float(score[2]), float(score[1]))

    def test_lateral_correction_is_part_of_the_command(self):
        command = torch.tensor([[.3, -.1, 0.]]).repeat(2, 1)
        velocity = torch.tensor([[.3, -.1, 0.], [.3, .1, 0.]])
        score = command_tracking(command, velocity, torch.zeros(2))
        self.assertAlmostEqual(float(score[0]), 1.5)
        self.assertGreater(float(score[0]), float(score[1]))

    def test_idle_requires_both_translation_and_rotation_to_stop(self):
        command = torch.zeros((3, 3))
        velocity = torch.tensor([[0., 0., 0.], [.2, 0., 0.], [0., 0., 0.]])
        score = command_tracking(command, velocity, torch.tensor([0., 0., .8]))
        self.assertTrue(bool((score[1:] < score[0]).all()))
        self.assertTrue(bool(((score >= 0.) & (score <= 1.5)).all()))
