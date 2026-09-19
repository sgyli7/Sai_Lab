import math
import unittest
import torch
from sim2sim.research.kinematic_observer import KinematicObserver


class KinematicObserverTest(unittest.TestCase):
    def test_constant_motion_has_the_games_filter_lag(self):
        observer = KinematicObserver(1, 'cpu', dtype=torch.float64)
        for i in range(1, 9):
            angle = .5 * i * .005
            quat = torch.tensor([[math.cos(angle / 2), 0., 0., math.sin(angle / 2)]], dtype=torch.float64)
            observer.update(torch.full((1, 14), 2. * i * .005, dtype=torch.float64), quat)
            factor = 1. - math.exp(-i)
            torch.testing.assert_close(observer.qd, torch.full((1, 14), 2. * factor, dtype=torch.float64))
            torch.testing.assert_close(observer.angular_local(quat), torch.tensor([[0., 0., .5 * factor]], dtype=torch.float64))

    def test_reset_does_not_invent_velocity_at_a_teleport(self):
        observer = KinematicObserver(2, 'cpu')
        q = torch.full((2, 14), .4)
        quat = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
        observer.reset(torch.arange(2), q, quat)
        observer.update(q, -quat)
        self.assertEqual(float(observer.qd.abs().max()), 0.)
        self.assertEqual(float(observer.omega_world.abs().max()), 0.)

    def test_joint_wrap_uses_shortest_angle(self):
        observer = KinematicObserver(1, 'cpu', dtype=torch.float64)
        q = torch.full((1, 14), math.pi - .001, dtype=torch.float64)
        quat = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
        observer.reset(torch.arange(1), q, quat)
        observer.update(torch.full_like(q, -math.pi + .001), quat)
        expected = .002 / .005 * (1. - math.exp(-1.))
        torch.testing.assert_close(observer.qd, torch.full_like(q, expected))
