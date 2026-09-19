import unittest
import torch
from sim2sim.research.constraint_returns import ConstraintReturns, advantages
from sim2sim.research.sprint_constraints import SprintConstraints


class ConstraintReturnTests(unittest.TestCase):
    def test_probability_cannot_hide_one_violation_behind_many_satisfied_constraints(self):
        normalizer = ConstraintReturns(3, 'cpu')
        x = torch.tensor([[[-3., 0., -1.], [-1., 2., -4.]]])
        p = normalizer.probability(x, .25)
        torch.testing.assert_close(p, torch.tensor([[0., .25]]))

    def test_fractional_termination_scales_current_reward_and_future_value(self):
        rewards = torch.tensor([[2.], [4.]])
        values = torch.zeros_like(rewards)
        done = torch.tensor([[False], [True]])
        p = torch.tensor([[.5], [0.]])
        advantage, returns = advantages(rewards, values, torch.tensor([100.]), done, p, gamma=1., lam=1.)
        torch.testing.assert_close(returns, torch.tensor([[3.], [4.]]))
        torch.testing.assert_close(advantage, returns)
        self.assertEqual(done.tolist(), [[False], [True]])

    def test_full_violation_terminates_learning_return_and_physical_terminal_never_bootstraps(self):
        rewards = torch.tensor([[2., 2.]])
        values = torch.tensor([[3., 3.]])
        done = torch.tensor([[False, True]])
        p = torch.tensor([[1., 0.]])
        _, returns = advantages(rewards, values, torch.tensor([100., 100.]), done, p)
        torch.testing.assert_close(returns, torch.tensor([[0., 2.]]))

    def test_turn_grace_expires_at_fifty_decisions_and_reset_restarts_it(self):
        constraints = SprintConstraints(1, 'cpu')
        command = torch.tensor([[.3, 0., .8]])
        pos = torch.zeros((1, 3))
        q = torch.tensor([[1., 0., 0., 0.]])
        for _ in range(49):
            cost = constraints.observe(command, pos, torch.zeros(1), pos, q, pos)
            self.assertEqual(float(cost[0, 0]), 0.)
        cost = constraints.observe(command, pos, torch.zeros(1), pos, q, pos)
        self.assertGreater(float(cost[0, 0]), .47)
        constraints.reset(torch.tensor([0]))
        cost = constraints.observe(command, pos, torch.zeros(1), pos, q, pos)
        self.assertEqual(float(cost[0, 0]), 0.)
