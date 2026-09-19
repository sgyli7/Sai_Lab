"""Regressions at the actual command → physical step → reward boundary."""
import unittest
from types import SimpleNamespace

import numpy as np

from test_research import fake_world
from sim2sim.research.rewards import Objective
from sim2sim.research.schedules import roller_keyboard_commands
from sim2sim.research.world import World


class CommandObjectiveTests(unittest.TestCase):
    def step_reward(self, post_yaw, objective):
        w = fake_world('roller')
        w.roller_contract = True
        w.roller_target_yaw = 0.
        w.features['yaw'] = 1.
        command = roller_keyboard_commands('roller_turn_brake')[200]
        w.command = lambda: command.copy()
        w.backend_name = 'godot'
        w.backend = SimpleNamespace(send_step=lambda *a, **k: None)
        w.pending_ball = None
        reward = Objective(w, roller_objective=objective)
        World.send(w, np.zeros(14))
        w.features['yaw'] = post_yaw
        return reward.compute()[2]['heading']

    def test_requested_left_turn_rewards_left_response_after_step(self):
        self.assertGreater(self.step_reward(1.05, 'command_heading_v1'),
                           self.step_reward(.95, 'command_heading_v1'))

    def test_legacy_objective_is_retained_as_explicit_counterexample(self):
        self.assertLess(self.step_reward(1.05, 'legacy'),
                        self.step_reward(.95, 'legacy'))

    def test_zero_relative_error_holds_present_heading_not_reset_heading(self):
        w = fake_world('roller'); w.roller_contract = True
        w.roller_target_yaw = 0.; w.executed_heading_target = 1.
        w.features['yaw'] = 1.; w.executed_command[:] = 0.
        reward = Objective(w, roller_objective='command_heading_v1')
        self.assertAlmostEqual(reward.compute()[2]['heading'], 3.)

    def test_stop_task_rewards_sustained_hold_once_and_keeps_episode_running(self):
        w=fake_world('roller');w.roller_contract=True;w.roller_target_yaw=0.
        w.executed_heading_target=0.;w.executed_command[0]=-.5
        reward=Objective(w,roller_objective='stop_hold_v1');bonuses=[]
        for i in range(1,101):
            w.t=i*.02
            _,terminal,terms=reward.compute()
            self.assertFalse(terminal)
            bonuses.append(terms['stop_confirmed'])
            if i<59:self.assertEqual(terms['stop_confirmed'],0.)
        self.assertEqual(sum(bonuses),20.)
        self.assertEqual(terms['stop_hold'],8.)
        w.features['vel'][0]=.3
        for i in range(101,113):w.t=i*.02;terms=reward.compute()[2]
        self.assertEqual(terms['stop_hold'],0.)


if __name__ == '__main__':
    unittest.main()
