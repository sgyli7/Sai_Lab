"""The curriculum must keep both source phases and preserve the base data seam."""
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import grasp_balanced_prefix as g
from experiments.lingbot_vla2_g1.data import SHARED


class GraspPrefixTests(unittest.TestCase):
    def test_boundary_comes_from_actual_admitted_TRAIN_event(self):
        actual = g.phase_identity(SHARED / 'prepared/carry_prefix_native201/manifest.json')
        self.assertEqual(actual['first_contact_lift_tick'], 152)
        self.assertEqual(actual['approach_and_stabilization_stop_exclusive'], 252)
        self.assertFalse(actual['online_policy_phase_input'])

    def test_both_phases_use_unchanged_source_sample_and_record_draws(self):
        value = object.__new__(g.GraspBalancedPrefix)
        value.boundary, value.states = 252, [None] * 1396
        value.phase_draw_counts = {'approach_and_grasp': 0, 'carry_remainder': 0}
        with patch.object(g.random, 'random', side_effect=[0., .99]), \
             patch.object(g.random, 'randrange', side_effect=[0, 1395]) as choose, \
             patch.object(g.CarryPrefix, 'sample', side_effect=lambda index: index) as sample:
            self.assertEqual(value.sample(10), 0)
            self.assertEqual(value.sample(10), 1395)
        self.assertEqual(choose.call_args_list[0].args, (252,))
        self.assertEqual(choose.call_args_list[1].args, (252, 1396))
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(value.phase_draw_counts, {'approach_and_grasp': 1, 'carry_remainder': 1})


if __name__ == '__main__':
    unittest.main()
