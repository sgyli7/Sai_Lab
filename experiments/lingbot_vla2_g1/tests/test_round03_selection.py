"""A partial or post-hoc training metric cannot select a new checkpoint."""
import copy
import unittest

from experiments.lingbot_vla2_g1.round03_selection import select_from_progress
from experiments.lingbot_vla2_g1.data import SPLIT_SHA


def metric(value):
    return {'flow_loss': value, 'complete': True, 'frames': 9240, 'chunks': 190, 'episodes': 10}


class SelectionTests(unittest.TestCase):
    def fixture(self):
        a, b = metric(.16), metric(.1)
        result = {'schema': 'lingbot_vla2_g1_training_carry_round03_dual_v1', 'status': 'completed',
                  'test_opened': False, 'split_sha256': SPLIT_SHA, 'optimizer_steps': 2,
                  'baseline_validation': a, 'selected_step': 2, 'selected_validation': b}
        events = [{'event': 'validation_complete', 'step': 0, 'validation': a},
                  {'event': 'optimizer_step', 'step': 1}, {'event': 'optimizer_step', 'step': 2},
                  {'event': 'validation_complete', 'step': 2, 'validation': b},
                  {'event': 'training_completed', 'step': 2}]
        return result, events

    def test_lowest_complete_validation_is_selected(self):
        r, e = self.fixture()
        self.assertEqual(select_from_progress(r, e)['step'], 2)

    def test_incomplete_validation_and_missing_update_are_rejected(self):
        r, e = self.fixture()
        e[3]['validation'] = {**e[3]['validation'], 'complete': False}
        with self.assertRaisesRegex(ValueError, 'complete original'):
            select_from_progress(r, e)
        r, e = self.fixture()
        del e[1]
        with self.assertRaisesRegex(ValueError, 'Every real optimizer'):
            select_from_progress(r, e)

    def test_mismatched_selected_checkpoint_is_rejected(self):
        r, e = self.fixture()
        r['selected_step'] = 0
        with self.assertRaisesRegex(ValueError, 'minimum'):
            select_from_progress(r, e)


if __name__ == '__main__':
    unittest.main()
