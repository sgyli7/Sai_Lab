"""A final physical candidate is a distinct role from original-validation best."""
import copy
import unittest

from experiments.lingbot_vla2_g1.data import SPLIT_SHA
from experiments.lingbot_vla2_g1.round04_selection import roles_from_progress


def example():
    baseline = {'flow_loss': .1, 'total_loss': .1, 'complete': True,
                'frames': 9240, 'chunks': 190, 'episodes': 10}
    worse = {**baseline, 'flow_loss': .2, 'total_loss': .2}
    result = {'schema': 'lingbot_vla2_g1_training_task_round04_v1', 'status': 'completed',
              'test_opened': False, 'split_sha256': SPLIT_SHA, 'optimizer_steps': 2,
              'baseline_validation': baseline, 'selected_step': 0, 'selected_validation': baseline,
              'final_weights_saved': True, 'physical_candidate_step': 2,
              'physical_candidate_delta': '/CPU_fixture/final', 'final_delta': '/CPU_fixture/final',
              'task_TRAIN_diagnostic_is_held_out': False, 'overall_robot_goal_completed': False,
              'training_draw_counts': {'original80TRAIN': 4, 'carry_prefixes': 4, 'full_scripted_TRAIN': 8},
              'carry_family_draw_counts': {'old': 2, 'new': 2},
              'carry_phase_draw_counts': {'grasp': 2, 'carry': 2},
              'full_teacher_phase_draw_counts': {'grasp': 4, 'carry': 2, 'place': 2}}
    events = [{'event': 'validation_complete', 'step': 0, 'validation': baseline},
              {'event': 'optimizer_step', 'step': 1}, {'event': 'optimizer_step', 'step': 2},
              {'event': 'validation_complete', 'step': 2, 'validation': worse},
              {'event': 'training_completed', 'step': 2, 'final_delta': '/CPU_fixture/final'}]
    return result, events


class CandidateRoleTests(unittest.TestCase):
    def test_worse_source_validation_final_is_admitted_as_separate_role(self):
        result, events = example()
        original = copy.deepcopy(result)
        source = roles_from_progress(result, events)
        self.assertEqual(source['step'], 0)
        self.assertEqual(result, original)

    def test_source_selection_cannot_be_relabelled_as_final(self):
        result, events = example()
        result['selected_step'] = 2
        result['selected_validation'] = events[-2]['validation']
        with self.assertRaisesRegex(ValueError, 'validation minimum'):
            roles_from_progress(result, events)

    def test_partial_update_cannot_be_called_actual_final(self):
        result, events = example()
        result['physical_candidate_step'] = 1
        with self.assertRaisesRegex(ValueError, 'actual optimizer boundary'):
            roles_from_progress(result, events)


if __name__ == '__main__':
    unittest.main()
