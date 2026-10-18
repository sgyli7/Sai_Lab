"""Synthetic CPU accounting fixtures cannot become actual candidates."""
import copy
from pathlib import Path
import unittest

from experiments.lingbot_vla2_g1.data import OWN
from experiments.lingbot_vla2_g1.robust_training_v5 import PHASE_PROBABILITIES
from experiments.lingbot_vla2_g1.robust_training_v7 import SOURCE_PROBABILITIES, RECOVERY_PHASE_PROBABILITIES, AUGMENTATION
from experiments.lingbot_vla2_g1.round07_selection import roles_from_progress


def validation(loss):
    return {'complete': True, 'noise_seed': 42, 'episodes': 10, 'chunks': 190, 'frames': 9240,
            'flow_loss': loss, 'total_loss': loss + .001}


def fixture(improves=False):
    final = OWN / 'synthetic_R7' / ('selected' if improves else 'final')
    selected = final if improves else OWN / 'training_round02/selected'
    last = validation(.164 if improves else .167)
    d = {'schema': 'lingbot_vla2_g1_training_task_round07_v1', 'synthetic': True,
         'status': 'completed', 'overall_robot_goal_completed': False,
         'optimizer_steps': 3, 'initial_total_optimizer_updates': 2040,
         'model_weight_lineage_optimizer_updates': 2043,
         'training_source_weights': SOURCE_PROBABILITIES, 'full_teacher_phase_probabilities': PHASE_PROBABILITIES,
         'state_augmentation': AUGMENTATION,
         'recovery_teacher_phase_probabilities': RECOVERY_PHASE_PROBABILITIES,
         'former_nonformal_development_scene_explicitly_reclassified_TRAIN': True,
         'fresh_independent_development_cohort_required': True, 'baseline_validation': validation(.1667),
         'source_best_reference_delta': str(OWN / 'training_round02/selected'),
         'new_source_best_permanent_path': str(OWN / 'synthetic_R7_permanent_source_best'),
         'source_best_reference_validation': validation(.165), 'selected_step': 3 if improves else 0,
         'selected_validation': last if improves else validation(.165), 'last_validation': last,
         'final_validation_complete': True, 'selected_delta': str(selected), 'final_delta': str(final),
         'physical_candidate_step': 3, 'physical_candidate_delta': str(final), 'final_weights_saved': True,
         'physical_candidate_is_source_validation_best': improves,
         'source_validation_best_origin': 'round07' if improves else 'inherited_R2',
         'task_TRAIN_diagnostic_is_held_out': False,
         'training_draw_counts': {'original80TRAIN': 2, 'carry_prefixes': 2, 'full_scripted_TRAIN': 8, 'recovery_scripted_TRAIN': 12},
         'carry_family_draw_counts': {'one': 1, 'two': 1}, 'carry_phase_draw_counts': {'one': 1, 'two': 1},
         'full_teacher_phase_draw_counts': {'approach_grasp': 2, 'transport': 4, 'placement': 2},
         'recovery_teacher_phase_draw_counts': {'approach_grasp': 1, 'transport': 1, 'alignment': 5, 'lower_release': 5},
         'recovery_family_draw_counts': {'prior_R5_recovery04': 6, 'new_R6_recovery08': 6},
         'state_augmentation_counts': {'original80TRAIN': {'draws': 2, 'eligible': 0, 'augmented': 0},
             'carry_prefixes': {'draws': 2, 'eligible': 2, 'augmented': 1},
             'full_scripted_TRAIN': {'draws': 8, 'eligible': 8, 'augmented': 4},
             'recovery_scripted_TRAIN': {'draws': 12, 'eligible': 12, 'augmented': 6}}}
    events = [{'event': 'training_started', 'step': 0},
              {'event': 'validation_complete', 'step': 0, 'validation': d['baseline_validation'], 'selected': False},
              *[{'event': 'optimizer_step', 'step': step} for step in (1, 2, 3)],
              {'event': 'training_stopped', 'step': 3},
              {'event': 'validation_complete', 'step': 3, 'validation': last, 'selected': improves},
              {'event': 'training_completed', 'step': 3, 'selected_step': d['selected_step'],
               'selected_delta': str(selected), 'final_delta': str(final)}]
    return d, events


class RoleTests(unittest.TestCase):
    def test_inherited_best_and_new_final_are_distinct(self):
        result, events = fixture()
        best = roles_from_progress(result, events)
        self.assertEqual(best.step, 0)
        self.assertNotEqual(result['selected_delta'], result['physical_candidate_delta'])

    def test_strictly_improved_final_can_have_both_roles(self):
        result, events = fixture(True)
        self.assertEqual(roles_from_progress(result, events).step, 3)

    def test_no_silent_source_best_or_final_relabelling(self):
        result, events = fixture()
        bad = copy.deepcopy(events)
        bad[-2]['selected'] = True
        with self.assertRaisesRegex(ValueError, 'strict global'):
            roles_from_progress(result, bad)
        bad = copy.deepcopy(result)
        bad['physical_candidate_step'] = 0
        with self.assertRaisesRegex(ValueError, 'Actual final'):
            roles_from_progress(bad, events)

    def test_missing_update_or_augmentation_mismatch_is_rejected(self):
        result, events = fixture()
        missing = [event for event in events if not (event['event'] == 'optimizer_step' and event['step'] == 2)]
        with self.assertRaisesRegex(ValueError, 'Contiguous actual'):
            roles_from_progress(result, missing)
        result['state_augmentation_counts']['original80TRAIN']['augmented'] = 1
        with self.assertRaisesRegex(ValueError, 'perturbation counts'):
            roles_from_progress(result, events)


if __name__ == '__main__':
    unittest.main()
