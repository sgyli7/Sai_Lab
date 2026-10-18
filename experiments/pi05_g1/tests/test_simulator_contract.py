"""CPU-only checks for simulator admission; no physics/model initialization."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.pi05_g1.contract import DATASET_NAMES
from experiments.pi05_g1.simulator import PolicyProcess, canonical_config, shared_action_bounds, validate_case
from experiments.pi05_g1.placement import APPLE_NAME, evaluate as evaluate_placement


class SimulatorAdmissionTests(unittest.TestCase):
    def test_camera_callable_fingerprint_uses_stable_import_identity(self):
        value = canonical_config({'spawn': {'func': validate_case}})
        self.assertEqual(value['spawn']['func'], {'callable': 'experiments.pi05_g1.simulator.validate_case'})

    def test_frozen_case_rejects_nonfinite_and_outside_offsets(self):
        for offset in ([float('nan'), 0], [float('inf'), 0], [.021, 0]):
            with self.assertRaises(ValueError):
                validate_case({'case_id': 'c', 'seed': 0, 'apple_xy_offset_m': offset})
        validate_case({'case_id': 'c', 'seed': 0, 'apple_xy_offset_m': [.02, 0]})

    def test_bounds_intersect_joint_limits_in_named_order(self):
        names = tuple(reversed(DATASET_NAMES))
        limits = np.repeat([[-2., 2.]], 43, axis=0)
        limits[names.index('left_shoulder_pitch_joint')] = [-.2, .4]
        lower, upper = shared_action_bounds({'q01': [-1.] * 32, 'q99': [1.] * 32}, limits, names)
        self.assertAlmostEqual(lower[0], -.2)
        self.assertAlmostEqual(upper[0], .4)
        self.assertEqual(lower[28], -1)

    def test_empty_limit_intersection_rejected(self):
        with self.assertRaises(ValueError):
            shared_action_bounds({'q01': [3.] * 32, 'q99': [4.] * 32}, np.repeat([[-1., 1.]], 43, axis=0), DATASET_NAMES)

    def test_policy_identity_mismatch_rejected(self):
        command = [sys.executable, '-u', '-c', "import sys,json; r=json.loads(sys.stdin.readline()); print(json.dumps({'episode_id':'wrong','control_tick':r['control_tick'],'actions':[[0]*32],'inference_seconds':0}))"]
        with tempfile.TemporaryDirectory() as temporary:
            model = PolicyProcess(command, Path(temporary) / 'stderr')
            try:
                with self.assertRaisesRegex(ValueError, 'identity'):
                    model.infer(np.zeros((2, 2, 3), np.uint8), np.zeros(32), {'case_id': 'c', 'seed': 0}, 0, 'pick')
            finally:
                model.close()


class PlacementMeasurementTests(unittest.TestCase):
    def evidence(self, count=121):
        rows, samples = [], []
        for tick in range(count):
            rows.append({'control_tick': tick, 'physics_steps_since_reset': tick * 4,
                         'upright': 1., 'root_link_pose_w': [0., 0., 0., 0., 0., 0., 1.],
                         'original_terms': {'object_dropped': False},
                         'acceptance_truth_only': {APPLE_NAME: {
                             'pose': [0., 0., 0., 0., 0., 0., 1.],
                             'velocity': [0., 0., 0., .5, -.5, .3]}}})
            samples.append({'tick': tick, 'minimum_footprint_margin_m': .02,
                            'plate_support': True, 'separation_proved': True,
                            'standing': True, 'ready': False})
        legacy = {'schema': 'g1_original_source_conservative_placement_audit_v1',
                  'completed_source_controls': count - 1, 'samples': samples,
                  'source_diagnostic_placement_passed': False, 'max_stable_seconds': 0.}
        return legacy, rows

    def assess(self, legacy, rows, **kwargs):
        return evaluate_placement(legacy, rows, first_grasp_tick=5,
                                  source_success_candidate_tick=10, **kwargs)

    def test_pose_stable_velocity_jitter_passes_and_preserves_legacy_failure(self):
        result = self.assess(*self.evidence())
        self.assertTrue(result['stable_placement_passed'])
        self.assertFalse(result['legacy_speed_audit_passed'])
        self.assertEqual(result['qualified_window']['start_tick'], 10)
        self.assertEqual(result['qualified_window']['end_tick'], 110)

    def test_true_drift_fails(self):
        legacy, rows = self.evidence()
        for tick, row in enumerate(rows):
            row['acceptance_truth_only'][APPLE_NAME]['pose'][0] = .02 * tick / 120
        self.assertFalse(self.assess(legacy, rows)['stable_placement_passed'])

    def test_out_and_back_motion_cannot_hide_behind_equal_endpoints(self):
        legacy, rows = self.evidence()
        rows[60]['acceptance_truth_only'][APPLE_NAME]['pose'][0] = .02
        self.assertFalse(self.assess(legacy, rows)['stable_placement_passed'])

    def test_true_rotation_fails(self):
        legacy, rows = self.evidence()
        for tick, row in enumerate(rows):
            angle = np.deg2rad(20 * tick / 120)
            row['acceptance_truth_only'][APPLE_NAME]['pose'][3:] = [0., 0., np.sin(angle / 2), np.cos(angle / 2)]
        self.assertFalse(self.assess(legacy, rows)['stable_placement_passed'])

    def test_contact_or_support_interruption_cannot_be_filtered_out(self):
        for field in ('plate_support', 'separation_proved', 'standing'):
            legacy, rows = self.evidence()
            legacy['samples'][60][field] = False
            self.assertFalse(self.assess(legacy, rows)['stable_placement_passed'])

    def test_insufficient_two_seconds_fails(self):
        self.assertFalse(self.assess(*self.evidence(110))['stable_placement_passed'])

    def test_quaternion_sign_flip_is_same_rotation(self):
        legacy, rows = self.evidence()
        for i, row in enumerate(rows):
            row['acceptance_truth_only'][APPLE_NAME]['pose'][6] = -2. if i % 2 else 2.
        self.assertTrue(self.assess(legacy, rows)['stable_placement_passed'])

    def test_zero_or_nonfinite_quaternion_rejected(self):
        for quaternion in ([0.] * 4, [0., 0., float('nan'), 1.]):
            legacy, rows = self.evidence()
            rows[60]['acceptance_truth_only'][APPLE_NAME]['pose'][3:] = quaternion
            with self.assertRaises(ValueError):
                self.assess(legacy, rows)

    def test_reset_or_extra_physics_integration_rejected(self):
        for key, value in [('control_tick', 0), ('physics_steps_since_reset', 241)]:
            legacy, rows = self.evidence()
            rows[60][key] = value
            with self.assertRaises(ValueError):
                self.assess(legacy, rows)

    def test_grasp_before_candidate_and_no_prior_fall_or_drop_required(self):
        legacy, rows = self.evidence()
        self.assertFalse(evaluate_placement(legacy, rows, first_grasp_tick=10,
            source_success_candidate_tick=5)['stable_placement_passed'])
        self.assertFalse(self.assess(legacy, rows, fell=True)['stable_placement_passed'])
        self.assertFalse(self.assess(legacy, rows, dropped=True)['stable_placement_passed'])


if __name__ == '__main__':
    unittest.main()
