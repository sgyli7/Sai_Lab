"""Replay the actual R6 failure at the alignment/lowering boundary on CPU."""
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.lingbot_vla2_g1.data import OWN
import os
if os.environ.get('LINGBOT_RELEASE_CPU_NEGATIVE_CONTROL') == 'previous_v5':
    from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v5 import PlacementTeacher, PREFIX_CONTROLS
else:
    from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v7 import PlacementTeacher, PREFIX_CONTROLS
from experiments.lingbot_vla2_g1.diagnostics.teacher_measurement_v7 import attribute_teacher


class R6RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = OWN / 'round06_development_plan/episodes/round06_fresh_full60'
        cls.rows = [json.loads(x) for x in (folder / 'trajectory.jsonl').read_text().splitlines()]
        cls.result = json.loads((folder / 'result.json').read_text())
        cls.config = json.loads((folder / 'config.json').read_text())
        cls.objects = json.loads(Path(cls.config['geometry_definition']).read_text())
        cls.robot = json.loads(Path(cls.config['robot_geometry']).read_text())

    def teacher(self):
        return PlacementTeacher(self.rows, self.objects, self.robot, self.result['body_names'],
                                action_bounds=self.result['actual_action_bounds'])

    def test_full_applied_R6_prefix_is_exact_and_handoff_keeps_gripping_targets(self):
        teacher = self.teacher()
        for tick in range(0, PREFIX_CONTROLS, 50):
            actual, info = teacher.actions(self.rows[tick])
            np.testing.assert_array_equal(actual, np.asarray(
                [r['applied_action32'] for r in self.rows[tick+1:tick+51]], np.float32))
            self.assertFalse(info['privileged_truth_used'])
        np.testing.assert_array_equal(teacher.last, teacher.replay[-1])
        self.assertEqual(teacher.height, float(teacher.replay[-1, 28]))

    def test_actual_R6_outside_footprint_does_not_trigger_teacher_lowering(self):
        teacher = self.teacher()
        row = self.rows[2050]
        teacher.phase = 'bilateral_arm_alignment'
        teacher.recovery_ready_since = 2000
        _, _, _, margin = teacher.geometry(row)
        self.assertLess(margin, 0)
        self.assertLess(row['applied_action32'][28], teacher.height - .02)
        chunk, info = teacher.actions(row)
        self.assertEqual(info['phase'], 'bilateral_arm_alignment')
        self.assertIsNone(teacher.recovery_ready_since)
        np.testing.assert_allclose(chunk[:, 28], teacher.replay[-1, 28], atol=1e-7)

    def test_both_palms_correct_actual_R6_horizontal_error(self):
        teacher = self.teacher()
        row = self.rows[1950]
        teacher.phase = 'bilateral_arm_alignment'
        teacher.upper = np.asarray(row['applied_action32'][:28], float)
        teacher.last = np.asarray(row['applied_action32'], np.float32)
        start = teacher.upper.copy()
        center, goal, _, _ = teacher.geometry(row)
        chunk, _ = teacher.actions(row)
        for side, offset in [('left', 0), ('right', 7)]:
            jac, _ = teacher.kinematics.jacobian(row, side)
            delta = chunk[-1, offset:offset+7] - start[offset:offset+7]
            self.assertGreater(float((jac[:3] @ delta) @ np.r_[goal[:2]-center[:2], 0]), 0)
        self.assertTrue(np.all(chunk[:, 29:] == 0))

    def test_actual_prior_success_inside_margin_can_lower(self):
        prior = OWN / 'placement_recovery_teacher04/episode/trajectory.jsonl'
        rows = [json.loads(x) for x in prior.read_text().splitlines()]
        teacher = self.teacher()
        row = rows[2150]
        teacher.phase = 'bilateral_arm_alignment'
        teacher.recovery_ready_since = 2100
        _, _, _, margin = teacher.geometry(row)
        self.assertGreater(margin, .015)
        chunk, info = teacher.actions(row)
        self.assertEqual(info['phase'], 'lower_held_box')
        self.assertLess(chunk[-1, 28], teacher.replay[-1, 28] - .03)

    def test_attribution_uses_actual_declared_prefix_and_query_boundary(self):
        config = {'privileged_TRAIN_teacher': {'TRAIN_only': True, 'collector_version': 7,
                                               'initial_replay_controls': 1800}}
        for ticks in [0, 1799, 1800, 1801, 2750, 3000]:
            actual = attribute_teacher({'success': False}, [None]*(ticks+1), config)
            self.assertEqual(actual['recorded_TRAIN_action_prefix_controls'], min(1800, ticks))
            self.assertEqual(actual['privileged_truth_feedback_queries'], len(range(1800, ticks, 50)))
            self.assertFalse(actual['counts_as_LingBot_success'])
        config['privileged_TRAIN_teacher']['initial_replay_controls'] = 1700
        with self.assertRaises(ValueError):
            attribute_teacher({}, [None]*2751, config)

    def floor_state_teacher(self):
        folder = OWN / 'placement_recovery_teacher05/episode'
        rows = [json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
        events = [json.loads(x) for x in (folder/'decoded.jsonl').read_text().splitlines()][1:]
        teacher = self.teacher()
        for event in events:
            if event['control_tick'] >= 2300:
                break
            chunk, _ = teacher.actions(rows[event['control_tick']])
            np.testing.assert_array_equal(chunk, np.asarray(event['actions32'], np.float32))
        return teacher, rows[2300]

    def test_actual_floor_contact_freezes_height_and_defers_unsafe_release(self):
        teacher, row = self.floor_state_teacher()
        previous_height = teacher.height
        center, goal, clearance, margin = teacher.geometry(row)
        self.assertLessEqual(clearance, .025)
        self.assertLess(margin, .025)
        chunk, info = teacher.actions(row)
        self.assertAlmostEqual(float(chunk[-1,28]), previous_height, places=7)
        self.assertEqual(info['phase'], 'lower_held_box')
        self.assertTrue(info['floor_centering_pending'])
        self.assertIsNone(teacher.floor_release_ready_since)
        self.assertTrue(all(np.asarray(prediction[:2]) @ (goal[:2]-center[:2]) > 0
                            for prediction in info['bilateral_Jacobian_predicted_translation_m']))

    def test_supported_centered_fixture_opens_only_after_one_second_and_holds_height(self):
        teacher, row = self.floor_state_teacher()
        center, goal, _, _ = teacher.geometry(row)
        teacher.geometry = lambda unused: (goal.copy(), goal.copy(), .0, .055)
        teacher.floor_release_ready_since = 2250
        previous_height = teacher.height
        chunk, info = teacher.actions(row)
        self.assertEqual(info['phase'], 'open_palms')
        self.assertAlmostEqual(float(chunk[-1,28]), previous_height, places=7)
        self.assertEqual(teacher.release_tick, 2300)

    def test_actual06_release_state_is_held_for_more_centering_with_greater_XY_correction(self):
        import copy
        from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v6 import PlacementTeacher as Old
        folder = OWN/'placement_recovery_teacher06/episode'
        rows = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
        old = Old(self.rows,self.objects,self.robot,self.result['body_names'],
                  action_bounds=self.result['actual_action_bounds'])
        for tick in range(0,2550,50):old.actions(rows[tick])
        candidate = self.teacher()
        candidate.__dict__.update(copy.deepcopy(old.__dict__))
        previous, previous_info = old.actions(rows[2550])
        actual, info = candidate.actions(rows[2550])
        self.assertEqual(previous_info['phase'],'open_palms')
        self.assertEqual(info['phase'],'lower_held_box')
        self.assertIsNone(candidate.floor_release_ready_since)
        self.assertGreater(info['horizontal_center_error_m'],.010)
        self.assertLess(info['full_footprint_XY_margin_m'],.050)
        for before,after in zip(previous_info['bilateral_Jacobian_predicted_translation_m'],
                                info['bilateral_Jacobian_predicted_translation_m']):
            self.assertGreater(np.linalg.norm(after[:2]),1.2*np.linalg.norm(before[:2]))
        bounds=self.result['actual_action_bounds']
        self.assertTrue(np.all(actual >= np.asarray(bounds['lower'])-1e-6))
        self.assertTrue(np.all(actual <= np.asarray(bounds['upper'])+1e-6))
