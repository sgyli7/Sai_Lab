"""Replay the actual R6 failure at the alignment/lowering boundary on CPU."""
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.lingbot_vla2_g1.data import OWN
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v5 import PlacementTeacher, PREFIX_CONTROLS
from experiments.lingbot_vla2_g1.diagnostics.teacher_measurement_v5 import attribute_teacher


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
        config = {'privileged_TRAIN_teacher': {'TRAIN_only': True, 'collector_version': 5,
                                               'initial_replay_controls': 1800}}
        for ticks in [0, 1799, 1800, 1801, 2750, 3000]:
            actual = attribute_teacher({'success': False}, [None]*(ticks+1), config)
            self.assertEqual(actual['recorded_TRAIN_action_prefix_controls'], min(1800, ticks))
            self.assertEqual(actual['privileged_truth_feedback_queries'], len(range(1800, ticks, 50)))
            self.assertFalse(actual['counts_as_LingBot_success'])
        config['privileged_TRAIN_teacher']['initial_replay_controls'] = 1700
        with self.assertRaises(ValueError):
            attribute_teacher({}, [None]*2751, config)
