"""Exercise actual physically in-bin alignment; preserve final acceptance."""
import json
import numpy as np

from .test_recovery_teacher_v3 import ActualRecoveryTests
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v3 import PlacementTeacher as Baseline
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v4 import PlacementTeacher
from experiments.lingbot_vla2_g1.data import OWN


class ActualRecoveryV4Tests(ActualRecoveryTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source_rows = cls.rows
        path = OWN / 'placement_recovery_teacher03/episode/trajectory.jsonl'
        cls.rows = [json.loads(line) for line in path.read_text().splitlines()]

    def teacher(self, implementation=PlacementTeacher):
        return implementation(self.source_rows, self.objects, self.robot, self.result['body_names'],
                              action_bounds=self.result['actual_action_bounds'])

    def test_actual_safe_footprint_can_lower_while_continuing_XY_correction(self):
        row = self.rows[2150]
        for implementation in (Baseline, PlacementTeacher):
            teacher = self.teacher(implementation)
            teacher.phase = 'bilateral_arm_alignment'
            teacher.upper = np.asarray(row['applied_action32'][:28], float)
            teacher.last = np.asarray(row['applied_action32'], np.float32)
            teacher.recovery_ready_since = 2100
            center, goal, _, margin = teacher.geometry(row)
            self.assertGreater(margin, .015)
            self.assertLess(np.linalg.norm(goal[:2]-center[:2]), .045)
            chunk, info = teacher.actions(row)
            if implementation is Baseline:
                self.assertEqual(info['phase'], 'bilateral_arm_alignment')
                self.assertEqual(float(chunk[-1,28]), np.float32(teacher.replay[-1,28]))
            else:
                self.assertEqual(info['phase'], 'lower_held_box')
                self.assertLess(chunk[-1,28], teacher.replay[-1,28]-.03)
                self.assertTrue(np.all(chunk[:,29:]==0))
                self.assertTrue(all(shift[2]<0 for shift in info['bilateral_Jacobian_predicted_translation_m']))

    def test_insufficient_footprint_margin_cannot_use_earlier_lowering(self):
        teacher = self.teacher()
        row = self.rows[2150]
        teacher.phase = 'bilateral_arm_alignment'
        teacher.recovery_ready_since = 2100
        center, goal, clearance, _ = teacher.geometry(row)
        teacher.geometry = lambda unused: (center, goal, clearance, -.001)
        chunk, info = teacher.actions(row)
        self.assertEqual(info['phase'], 'bilateral_arm_alignment')
        self.assertEqual(float(chunk[-1,28]), np.float32(teacher.replay[-1,28]))
