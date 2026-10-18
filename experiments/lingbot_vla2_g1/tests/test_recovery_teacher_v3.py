"""Actual long-carry trace checks at the declared recovery handoff."""
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.lingbot_vla2_g1.data import OWN
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v3 import PlacementTeacher, PREFIX_CONTROLS
from experiments.pi05_g1_mobile.acceptance import rotation_xyzw


class ActualRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = OWN / 'coherent_noise01_development_plan/episodes/round05_coherent_full60'
        cls.rows = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text().splitlines()]
        cls.result = json.loads((folder / 'result.json').read_text())
        cls.config = json.loads((folder / 'config.json').read_text())
        cls.objects = json.loads(Path(cls.config['geometry_definition']).read_text())
        cls.robot = json.loads(Path(cls.config['robot_geometry']).read_text())

    def teacher(self):
        return PlacementTeacher(self.rows, self.objects, self.robot, self.result['body_names'],
                                action_bounds=self.result['actual_action_bounds'])

    def test_exact_applied_prefix_and_measured_handoff(self):
        teacher = self.teacher()
        self.assertEqual(len(teacher.replay), PREFIX_CONTROLS)
        for tick in range(0, PREFIX_CONTROLS, 50):
            chunk, info = teacher.actions(self.rows[tick])
            np.testing.assert_array_equal(chunk, np.asarray(
                [row['applied_action32'] for row in self.rows[tick+1:tick+51]], np.float32))
            self.assertFalse(info['privileged_truth_used'])
        np.testing.assert_array_equal(teacher.last, teacher.replay[-1])
        self.assertEqual(teacher.height, float(teacher.replay[-1, 28]))

    def test_handoff_navigation_reduces_actual_bin_error(self):
        teacher = self.teacher()
        row = self.rows[PREFIX_CONTROLS]
        chunk, info = teacher.actions(row)
        center, goal, _, _ = teacher.geometry(row)
        self.assertGreater(np.linalg.norm(goal[:2]-center[:2]), .16)
        rot = rotation_xyzw(row['root_link_pose_w'])
        world_velocity = rot[:2,:2] @ chunk[-1,29:31]
        self.assertGreater(float(world_velocity @ (goal[:2]-center[:2])), 0)
        self.assertLessEqual(np.linalg.norm(chunk[-1,29:31]), .300001)
        np.testing.assert_array_equal(chunk[:, :28], np.repeat(teacher.replay[-1:,:28],50,0))
        self.assertEqual(info['phase'], 'align_box_over_bin')

    def test_near_bin_uses_bilateral_correction_without_base_drive(self):
        teacher = self.teacher()
        row = self.rows[2400]
        teacher.upper = np.asarray(row['applied_action32'][:28], float)
        teacher.last = np.asarray(row['applied_action32'], np.float32)
        teacher.phase = 'bilateral_arm_alignment'
        start = teacher.upper.copy()
        chunk, _ = teacher.actions(row)
        center, goal, _, _ = teacher.geometry(row)
        error = np.r_[goal[:2]-center[:2],0.]
        for side, offset in [('left',0),('right',7)]:
            jac, _ = teacher.kinematics.jacobian(row,side)
            change = chunk[-1,offset:offset+7]-start[offset:offset+7]
            self.assertGreater(float((jac[:3] @ change) @ error),0)
        self.assertTrue(np.all(chunk[:,29:] == 0))


if __name__ == '__main__':
    unittest.main()
