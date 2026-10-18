"""CPU regression checks against real failed TRAIN evidence, no new physics."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha
from .privileged_teacher import PlacementTeacher
from .teacher_measurement import attribute_teacher


class ActualTRAINCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = OWN / 'resumed_turn01/episodes/resume01_native_teacher01'
        cls.config = json.loads((cls.folder / 'config.json').read_text())
        cls.result = json.loads((cls.folder / 'result.json').read_text())
        cls.rows = [json.loads(line) for line in (cls.folder / 'trajectory.jsonl').read_text().splitlines()]
        assert sha(cls.folder / 'trajectory.jsonl') == cls.result['trace_sha256']
        cls.objects = json.loads(Path(cls.config['geometry_definition']).read_text())
        cls.robot = json.loads(Path(cls.config['robot_geometry']).read_text())

    def teacher(self):
        return PlacementTeacher(self.rows, self.objects, self.robot, self.result['body_names'])

    def test_actual_TRAIN_replay_and_boundary(self):
        teacher = self.teacher()
        for tick in range(0, 1450, 50):
            actions, _ = teacher.actions(self.rows[tick])
            np.testing.assert_array_equal(actions, np.asarray([r['applied_action32'] for r in self.rows[tick + 1:tick + 51]], np.float32))
        actions, _ = teacher.actions(self.rows[1450])
        np.testing.assert_array_equal(actions[:12], np.asarray([r['applied_action32'] for r in self.rows[1451:1463]], np.float32))
        self.assertTrue(np.all(actions[12:, 29:] == 0))

    def test_actual_navigation_error_and_negative_control(self):
        for tick in (1500, 1700, 1854):
            teacher = self.teacher()
            actions, diagnostic = teacher.actions(self.rows[tick])
            center, goal, _, _ = teacher.geometry(self.rows[tick])
            root = rotation_xyzw(self.rows[tick]['root_link_pose_w'])
            yaw = np.arctan2(root[1, 0], root[0, 0])
            world_velocity = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ actions[-1, 29:31]
            self.assertGreater(float((goal[:2] - center[:2]) @ world_velocity), 0)
            self.assertLess(float((goal[:2] - center[:2]) @ -world_velocity), 0)
            self.assertLessEqual(np.linalg.norm(world_velocity), .300001)
            self.assertEqual(actions.shape, (50, 32))
            self.assertTrue(np.isfinite(actions).all())
            self.assertEqual(diagnostic['phase'], 'align_box_over_bin')

    def test_attribution_preserves_every_physical_value(self):
        physical = self.result['acceptance']
        original = copy.deepcopy(physical)
        reported = attribute_teacher(physical, self.rows, {'privileged_TRAIN_teacher': {'TRAIN_only': True}})
        for key in physical:
            if key not in ('policy_external_interventions', 'intervention_rule'):
                self.assertEqual(physical[key], reported[key])
        self.assertEqual(physical, original)
        self.assertEqual(reported['policy_external_interventions'], 3000)
        self.assertFalse(reported['counts_as_LingBot_success'])
        self.assertFalse(reported['autonomous_model_policy'])
        with self.assertRaises(ValueError):
            attribute_teacher(physical, self.rows, {})


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ActualTRAINCollectorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [ROOT / 'experiments/lingbot_vla2_g1/diagnostics' / name for name in (
        'privileged_teacher.py', 'training_teacher_kinematics.py', 'teacher_measurement.py',
        'teacher_sim_entry.py', 'run_teacher_case.py', 'teacher_audit.py', 'teacher_CPU_gate.py')]
    report = {'schema': 'lingbot_privileged_teacher_CPU_gate_v1', 'passed': result.wasSuccessful(),
              'tests_run': result.testsRun, 'source_files_sha256': {str(p): sha(p) for p in sources},
              'actual_failed_TRAIN_trace_sha256': ActualTRAINCollectorTests.result['trace_sha256'],
              'GPU_started': False, 'new_physics_trial': False, 'actual_robot_success_proved': False}
    with (OWN / 'diagnostics/privileged_teacher_CPU_gate.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 2)


if __name__ == '__main__':
    main()
