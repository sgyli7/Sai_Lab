"""Exercise the actual stalled TRAIN row, including a previous-version control."""
import argparse
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.lingbot_vla2_g1.data import OWN, sha
from .privileged_teacher import PlacementTeacher as VersionOne
from .privileged_teacher_v2 import PlacementTeacher as VersionTwo


class ActualStalledCollectorTests(unittest.TestCase):
    teacher_type = VersionTwo

    @classmethod
    def setUpClass(cls):
        cls.folder = OWN / 'placement_teacher01/episode'
        cls.config = json.loads((cls.folder / 'config.json').read_text())
        cls.result = json.loads((cls.folder / 'result.json').read_text())
        assert sha(cls.folder / 'trajectory.jsonl') == cls.result['trace_sha256']
        cls.rows = [json.loads(line) for line in (cls.folder / 'trajectory.jsonl').read_text().splitlines()]
        cls.source = [json.loads(line) for line in Path(cls.config['privileged_TRAIN_teacher']['source_trace']).read_text().splitlines()]
        cls.objects = json.loads(Path(cls.config['geometry_definition']).read_text())
        cls.robot = json.loads(Path(cls.config['robot_geometry']).read_text())

    def teacher(self):
        kwargs = {'action_bounds': self.result['actual_action_bounds']} if self.teacher_type is VersionTwo else {}
        return self.teacher_type(self.source, self.objects, self.robot, self.result['body_names'], **kwargs)

    def test_preserves_actual_TRAIN_replay_and_chunk_boundary(self):
        teacher = self.teacher()
        for tick in range(0, 1500, 50):
            chunk, _ = teacher.actions(self.rows[tick])
            valid = min(50, 1462 - tick)
            if valid > 0:
                np.testing.assert_array_equal(chunk[:valid], np.asarray(
                    [r['applied_action32'] for r in self.source[tick + 1:tick + valid + 1]], np.float32))
            if valid < 50:
                self.assertTrue(np.all(chunk[valid:, 29:] == 0))

    def test_both_palms_correct_actual_eleven_cm_plateau(self):
        for tick in (2200, 2600, 2950):
            teacher = self.teacher()
            row = self.rows[tick]
            teacher.upper = np.asarray(row['applied_action32'][:28], float)
            teacher.last = np.asarray(row['applied_action32'], np.float32)
            start = teacher.upper.copy()
            chunk, diagnostic = teacher.actions(row)
            center, goal, _, _ = teacher.geometry(row)
            error = np.r_[goal[:2] - center[:2], 0.]
            self.assertGreater(np.linalg.norm(error), .10)
            self.assertTrue(np.all(chunk[:, 29:] == 0), 'Near-bin correction must stop base motion')
            for side, offset in (('left', 0), ('right', 7)):
                jacobian, _ = teacher.kinematics.jacobian(row, side)
                change = chunk[-1, offset:offset + 7] - start[offset:offset + 7]
                shift = jacobian[:3] @ change
                self.assertGreater(float(shift @ error), .001, 'Both palms must move toward measured bin center')
                self.assertLessEqual(np.max(np.abs(change)), .300001)
            np.testing.assert_array_equal(chunk[:, 14:28], np.repeat(start[None, 14:28].astype(np.float32), 50, axis=0))
            self.assertTrue(np.isfinite(chunk).all())
            self.assertEqual(diagnostic['phase'], 'bilateral_arm_alignment')

    def test_lowering_preserves_limits_and_commands_downward(self):
        teacher = self.teacher()
        row = self.rows[2600]
        teacher.upper = np.asarray(row['applied_action32'][:28], float)
        teacher.last = np.asarray(row['applied_action32'], np.float32)
        teacher.phase = 'lower_held_box'
        teacher.entered = 2600
        chunk, diagnostic = teacher.actions(row)
        self.assertLess(chunk[-1, 28], .75)
        self.assertGreaterEqual(chunk[-1, 28], .45)
        self.assertTrue(np.all(chunk[:, :28] >= np.asarray(self.result['actual_action_bounds']['lower'][:28])))
        self.assertTrue(np.all(chunk[:, :28] <= np.asarray(self.result['actual_action_bounds']['upper'][:28])))
        if self.teacher_type is VersionTwo:
            self.assertTrue(all(p[2] < 0 for p in diagnostic['bilateral_Jacobian_predicted_translation_m']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('v1', 'v2'), default='v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ActualStalledCollectorTests.teacher_type = VersionOne if args.version == 'v1' else VersionTwo
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ActualStalledCollectorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [Path(__file__), Path(__file__).with_name('privileged_teacher_v2.py'),
               Path(__file__).with_name('training_teacher_kinematics.py')]
    report = {'schema': 'lingbot_actual_stalled_TRAIN_collector_v2_CPU_gate',
              'collector': args.version, 'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
              'source_files_sha256': {str(p): sha(p) for p in sources},
              'actual_failed_TRAIN_trace_sha256': ActualStalledCollectorTests.result['trace_sha256'],
              'GPU_started': False, 'new_physical_trial': False,
              'actual_robot_success_proved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 2)


if __name__ == '__main__':
    main()
