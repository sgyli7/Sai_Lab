"""Evidence can be valid while a teacher's physical task still failed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import success_data as s
from experiments.lingbot_vla2_g1.data import OWN


class TeacherAdmissionTests(unittest.TestCase):
    def test_actual_failed_LingBot_case_cannot_be_success_teacher(self):
        case = OWN / 'formal_episodes_round02_pose_speed/long5m_center_s1040'
        audit = OWN / 'evaluation_segments_round02_pilot01/case_00_cpu_audit.json'
        with tempfile.TemporaryDirectory(dir=OWN) as directory:
            root = Path(directory)
            declaration = root / 'declaration.json'
            declaration.write_text(json.dumps({'schema': 'lingbot_fresh_native_TRAIN_declaration_v1',
                                               'train_case_ids': ['long5m_center_s1040']}))
            with self.assertRaisesRegex(ValueError, 'actual complete autonomous native success'):
                s.prepare(case, audit, declaration, root / 'forbidden_teacher')
            self.assertFalse((root / 'forbidden_teacher').exists())

    def test_case_not_prospectively_declared_TRAIN_is_rejected(self):
        case = OWN / 'formal_episodes_round02_pose_speed/long5m_center_s1040'
        audit = OWN / 'evaluation_segments_round02_pilot01/case_00_cpu_audit.json'
        with tempfile.TemporaryDirectory(dir=OWN) as directory:
            root = Path(directory)
            declaration = root / 'declaration.json'
            declaration.write_text(json.dumps({'schema': 'lingbot_fresh_native_TRAIN_declaration_v1', 'train_case_ids': []}))
            with self.assertRaisesRegex(ValueError, 'Prospective TRAIN'):
                s.prepare(case, audit, declaration, root / 'forbidden_teacher')

    def test_carry_only_flag_cannot_be_relabelled_as_full_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'manifest.json'
            path.write_text(json.dumps({'schema': 'lingbot_fresh_full_success_TRAIN_episode_v1',
                                        'status': 'completed', 'task_success_teacher': False}))
            with patch.object(s, 'OWN', root), self.assertRaisesRegex(ValueError, 'full-success'):
                s.FullSuccessEpisode(path, None)


if __name__ == '__main__':
    unittest.main()
