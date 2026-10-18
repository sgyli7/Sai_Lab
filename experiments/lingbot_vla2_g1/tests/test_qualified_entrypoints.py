import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import sim_launcher_pose_speed as launcher
from experiments.lingbot_vla2_g1 import run_case_pose_speed as runner
from experiments.lingbot_vla2_g1.acceptance_context import admit_qualification


class QualifiedEntrypointTests(unittest.TestCase):
    def test_docker_keeps_owned_writes_and_uses_qualified_inner_entry(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, 'ROOT', Path(directory)):
            command = launcher.docker_command(Path(directory) / 'case.json', gpu=False)
            self.assertIn('experiments.pi05_g1_mobile.sim_launcher_pose_speed', command)
            evidence = Path(directory) / '.scratch/lingbot_vla2_g1'
            self.assertIn(str(evidence) + ':' + str(evidence) + ':rw', command)
            self.assertNotIn(str(Path(directory) / '.scratch/pi05_g1/round03_mobile') + ':' + str(Path(directory) / '.scratch/pi05_g1/round03_mobile') + ':rw', command)

    def test_wrong_measurement_never_starts_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config.json'
            path.write_text(json.dumps({'output_dir': str(root / 'raw'), 'case': {'case_id': 'x'}, 'policy_id': 'lingbot_vla2_selected', 'acceptance_measurement': {'wrong': True}, 'frozen_files': {}}))
            with patch('sys.argv', ['runner', '--config', str(path)]), patch.object(runner, 'run_owned_docker') as docker:
                with self.assertRaisesRegex(ValueError, 'Qualified shared placement measurement'):
                    runner.main()
                docker.assert_not_called()
            self.assertEqual(json.loads((root / 'raw/result.json').read_text())['status'], 'failed')

    def test_failed_or_mislabeled_primitive_cannot_qualify_formal_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qualification.json'
            path.write_text(json.dumps({'schema': 'pi05_g1_mobile_placement_primitive_qualification_v1', 'status': 'completed', 'passed': False}))
            with self.assertRaisesRegex(ValueError, 'Completed prospective'):
                admit_qualification(path)
            path.write_text(json.dumps({'schema': 'pi05_g1_mobile_placement_primitive_qualification_v1', 'status': 'completed', 'passed': True, 'full_task_success': True}))
            with self.assertRaisesRegex(ValueError, 'mislabeled'):
                admit_qualification(path)


if __name__ == '__main__':
    unittest.main()
