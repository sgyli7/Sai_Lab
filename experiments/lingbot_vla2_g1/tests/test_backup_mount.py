"""Container visibility for permanent model metadata, with original physics."""
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import sim_launcher_with_backup as launcher


class BackupMountTests(unittest.TestCase):
    def test_source_command_unchanged_except_one_read_only_task_backup_mount(self):
        source = ['docker', 'run', '--read-only', '--device', 'nvidia.com/gpu=all',
                  '--volume', '/source:/source:ro', '--volume', '/evidence:/evidence:rw',
                  '--workdir', '/source', '--entrypoint', '/isaac-sim/python.sh', 'frozen-image']
        with patch.object(launcher, '_source_docker_command', return_value=source):
            actual = launcher.docker_command(config='synthetic', gpu=True)
        expected_mount = f'{launcher.BACKUP}:{launcher.BACKUP}:ro'
        index = actual.index(expected_mount)
        self.assertEqual(actual[index-1], '--volume')
        self.assertEqual(actual[:index-1]+actual[index+1:], source)
        self.assertEqual(actual.count(expected_mount), 1)
        self.assertNotIn(f'{launcher.BACKUP}:{launcher.BACKUP}:rw', actual)

    def test_owned_launcher_restored_after_success_and_exception(self):
        original = launcher.base.docker_command
        for fail in (False, True):
            def run(config):
                self.assertIs(launcher.base.docker_command, launcher.docker_command)
                if fail:
                    raise RuntimeError('synthetic failure')
                return 'synthetic successful return'
            with patch.object(launcher.base, 'run_owned_docker', side_effect=run):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                        launcher.run_owned_docker('synthetic')
                else:
                    self.assertEqual(launcher.run_owned_docker('synthetic'), 'synthetic successful return')
            self.assertIs(launcher.base.docker_command, original)
