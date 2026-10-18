"""Protect historical model bytes and recover an interrupted alias transition."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import permanent_checkpoint_alias as module


class PermanentAliasTests(unittest.TestCase):
    def fixture(self, root):
        own, backup = root / 'own', root / 'backup'
        own.mkdir(); backup.mkdir()
        local, permanent = own / 'final', backup / 'final'
        for folder in (local, permanent):
            folder.mkdir()
            (folder / 'trainable.safetensors').write_bytes(b'synthetic_checkpoint_bytes')
            (folder / 'metadata.json').write_text(json.dumps({'weights_sha256': module.sha(folder / 'trainable.safetensors')}))
        return own, backup, local, permanent

    def test_same_bytes_stay_readable_and_operation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            own, backup, local, permanent = self.fixture(Path(directory))
            original = module.inventory(local)
            with patch.object(module, 'OWN', own), patch.object(module, 'BACKUP', backup):
                self.assertEqual(module.use_permanent_copy(local, permanent), original)
                self.assertTrue(local.is_symlink())
                self.assertEqual(module.use_permanent_copy(local, permanent), original)
                self.assertEqual(module.inventory(permanent), original)
            self.assertEqual(len(list(own.iterdir())), 1)
            self.assertEqual(len([p for p in own.rglob('*') if p.is_file()]), 0)
            self.assertEqual((local / 'trainable.safetensors').read_bytes(), b'synthetic_checkpoint_bytes')

    def test_changed_permanent_copy_cannot_replace_working_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            own, backup, local, permanent = self.fixture(Path(directory))
            original = module.inventory(local)
            (permanent / 'trainable.safetensors').write_bytes(b'changed')
            with patch.object(module, 'OWN', own), patch.object(module, 'BACKUP', backup):
                with self.assertRaises(ValueError):
                    module.use_permanent_copy(local, permanent)
            self.assertFalse(local.is_symlink())
            self.assertEqual(module.inventory(local), original)

    def test_failure_after_rename_restores_original_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            own, backup, local, permanent = self.fixture(Path(directory))
            original = module.inventory(local)
            with patch.object(module, 'OWN', own), patch.object(module, 'BACKUP', backup), \
                    patch.object(module, 'create_alias', side_effect=OSError('synthetic interrupted transition')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    module.use_permanent_copy(local, permanent)
            self.assertFalse(local.is_symlink())
            self.assertEqual(module.inventory(local), original)
            self.assertEqual(module.inventory(permanent), original)
            self.assertEqual(list(own.iterdir()), [local])


if __name__ == '__main__':
    unittest.main()
