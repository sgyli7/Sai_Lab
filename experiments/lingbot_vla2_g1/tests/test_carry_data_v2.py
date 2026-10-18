"""A valid carry prefix cannot silently acquire full-success labels or extra rows."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.lingbot_vla2_g1 import carry_data_v2 as c
from experiments.lingbot_vla2_g1.data import OWN


class NewCarryAdmissionTests(unittest.TestCase):
    def changed_manifest(self, **fields):
        path = OWN / 'prepared/fresh_native2026101702_carry_prefix/manifest.json'
        data = json.loads(path.read_text())
        data.update(fields)
        directory = tempfile.TemporaryDirectory(dir=OWN)
        self.addCleanup(directory.cleanup)
        copy = Path(directory.name) / 'manifest.json'
        copy.write_text(json.dumps(data))
        return copy

    def test_partial_carry_cannot_be_declared_full_success(self):
        with self.assertRaisesRegex(ValueError, 'task-failed'):
            c.admit(self.changed_manifest(task_success_teacher=True))

    def test_extra_unqualified_tail_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'boundary'):
            c.admit(self.changed_manifest(length=1463))


if __name__ == '__main__':
    unittest.main()
