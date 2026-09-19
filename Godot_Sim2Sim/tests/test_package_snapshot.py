"""A frozen export must fingerprint its own files despite later workspace edits."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim2sim.standalone.package import source_hashes


class TestPackageSnapshot(unittest.TestCase):
    def test_snapshot_inventory_survives_concurrent_workspace_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repo';project=Path(directory)/'frozen'
            for path,text in [(root/'godot/live_only.gd','new task'),
                              (root/'src/reference.py','reference'),
                              (project/'snapshot_only.gd','frozen source'),
                              (project/'.godot/cache','generated')]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            with patch('sim2sim.standalone.package.subprocess.check_output',
                       return_value=b'godot/live_only.gd\0src/reference.py\0'):
                result=source_hashes(root,project)
            self.assertEqual(result,{
                'godot/snapshot_only.gd':hashlib.sha256(b'frozen source').hexdigest(),
                'src/reference.py':hashlib.sha256(b'reference').hexdigest(),
            })


if __name__=='__main__':
    unittest.main()
