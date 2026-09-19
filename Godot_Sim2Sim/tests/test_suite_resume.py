import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim2sim.standalone.suite import run


class FrozenSuiteTests(unittest.TestCase):
    def test_same_model_hash_cannot_resume_after_control_or_native_binary_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);project=root/'project';assets=project/'runtime_assets';assets.mkdir(parents=True)
            (project/'native').mkdir()
            library=project/'native/policy.so';library.write_bytes(b'original native binary')
            (assets/'ui-font.bin').write_bytes(b'original font')
            deployment=dict(policies={'roller':{'sha256':'unchanged-model'}},control_config={})
            (assets/'deployment.json').write_text(json.dumps(deployment))
            case=root/'case.json';case.write_text('{}')
            result=dict(case='probe',completed=True,seed=915000,models={'roller':'unchanged-model'},task_metrics={})
            with patch('sim2sim.standalone.suite.run_case',return_value=result) as replay:
                output=root/'suite';run([case],output,workers=1,project=project)
                run([case],output,workers=1,project=project,resume=True)
                self.assertEqual(replay.call_count,2)
                frozen=output/'runtime/runtime_assets/deployment.json';original=frozen.read_bytes()
                deployment['control_config']={'roller':{'brake_heading_gain':1.}}
                frozen.write_text(json.dumps(deployment))
                with self.assertRaisesRegex(RuntimeError,'Frozen runtime changed'):
                    run([case],output,workers=1,project=project,resume=True)
                frozen.write_bytes(original)
                font=output/'runtime/runtime_assets/ui-font.bin'
                font.write_bytes(b'different font')
                with self.assertRaisesRegex(RuntimeError,'Frozen runtime changed'):
                    run([case],output,workers=1,project=project,resume=True)
                font.write_bytes(b'original font')
                (output/'runtime/native/policy.so').write_bytes(b'different native binary')
                with self.assertRaisesRegex(RuntimeError,'Frozen runtime changed'):
                    run([case],output,workers=1,project=project,resume=True)
                self.assertEqual(replay.call_count,2)


if __name__=='__main__':unittest.main()
