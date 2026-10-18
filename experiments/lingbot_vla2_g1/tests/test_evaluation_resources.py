import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from experiments.lingbot_vla2_g1 import evaluate
from experiments.lingbot_vla2_g1.data import sha


class EvaluationResourceTests(unittest.TestCase):
    def test_failed_disk_admission_cancels_every_unused_fifo_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / 'plan.json'
            plan_path.write_text('{}')
            plan = {'jobs': [{'max_seconds': 900} for _ in range(3)]}
            tickets = [{'ticket': 't' + str(i), 'status': 'waiting'} for i in range(3)]
            with patch.object(evaluate, 'OWN', root), patch.object(evaluate, 'load_plan', return_value=plan), patch.object(evaluate, 'admit_allocation', return_value={}), patch.object(evaluate, 'verify_payloads'), patch.object(evaluate.gpu_queue, 'request', side_effect=tickets), patch.object(evaluate.gpu_queue, 'status', return_value={'tickets': tickets}), patch.object(evaluate.gpu_queue, 'cancel') as cancel, patch.object(evaluate.shutil, 'disk_usage', return_value=Mock(free=0)), patch.object(evaluate.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'Keep32GiB'):
                    evaluate.run(plan_path, root / 'receipt.json', 3, [])
                self.assertEqual([call.args[1] for call in cancel.call_args_list], ['t0', 't1', 't2'])
                popen.assert_not_called()
            result = json.loads((root / 'receipt.json').read_text())
            self.assertEqual(result['status'], 'infrastructure_failed')
            self.assertEqual(result['jobs'], [])

    def test_resume_rejects_extra_raw_files_or_missing_weight_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'case'
            folder.mkdir()
            result = folder / 'result.json'
            result.write_text('{}')
            audit = root / 'audit.json'
            audit.write_text('{}')
            plan = {'jobs': [{'result_path': str(result), 'policy': 'p', 'case_id': 'c'}]}
            record = {'status': 'completed', 'index': 0, 'policy': 'p', 'case_id': 'c', 'case_files_sha256': evaluate.raw_files(folder), 'cpu_audit_path': str(audit), 'cpu_audit_sha256': sha(audit)}
            doc = {'schema': 'lingbot_vla2_g1_evaluation_segment_v1', 'plan_sha256': 'plan', 'model_payloads_verified_before': True, 'model_payloads_verified_end': True, 'jobs': [record]}
            receipt = root / 'receipt.json'
            receipt.write_text(json.dumps(doc))
            self.assertEqual(len(evaluate.previous_records([receipt], 'plan', plan)), 1)
            (folder / 'extra.txt').write_text('changed after completion')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                evaluate.previous_records([receipt], 'plan', plan)
            doc['model_payloads_verified_end'] = False
            receipt.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'phase boundary'):
                evaluate.previous_records([receipt], 'plan', plan)


if __name__ == '__main__':
    unittest.main()
