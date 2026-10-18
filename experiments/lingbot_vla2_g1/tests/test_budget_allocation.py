import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.lingbot_vla2_g1 import budget_allocated as allocated
from experiments.lingbot_vla2_g1.data import sha


class AllocationTests(unittest.TestCase):
    def fixture(self, root):
        rows = [{'stage': 'training', 'elapsed_seconds': 4300., 'status': 'completed', 'last_unit_properties': {'ActiveState': 'inactive'}, 'docker_cleanup': {'complete': True}}]
        receipt = {'schema': 'lingbot_vla2_budget_reallocation_v1', 'total_limit_seconds': 28800, 'new_stage_limits_seconds': allocated.LIMITS, 'preserved_run_count': 1, 'preserved_runs_sha256': allocated.runs_digest(rows)}
        allocation = root / 'gpu_budget_allocation_v2.json'
        allocation.write_text(json.dumps(receipt))
        ledger = root / 'gpu_budget.json'
        doc = {'total_limit_seconds': 28800, 'stage_limits_seconds': allocated.LIMITS, 'active_allocation_sha256': sha(allocation), 'runs': rows}
        ledger.write_text(json.dumps(doc))
        return ledger, doc

    def test_new_work_keeps_all_old_charges_and_total_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger, doc = self.fixture(root)
            with patch.object(allocated, 'OWN', root):
                self.assertEqual(allocated.admit_allocation(ledger)['used_total_seconds'], 4300.)
                doc['runs'][0]['elapsed_seconds'] = 0
                ledger.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'Prior actual GPU charges'):
                    allocated.admit_allocation(ledger)

    def test_larger_total_or_unclosed_run_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger, doc = self.fixture(root)
            with patch.object(allocated, 'OWN', root):
                doc['total_limit_seconds'] = 40000
                ledger.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    allocated.admit_allocation(ledger)
                doc['total_limit_seconds'] = 28800
                doc['runs'].append({'stage': 'evaluation', 'status': 'running', 'elapsed_seconds': 1})
                ledger.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'unfinished'):
                    allocated.admit_allocation(ledger)


if __name__ == '__main__':
    unittest.main()
