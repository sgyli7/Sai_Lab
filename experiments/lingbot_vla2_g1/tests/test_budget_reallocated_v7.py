"""Synthetic accounting regressions; no GPU, queue mutation, or goal claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import budget_reallocated_v7 as module
from experiments.lingbot_vla2_g1.budget_allocated import runs_digest
from experiments.lingbot_vla2_g1.data import sha


class ReallocationTests(unittest.TestCase):
    limits = {'preflight': 1200, 'training': 16800, 'evaluation': 39600}

    def closed(self, seconds=1234):
        return {'status': 'completed', 'stage': 'evaluation', 'elapsed_seconds': seconds,
                'last_unit_properties': {'ActiveState': 'inactive'},
                'docker_cleanup': {'complete': True}, 'synthetic': True}

    def fixture(self, root):
        rows = [self.closed()]
        previous = root / module.SNAPSHOT
        previous.write_text(json.dumps({'runs': rows, 'total_limit_seconds': 43200, 'synthetic': True}))
        old = root / 'gpu_budget_allocation_user_resume_v6.json'
        old.write_text(json.dumps({'synthetic': True, 'prior_limit': 43200}))
        allocation = {'schema': 'lingbot_vla2_stage_reallocation_v7', 'total_limit_seconds': 57600,
                      'previous_total_limit_seconds': 43200,
                      'finite_completion_continuation_under_latest_user_instruction': True,
                      'historical_charges_not_reset': True,
                      'owner_thread': module.OWNER, 'user_authorization': '开始执行计划把， 直到目标达成',
                      'old_wall_deadline_enforced': False, 'new_stage_limits_seconds': self.limits,
                      'supervisor_sha256': sha(module.__file__), 'previous_ledger_snapshot': str(previous),
                      'previous_ledger_sha256': sha(previous), 'previous_allocation_path': str(old),
                      'previous_allocation_sha256': sha(old), 'preserved_run_count': 1,
                      'preserved_runs_sha256': runs_digest(rows), 'synthetic': True}
        path = root / module.ALLOCATION
        path.write_text(json.dumps(allocation))
        ledger = root / 'gpu_budget.json'
        document = {'runs': rows, 'total_limit_seconds': 57600, 'stage_limits_seconds': self.limits,
                    'active_allocation_sha256': sha(path), 'synthetic': True}
        ledger.write_text(json.dumps(document))
        return ledger, document

    def test_preserved_realistic_charge_and_new_completed_charge_admit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, data = self.fixture(root)
            with patch.object(module, 'OWN', root):
                self.assertEqual(module.admit_allocation(path)['used_total_seconds'], 1234)
                data['runs'].append(self.closed(78.25))
                path.write_text(json.dumps(data))
                self.assertEqual(module.admit_allocation(path)['used_total_seconds'], 1312.25)

    def test_historical_charge_edit_and_active_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, data = self.fixture(root)
            with patch.object(module, 'OWN', root):
                edited = copy.deepcopy(data)
                edited['runs'][0]['elapsed_seconds'] -= 1
                path.write_text(json.dumps(edited))
                with self.assertRaisesRegex(ValueError, 'Historical actual'):
                    module.admit_allocation(path)
                active = self.closed(10)
                active['status'] = 'running'
                data['runs'].append(active)
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'unfinished'):
                    module.admit_allocation(path)

    def test_total_growth_invalid_charge_and_exceeded_stage_are_rejected(self):
        grown = dict(self.limits, training=16801)
        with self.assertRaisesRegex(ValueError, 'finite57600'):
            module.valid_limits(grown)
        for value in (-1, float('nan'), True):
            with self.assertRaisesRegex(ValueError, 'Invalid historical'):
                module.valid_rows([self.closed(value)], self.limits)
        with self.assertRaisesRegex(ValueError, 'exceed proposed'):
            module.valid_rows([self.closed(39601)], self.limits)

    def test_actual_activation_preserves_prior_total_and_rows_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own = root / 'own'
            own.mkdir()
            rows = [self.closed(1234.25)]
            old = own / 'gpu_budget_allocation_user_resume_v6.json'
            old.write_text(json.dumps({'synthetic': True, 'original_total': 43200}))
            ledger = own / 'gpu_budget.json'
            previous = {'runs': rows, 'total_limit_seconds': 43200,
                        'stage_limits_seconds': {'preflight': 1200, 'training': 14400, 'evaluation': 27600},
                        'active_allocation_sha256': sha(old), 'synthetic': True}
            ledger.write_text(json.dumps(previous))
            queue = root / '.scratch/vla_gpu_coordination/queue.json'
            queue.parent.mkdir(parents=True)
            queue.write_text(json.dumps({'tickets': []}))
            admission = {'path': str(old), 'sha256': sha(old)}
            with patch.object(module, 'OWN', own), patch.object(module, 'ROOT', root), \
                    patch('experiments.lingbot_vla2_g1.budget_reallocated_v6.admit_allocation', return_value=admission):
                value = module.activate(self.limits, 'Synthetic CPU completion allocation; no GPU.')
            changed = json.loads(ledger.read_text())
            self.assertEqual(value['total_limit_seconds'], 57600)
            self.assertEqual(changed['runs'], rows)
            self.assertEqual(changed['total_limit_seconds'], 57600)
            self.assertEqual(value['used_total_seconds'], 1234.25)
            self.assertEqual(json.loads((own / module.SNAPSHOT).read_text()), previous)
            allocation = json.loads((own / module.ALLOCATION).read_text())
            self.assertEqual(allocation['previous_total_limit_seconds'], 43200)
            self.assertTrue(allocation['historical_charges_not_reset'])

    def test_activation_refuses_owned_waiter_before_any_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own = root / 'own'
            own.mkdir()
            (own / 'gpu_budget.json').write_text(json.dumps({'runs': [self.closed()]}))
            queue = root / '.scratch/vla_gpu_coordination/queue.json'
            queue.parent.mkdir(parents=True)
            queue.write_text(json.dumps({'tickets': [{'owner': module.OWNER, 'status': 'waiting'}]}))
            with patch.object(module, 'OWN', own), patch.object(module, 'ROOT', root), \
                    patch('experiments.lingbot_vla2_g1.budget_reallocated_v6.admit_allocation', return_value={}):
                with self.assertRaisesRegex(ValueError, 'owned tickets'):
                    module.activate(self.limits, 'Synthetic CPU regression')
            self.assertFalse((own / module.SNAPSHOT).exists())
            self.assertFalse((own / module.ALLOCATION).exists())


if __name__ == '__main__':
    unittest.main()
