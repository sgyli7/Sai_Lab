"""Reject changed charges and unsafe admission; expired old wall is historical."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import budget_continuation as b


class ContinuationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(b, 'OWN', self.root).start()
        row = {'stage': 'training', 'elapsed_seconds': 100., 'status': 'completed',
               'last_unit_properties': {'ActiveState': 'inactive'}, 'docker_cleanup': {'complete': True}}
        self.previous = self.root / 'previous.json'
        self.previous.write_text(json.dumps({'runs': [row]}))
        wall = self.root / 'original_goal_wall_deadline.json'
        wall.write_text(json.dumps({'wall_deadline_unix': 1}))
        self.allocation_path = self.root / 'gpu_budget_allocation_user_resume_v4.json'
        self.allocation = {'schema': 'lingbot_vla2_user_resume_allocation_v4',
            'total_limit_seconds': b.TOTAL, 'new_stage_limits_seconds': b.LIMITS,
            'user_authorization': '没成功你停下来个屁啊', 'old_wall_deadline_enforced': False,
            'previous_ledger_snapshot': str(self.previous), 'previous_ledger_sha256': b.sha(self.previous),
            'preserved_run_count': 1, 'preserved_runs_sha256': b.runs_digest([row]),
            'previous_wall_window_path': str(wall), 'previous_wall_window_sha256': b.sha(wall)}
        self.allocation_path.write_text(json.dumps(self.allocation))
        self.doc = {'runs': [row], 'total_limit_seconds': b.TOTAL, 'stage_limits_seconds': b.LIMITS,
                    'active_allocation_sha256': b.sha(self.allocation_path)}
        self.path = self.root / 'gpu_budget.json'

    def admit(self):
        self.path.write_text(json.dumps(self.doc))
        return b.admit_allocation(self.path)

    def test_expired_old_wall_does_not_reset_or_block_accounting(self):
        self.assertEqual(self.admit()['used_total_seconds'], 100.)
        self.assertFalse(self.admit()['old_wall_deadline_enforced'])

    def test_modified_historical_charge_rejected(self):
        self.doc['runs'][0]['elapsed_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'Historical'):
            self.admit()

    def test_unfinished_new_owner_rejected(self):
        self.doc['runs'].append({'stage': 'evaluation', 'elapsed_seconds': 0, 'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.admit()

    def test_changed_allocation_header_rejected(self):
        self.doc['active_allocation_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Ledger'):
            self.admit()

    def test_total_or_stage_exhaustion_rejected(self):
        row = {'stage': 'evaluation', 'elapsed_seconds': b.TOTAL, 'status': 'completed',
               'last_unit_properties': {'ActiveState': 'inactive'}, 'docker_cleanup': {'complete': True}}
        self.doc['runs'].append(row)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.admit()

    def test_finite_duration_passed_without_wall_clock_dependency(self):
        self.admit()
        argv = ['budget_continuation', '--ledger', str(self.path), '--stage', 'evaluation', '--max-seconds', '900', '--', 'worker']
        with patch.object(b.sys, 'argv', argv), patch.object(b.budget, 'main') as main:
            b.main()
        main.assert_called_once_with()
        self.assertEqual(argv[argv.index('--max-seconds') + 1], '900')
        self.assertEqual(b.budget.TOTAL, b.TOTAL)

    def test_nonfinite_and_duplicate_durations_rejected_before_admission(self):
        for flags in (['--max-seconds', 'nan'], ['--max-seconds', '-1'], ['--max-seconds', '1', '--max-seconds', '2']):
            with self.subTest(flags=flags), patch.object(b.sys, 'argv', ['entry', '--ledger', str(self.path), *flags]), self.assertRaisesRegex(ValueError, 'finite'):
                b.main()


if __name__ == '__main__':
    unittest.main()
