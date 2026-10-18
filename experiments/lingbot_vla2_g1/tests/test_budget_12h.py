"""The extension must reject altered old charges and unfinished ownership."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import budget_12h as b


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(b, 'OWN', self.root).start()
        row = {'stage': 'training', 'elapsed_seconds': 100,
               'status': 'completed', 'last_unit_properties': {'ActiveState': 'inactive'},
               'docker_cleanup': {'complete': True}}
        self.previous = self.root / 'previous.json'
        self.previous.write_text(json.dumps({'runs': [row], 'total_limit_seconds': 28800}))
        self.allocation = self.root / 'gpu_budget_allocation_12h_v3.json'
        self.allocation.write_text(json.dumps({
            'schema': 'lingbot_vla2_user_12h_allocation_v1',
            'total_limit_seconds': b.TOTAL, 'new_stage_limits_seconds': b.LIMITS,
            'user_authorization': '我也给你 12各小时时间 ，加油 做出最好成绩',
            'previous_ledger_snapshot': str(self.previous),
            'previous_ledger_sha256': b.sha(self.previous),
            'preserved_run_count': 1, 'preserved_runs_sha256': b.runs_digest([row])}))
        self.doc = {'runs': [row], 'total_limit_seconds': b.TOTAL,
                    'stage_limits_seconds': b.LIMITS,
                    'active_allocation_sha256': b.sha(self.allocation)}
        self.path = self.root / 'gpu_budget.json'

    def admit(self):
        self.path.write_text(json.dumps(self.doc))
        return b.admit_allocation(self.path)

    def test_actual_charge_preserved(self):
        self.assertEqual(self.admit()['used_total_seconds'], 100)

    def test_reduced_historical_charge_rejected(self):
        self.doc['runs'][0]['elapsed_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'Historical'):
            self.admit()

    def test_new_unfinished_owner_rejected(self):
        self.doc['runs'].append({'stage': 'evaluation', 'elapsed_seconds': 0,
                                 'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.admit()

    def test_old_header_rejected(self):
        self.doc['total_limit_seconds'] = 28800
        with self.assertRaisesRegex(ValueError, 'Ledger'):
            self.admit()

    def write_timing(self, **changes):
        timing = {'schema': 'lingbot_original_goal_wall_deadline_v1',
                  'goal_created_unix': 1792171987, 'wall_limit_seconds': 43200,
                  'wall_deadline_unix': 1792215187, 'queue_wait_counts_toward_wall': True}
        timing.update(changes)
        (self.root / 'original_goal_wall_deadline.json').write_text(json.dumps(timing))

    def test_queue_wait_cannot_restart_original_wall_clock(self):
        self.write_timing()
        with patch.object(b.time, 'time', return_value=1792215117):
            self.assertEqual(b.wall_remaining(), 60)
        with patch.object(b.time, 'time', return_value=1792215188):
            self.assertEqual(b.wall_remaining(), 0)

    def test_later_restarted_goal_rejected(self):
        self.write_timing(goal_created_unix=1792209982, wall_deadline_unix=1792253182)
        with self.assertRaisesRegex(ValueError, 'wall window'):
            b.wall_remaining()


if __name__ == '__main__':
    unittest.main()
