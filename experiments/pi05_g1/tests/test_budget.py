import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('owned_budget', Path(__file__).resolve().parents[1] / 'budget.py')
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)

class BudgetTests(unittest.TestCase):
    def test_memory_cap_reserves_host_headroom(self):
        high, maximum = budget.memory_limits(80, 90, 12, 60 * budget.GIB)
        self.assertEqual(maximum, 48 * budget.GIB)
        self.assertLess(high, maximum)
        with self.assertRaises(RuntimeError):
            budget.memory_limits(80, 90, 12, 10 * budget.GIB)

    def test_memory_limits_match_kernel_page_floor_without_spending_reserve(self):
        page = budget.os.sysconf('SC_PAGE_SIZE')
        maximum_before = 95121645568
        high, maximum = budget.memory_limits(80, 90, 16, maximum_before + 16*budget.GIB)
        self.assertEqual(maximum, maximum_before)
        self.assertEqual(high, 85609480192)
        available = 60*budget.GIB + 1234
        high, maximum = budget.memory_limits(80, 90, 12, available)
        self.assertEqual(high % page, 0)
        self.assertEqual(maximum % page, 0)
        self.assertLessEqual(maximum, available - 12*budget.GIB)
        self.assertLessEqual(maximum, 90*budget.GIB)
        self.assertTrue(0 < high < maximum)
        # Existing small CPU probe limits remain unchanged and enforceable.
        self.assertEqual(budget.memory_limits(.046875, .0625, 12, 100*budget.GIB),
                         (48*1024**2, 64*1024**2))

    def test_foreign_unit_cannot_be_signalled(self):
        with self.assertRaises(ValueError):
            budget.signal_owned_unit('docker.service', 'SIGKILL')

    def test_native_service_bounds_and_hard_runtime(self):
        with patch.object(budget.shutil, 'which', return_value='/usr/bin/python3'):
            args = budget.service_command('owned', ['python3', '-c', 'pass'], 32, 64, 3, 1)
        self.assertIn('--property=MemoryMax=64', args)
        self.assertIn('--property=MemoryHigh=32', args)
        self.assertIn('--property=MemorySwapMax=0', args)
        self.assertIn('--property=OOMPolicy=kill', args)
        self.assertIn('--property=RuntimeMaxSec=3s', args)
        self.assertIn('--property=KillMode=control-group', args)

    def test_missing_container_errors_are_case_insensitive_but_explicit(self):
        for message in ('error: no such object: abc', 'Error: No such object',
                        'Error response from daemon: No such container: abc'):
            self.assertTrue(budget.explicitly_missing_container(message))
        for message in ('permission denied', 'Cannot connect to Docker daemon',
                        'object inspection failed', 'no such object permission'):
            self.assertFalse(budget.explicitly_missing_container(message))

    def test_docker_cleanup_verifies_token_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cid = 'a'*64
            row = {'schema':'pi05_g1_docker_ownership_v1', 'budget_token':'owned',
                   'launch_id':'launch', 'cidfile':str(Path(tmp)/'cid'),
                   'container_id':cid, 'status':'registered'}
            (Path(tmp)/'launch-a.json').write_text(json.dumps(row))
            responses = [subprocess.CompletedProcess([],0,cid+'\n',''),
                         subprocess.CompletedProcess([],0,json.dumps({'sai.pi05_g1.budget':'foreign'}),'')]
            with patch.object(budget.subprocess, 'run', side_effect=responses) as run:
                with self.assertRaises(ValueError):
                    budget.cleanup_owned_docker(tmp, 'owned', budget.time.monotonic()+10)
            self.assertFalse(any('kill' in call.args[0] or 'rm' in call.args[0] for call in run.call_args_list))

    def test_owned_container_is_verified_gone_not_merely_service_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            cid = 'b'*64
            row = {'schema':'pi05_g1_docker_ownership_v1', 'budget_token':'owned',
                   'launch_id':'launch', 'cidfile':str(Path(tmp)/'cid'),
                   'container_id':cid, 'status':'finished', 'removed':True}
            (Path(tmp)/'launch-a.json').write_text(json.dumps(row))
            responses = [subprocess.CompletedProcess([],0,cid,'whole list'),
                         subprocess.CompletedProcess([],0,json.dumps({'sai.pi05_g1.budget':'owned'}),''),
                         subprocess.CompletedProcess([],0,'',''), subprocess.CompletedProcess([],0,'',''),
                         subprocess.CompletedProcess([],1,'','error: no such object: '+cid),
                         subprocess.CompletedProcess([],0,'','')]
            with patch.object(budget.subprocess, 'run', side_effect=responses) as run:
                result = budget.cleanup_owned_docker(tmp, 'owned', budget.time.monotonic()+10)
            self.assertTrue(result['complete'])
            self.assertTrue(result['containers'][0]['removed'])
            self.assertTrue(any(call.args[0][1:3] == ['rm', '--force'] for call in run.call_args_list))

    def test_d_state_cleanup_is_bounded_and_history_not_forgiven(self):
        class DState:
            pid = 12345678
            def poll(self):
                return None
            def wait(self, timeout):
                self.wait_timeout = timeout
                raise subprocess.TimeoutExpired('mock-owned-launcher', timeout)
        child = DState()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'ledger.json'
            old = {'stage':'training', 'status':'budget_expired', 'elapsed_seconds':2496, 'allowed_seconds':1080}
            ledger.write_text(json.dumps({'runs':[old]}))
            argv = ['budget', '--ledger', str(ledger), '--stage', 'training', '--max-seconds', '1', '--', sys.executable, '-c', 'pass']
            ticks = iter(i*.1 for i in range(1000))
            with patch.object(sys, 'argv', argv), patch.object(budget, 'host_available_bytes', return_value=100*budget.GIB), \
                 patch.object(budget.subprocess, 'Popen', return_value=child), \
                 patch.object(budget, 'unit_properties', return_value={'ActiveState':'deactivating'}), \
                 patch.object(budget, 'signal_owned_unit') as kill, \
                 patch.object(budget.os, 'killpg'), patch.object(budget.time, 'sleep'), \
                 patch.object(budget.time, 'monotonic', side_effect=lambda:next(ticks)):
                with self.assertRaises(SystemExit):
                    budget.main()
            data = json.loads(ledger.read_text())
            self.assertEqual(data['runs'][0], old)
            pending = data['runs'][-1]
            self.assertEqual(pending['status'], 'cleanup_pending')
            self.assertTrue(pending['accounting_open'])
            self.assertTrue(budget.blocks_new_launch(pending))
            self.assertNotIn('ended_utc', pending)
            self.assertEqual(child.wait_timeout, 1)
            self.assertTrue(any(call.args[1] == 'SIGKILL' for call in kill.call_args_list))

if __name__ == '__main__':
    unittest.main()
