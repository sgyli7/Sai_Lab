"""CPU mockqueue only; real budget validation tested by its specialist separately."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import round03_driver_allocated as subject
from experiments.pi05_g1_mobile import round03_driver as legacy
from experiments.pi05_g1_mobile.tests import test_round03_driver as fixtures


class AllocatedDriverTests(unittest.TestCase):
    def fixture(self,root):
        path,plan=fixtures.DriverTests().fixture(root)
        plan['frozen_files']={str(Path(subject.__file__).resolve()):legacy.sha(subject.__file__),
            str(Path(subject.__file__).with_name('budget_allocated.py').resolve()):legacy.sha(Path(subject.__file__).with_name('budget_allocated.py'))}
        path.write_text(json.dumps(plan));return path,plan

    def test_budget_command_only_module_differs(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(legacy,'MOBILE',Path(tmp)):
            path,plan=self.fixture(Path(tmp));stage=plan['stages'][0]
            old=subject._ORIGINAL_COMMAND(plan,stage);new=subject.budget_command(plan,stage)
            self.assertEqual(new[:2],old[:2]);self.assertEqual(new[3:],old[3:])
            self.assertEqual(new[2],'experiments.pi05_g1_mobile.budget_allocated')

    def test_dryrun_checks_allocation_no_queue_or_process_and_restores(self):
        from experiments.pi05_g1_mobile import budget_allocated
        with tempfile.TemporaryDirectory() as tmp,patch.object(legacy,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);queue=fixtures.Queue();oldcommand=legacy.budget_command;oldvalidate=legacy.validate_plan
            with patch.object(budget_allocated,'validate_ledger',return_value={'limits':{}}) as check:
                result=subject.execute(path,root/'dry',dry_run=True,queue=queue,cpu_runner=lambda *x:self.fail('no CPU process'))
                self.assertGreaterEqual(check.call_count,1)
            self.assertEqual(result['status'],'dry_run');self.assertFalse(queue.calls)
            self.assertEqual(result['commands'][0]['argv'][2],'experiments.pi05_g1_mobile.budget_allocated')
            self.assertIs(legacy.budget_command,oldcommand);self.assertIs(legacy.validate_plan,oldvalidate)

    def test_allocation_checked_immediately_before_mockrequest(self):
        from experiments.pi05_g1_mobile import budget_allocated
        with tempfile.TemporaryDirectory() as tmp,patch.object(legacy,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);queue=fixtures.Queue();order=[];request=queue.request
            def checked_ledger(path):order.append('validate');return {}
            def request_after_check(*args):
                self.assertEqual(order[-1],'validate');order.append('request');return request(*args)
            queue.request=request_after_check
            with patch.object(budget_allocated,'validate_ledger',side_effect=checked_ledger):
                result=subject.execute(path,root/'run',queue=queue)
            self.assertEqual(result['status'],'completed');self.assertEqual(order[-1],'request')
            self.assertEqual(queue.command[2],'experiments.pi05_g1_mobile.budget_allocated')
            self.assertEqual(len(result['completed_stages']),1)

    def test_invalid_allocation_or_missing_new_source_pin_prevents_queue(self):
        from experiments.pi05_g1_mobile import budget_allocated
        for problem in ('ledger','pin'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as tmp,patch.object(legacy,'MOBILE',Path(tmp)):
                root=Path(tmp);path,plan=self.fixture(root);queue=fixtures.Queue()
                if problem=='pin':plan['frozen_files'].pop(str(Path(subject.__file__).resolve()));path.write_text(json.dumps(plan))
                with patch.object(budget_allocated,'validate_ledger',side_effect=ValueError('invalid allocation chain')):
                    with self.assertRaises(ValueError):subject.execute(path,root/'run',queue=queue)
                self.assertFalse(queue.calls);self.assertFalse((root/'run').exists())
                self.assertIs(legacy.budget_command,subject._ORIGINAL_COMMAND)
                self.assertIs(legacy.validate_plan,subject._ORIGINAL_VALIDATE)

    def test_legacy_predicates_cancellation_and_restore_after_exception(self):
        from experiments.pi05_g1_mobile import budget_allocated
        for mode in ('partial','cancelwaiting','cleanupunknown'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch.object(legacy,'MOBILE',Path(tmp)),patch.object(budget_allocated,'validate_ledger',return_value={}):
                root=Path(tmp);path,plan=self.fixture(root);queue=fixtures.Queue(mode)
                result=subject.execute(path,root/'run',queue=queue)
                self.assertEqual(result['status'],'attention');self.assertFalse(result['automatic_retry'])
                self.assertEqual(queue.cancelled,1 if mode=='cancelwaiting' else 0)
                self.assertIs(legacy.budget_command,subject._ORIGINAL_COMMAND)
                self.assertIs(legacy.validate_plan,subject._ORIGINAL_VALIDATE)
                with self.assertRaises(FileExistsError):subject.execute(path,root/'run',dry_run=True,queue=queue)
                self.assertIs(legacy.budget_command,subject._ORIGINAL_COMMAND)
                self.assertIs(legacy.validate_plan,subject._ORIGINAL_VALIDATE)


if __name__=='__main__':unittest.main()
