import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim2sim.research.budget import ActiveBudget, _boot_id, process_start_ticks
from sim2sim.research.queue import ExperimentQueue


class ExperimentQueueTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.directory=Path(temporary.name)
        self.budget=ActiveBudget(self.directory);self.budget.initialize(6000)
        self.source=self.directory/'source';self.source.write_text('frozen')
        self.queue=ExperimentQueue(self.directory)
        self.item=dict(id='one',skill='standing',hypothesis='less drift at unchanged task success',
            source=str(self.source),control_version='v1',timeout_seconds=10,
            command=['python','train','--name','{attempt_id}'],
            evidence=[dict(path='{run_dir}/parity.json',equals={'passed':True})],max_attempts=2)

    def initialize(self,**changes):
        self.item.update(changes);self.queue.initialize(dict(experiments=[self.item]))

    def state(self):return json.loads(self.queue.path.read_text())['experiments'][0]

    def test_directory_alone_does_not_mean_completed(self):
        self.initialize()
        (self.directory/'runs/one_attempt_01').mkdir(parents=True)
        with patch('sim2sim.research.queue.run_supervised',return_value=0):self.queue.execute()
        self.assertEqual(self.state()['status'],'failed')
        self.assertIn('Missing experiment evidence',self.state()['attempts'][0]['reason'])

    def test_append_preserves_prior_attempts_and_rejects_duplicate_ids(self):
        self.initialize()
        with patch('sim2sim.research.queue.run_supervised',return_value=1):self.queue.execute()
        before=self.state()
        item={**self.item,'id':'two'}
        self.queue.initialize({'experiments':[item]},append=True)
        self.assertEqual(self.state(),before)
        with self.assertRaises(ValueError):self.queue.initialize({'experiments':[item]},append=True)

    def test_success_requires_passed_evidence_and_separate_decision(self):
        self.initialize()
        path=self.directory/'runs/one_attempt_01';path.mkdir(parents=True)
        (path/'parity.json').write_text('{"passed": true}')
        with patch('sim2sim.research.queue.run_supervised',return_value=0):self.queue.execute()
        self.assertEqual(self.state()['status'],'completed')
        self.assertIsNone(self.state()['decision'])
        self.queue.decide('one','keep_incumbent','No paired task improvement')
        self.assertEqual(self.state()['decision']['decision'],'keep_incumbent')

    def test_failed_commands_are_bounded_and_not_promotable(self):
        self.initialize(retry_failed=True)
        with patch('sim2sim.research.queue.run_supervised',return_value=1) as run:
            self.queue.execute();self.queue.execute()
        self.assertEqual(run.call_count,2)
        with self.assertRaises(ValueError):self.queue.decide('one','candidate_for_freeze','invalid')

    def test_insufficient_budget_never_starts(self):
        self.initialize(timeout_seconds=601)
        with patch('sim2sim.research.queue.run_supervised') as run:
            self.assertEqual(self.queue.execute(),'insufficient_budget')
        run.assert_not_called()

    def test_resume_uses_only_complete_checkpoint_and_new_attempt(self):
        self.initialize(resume_complete_checkpoint=True)
        path=self.directory/'runs/one_attempt_01';path.mkdir(parents=True)
        (path/'latest.pt').write_bytes(b'complete')
        (path/'latest.pt.partial').write_bytes(b'interrupted')
        with patch('sim2sim.research.queue.run_supervised',return_value=1) as run:
            self.queue.execute();self.queue.execute()
        self.assertEqual(run.call_args_list[1].args[1][-2:],['--resume',str(path/'latest.pt')])
        self.assertEqual(self.state()['attempts'][1]['id'],'one_attempt_02')

    def test_detached_running_job_is_not_duplicated_after_scheduler_loss(self):
        self.initialize()
        data=json.loads(self.queue.path.read_text());item=data['experiments'][0]
        item.update(status='running',attempts=[dict(id='one_attempt_01',status='running')])
        self.queue.path.write_text(json.dumps(data))
        self.budget.event('job_started',actor='one_attempt_01:abc',pid=os.getpid(),
            boot=_boot_id(),process_start_ticks=process_start_ticks(os.getpid()))
        with patch('sim2sim.research.queue.run_supervised') as run:
            self.assertEqual(self.queue.execute(),'prior_job_still_running')
        run.assert_not_called()


if __name__=='__main__':unittest.main()
