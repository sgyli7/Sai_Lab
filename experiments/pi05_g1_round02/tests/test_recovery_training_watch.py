import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pi05_g1_round02 import recovery_training_watch as watch


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'run02'
        self.output.mkdir()
        self.progress = self.output / 'progress.jsonl'
        self.progress.write_text('')
        self.ledger = self.root / 'ledger.json'
        self.meminfo = self.root / 'meminfo'
        self.meminfo.write_text('MemAvailable: 1234 kB\n')
        self.reader = watch.ProgressReader(self.progress)

    def write_rows(self, rows):
        self.progress.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def run_snapshot(self, rows):
        self.ledger.write_text(json.dumps({'runs': rows}))
        return watch.snapshot(self.reader, self.ledger, self.root, self.meminfo)

    def test_progress_complete_validation_and_trim(self):
        self.write_rows([
            {'event': 'training_started', 'step': 0, 'inherited_round_updates': 200},
            {'event': 'validation', 'step': 0, 'complete': True, 'metrics': {'combined_flow_loss': .1}},
            {'event': 'optimizer_step', 'step': 3, 'flow_loss': .2},
            {'event': 'validation', 'step': 3, 'complete': False, 'metrics': None},
            {'event': 'checkpoint_saved', 'step': 3, 'cpu_staging_cleanup': [
                {'phase': 'after_model_serialize', 'before': {'VmRSS_bytes': 100},
                 'after': {'VmRSS_bytes': 20}, 'trim_available': True, 'trim_return': 1}]}])
        result = self.run_snapshot([])
        self.assertEqual((result['local_step'], result['cumulative_step'], result['loss']), (3, 203, .2))
        self.assertEqual(result['last_complete_validation']['local_step'], 0)
        self.assertEqual(result['cpu_staging_trim'][0]['after_rss_bytes'], 20)
        self.assertEqual(result['host_memavailable_bytes'], 1234*1024)

    def test_exact_output_latest_run_and_closed_failure(self):
        foreign = {'command': ['train', '--output', str(self.output)+'_other'],
                   'status': 'completed', 'ended_utc': 'then'}
        owned = {'command': ['train', '--output=run02'], 'status': 'host_memory_guard',
                 'ended_utc': 'now', 'docker_cleanup': {'complete': True}}
        self.assertFalse(self.run_snapshot([foreign])['final'])
        self.assertTrue(self.run_snapshot([foreign, owned])['final'])
        running = {**owned, 'status': 'running'}
        self.assertFalse(self.run_snapshot([owned, running])['final'])
        for blocked in ({'accounting_open': True}, {'launcher_cleanup_pending': True},
                        {'docker_cleanup': {'complete': False}}, {'ended_utc': None}):
            self.assertFalse(self.run_snapshot([{**owned, **blocked}])['final'])

    def test_partial_line_rotation_and_malformed(self):
        line = json.dumps({'event': 'optimizer_step', 'step': 2, 'flow_loss': .3})
        self.progress.write_text(line)
        self.assertNotIn('local_step', self.reader.update())
        with self.progress.open('a') as stream:
            stream.write('\nnot json\n')
        self.assertEqual(self.reader.update()['local_step'], 2)
        self.assertEqual(self.reader.malformed_lines, 1)
        self.progress.unlink()
        self.write_rows([{'event': 'training_started', 'step': 0}])
        self.assertEqual(self.reader.update()['local_step'], 0)
        self.assertNotIn('loss', self.reader.state)

    def test_main_exits_on_closed_and_never_sleeps(self):
        self.ledger.write_text(json.dumps({'runs': [{'command': ['train', '--output', str(self.output)],
            'status': 'completed', 'ended_utc': 'now'}]}))
        with patch.object(watch.time, 'sleep') as sleep, patch('builtins.print') as output:
            watch.main(['--progress', str(self.progress), '--ledger', str(self.ledger)])
        sleep.assert_not_called()
        self.assertTrue(json.loads(output.call_args.args[0])['final'])
        with self.assertRaises(SystemExit):
            watch.main(['--progress', str(self.progress), '--ledger', str(self.ledger), '--interval-seconds', '1'])

    def test_unreadable_ledger_is_not_terminal(self):
        self.ledger.write_text('{')
        result = watch.snapshot(self.reader, self.ledger, self.root, self.meminfo)
        self.assertFalse(result['final'])
        self.assertTrue(any('ledger_read_error' in row for row in result['diagnostics'] if isinstance(row, dict)))


if __name__ == '__main__':
    unittest.main()
