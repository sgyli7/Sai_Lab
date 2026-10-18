import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.pi05_g1_round02.quiescence_probe import probe


class QuiescenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / 'ledger.json'
        self.token = 'a' * 32
        self.cid = 'b' * 64
        self.row = {'status': 'completed', 'ended_utc': 'then', 'elapsed_seconds': 12.5,
                    'budget_token': self.token, 'owned_unit': f'pi05-g1-owned-{self.token}.service',
                    'docker_cleanup': {'complete': True, 'containers': [{'container_id': self.cid}]}}
        self.compute = ''
        self.active = False
        self.container = None
        self.query_error = False
        self.calls = []

    def command(self, argv, **kwargs):
        self.calls.append(argv)
        code, out, err = 0, '', ''
        if argv[0] == 'systemctl':
            out = 'ActiveState=' + ('active' if self.active else 'inactive') + '\nSubState=dead\nResult=success\n'
        elif argv[:2] == ['docker', 'inspect']:
            if self.container is None:
                code, err = 1, 'Error: No such object: ' + self.cid
            else:
                out = json.dumps([self.container])
        elif argv[0] == 'nvidia-smi':
            out = self.compute
            if self.query_error:
                code, err = 1, 'driver unavailable'
        return subprocess.CompletedProcess(argv, code, out, err)

    def run_probe(self, name='probe'):
        self.ledger.write_text(json.dumps({'runs': [self.row]}))
        with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run', side_effect=self.command):
            return probe(self.ledger, self.root / (name+'.json'))

    def test_closed_and_command_evidence_bound(self):
        self.compute = 'GPU-123, 12, /usr/libexec/gnome-remote-desktop, 30\n'
        result = self.run_probe()
        self.assertTrue(result['passed'])
        self.assertEqual(result['total_gpu_seconds'], 12.5)
        self.assertEqual(len(result['allowed_display_background']), 1)
        self.assertTrue(result['container_checks'][0]['removed'])
        for path, digest in result['evidence_files'].items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        self.assertTrue(any(p.endswith('.command.json') for p in result['evidence_files']))
        self.assertTrue(all(c[0] in ('systemctl', 'docker', 'nvidia-smi') for c in self.calls))
        self.assertTrue(all(not set(c).intersection({'kill', 'stop', 'rm', 'run'}) for c in self.calls))
        self.assertIn('--user', self.calls[0])
        with self.assertRaises(FileExistsError):
            probe(self.ledger, self.root / 'probe.json')

    def test_extra_compute_and_failed_query_rejected(self):
        self.compute = 'GPU-123, 21, /venv/bin/python, 100\n'
        self.assertFalse(self.run_probe('extra')['passed'])
        self.compute, self.query_error = '', True
        self.assertFalse(self.run_probe('query_failure')['passed'])

    def test_exact_display_daemon_and_similar_names(self):
        self.compute = 'GPU-123, 12, /usr/libexec/gnome-remote-desktop-daemon, 176\n'
        result = self.run_probe('display_daemon')
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['allowed_display_background']), 1)
        for index, name in enumerate(('gnome-remote-desktop-daemon-fake',
                                      'fake-gnome-remote-desktop', 'gnome-remote-desktop.py')):
            self.compute = f'GPU-123, 12, /tmp/{name}, 176\n'
            result = self.run_probe(f'similar_{index}')
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['unexpected_compute_processes']), 1)

    def test_active_and_open_ledger_rejected(self):
        self.active = True
        result = self.run_probe('active')
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['active_owned_units']), 1)
        self.active = False
        self.row['accounting_open'] = True
        self.assertEqual(self.run_probe('open')['accounting_open_tokens'], [self.token])

    def test_running_container_and_foreign_label_rejected(self):
        self.container = {'Config': {'Labels': {'sai.pi05_g1.budget': self.token}},
                          'State': {'Running': True, 'Status': 'running'}}
        result = self.run_probe('running')
        self.assertFalse(result['passed'])
        self.assertEqual(result['active_owned_container_ids'], [self.cid])
        self.container['State'] = {'Running': False, 'Status': 'exited'}
        self.assertTrue(self.run_probe('stopped')['passed'])
        self.container['Config']['Labels']['sai.pi05_g1.budget'] = 'foreign'
        self.assertFalse(self.run_probe('foreign')['passed'])

    def test_systemctl_or_docker_permission_failure_not_absence(self):
        original = self.command
        def permission(argv, **kwargs):
            if argv[0] == 'systemctl' or argv[:2] == ['docker', 'inspect']:
                return subprocess.CompletedProcess(argv, 1, '', 'Permission denied')
            return original(argv, **kwargs)
        self.ledger.write_text(json.dumps({'runs': [self.row]}))
        with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run', side_effect=permission):
            self.assertFalse(probe(self.ledger, self.root / 'denied.json')['passed'])

    def test_collected_unit_and_ledger_race(self):
        original = self.command
        def collected(argv, **kwargs):
            if argv[0] == 'systemctl':
                return subprocess.CompletedProcess(argv, 1, '', 'Unit could not be found.')
            return original(argv, **kwargs)
        self.ledger.write_text(json.dumps({'runs': [self.row]}))
        with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run', side_effect=collected):
            self.assertTrue(probe(self.ledger, self.root / 'collected.json')['passed'])
        def raced(argv, **kwargs):
            if argv[0] == 'nvidia-smi':
                self.ledger.write_text('{}')
            return original(argv, **kwargs)
        with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run', side_effect=raced):
            self.assertFalse(probe(self.ledger, self.root / 'raced.json')['passed'])

    def test_missing_user_bus_is_not_collected_unit(self):
        original = self.command
        def bus_error(argv, **kwargs):
            if argv[0] == 'systemctl':
                return subprocess.CompletedProcess(argv, 1, '', 'Failed to connect to bus: endpoint not found')
            return original(argv, **kwargs)
        self.ledger.write_text(json.dumps({'runs': [self.row]}))
        with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run', side_effect=bus_error):
            self.assertFalse(probe(self.ledger, self.root / 'bus_error.json')['passed'])


if __name__ == '__main__':
    unittest.main()
