"""A new checkout must prepare inputs without rewriting a sealed experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

from sim2sim.research.bundle import bundle_paths
from sim2sim.research.setup import initialize
from sim2sim.research.tasks import TASKS


def tiny_actor():
    indices = helper.make_tensor('indices', TensorProto.INT64, [14], list(range(14)))
    graph = helper.make_graph([helper.make_node('Gather', ['obs', 'indices'], ['actions'], axis=1)],
        'fixture', [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 14])], [indices])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9).SerializeToString()


class SessionSetup(unittest.TestCase):
    def test_snapshots_are_independent_and_an_existing_budget_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); supplied = root / 'supplied'; supplied.mkdir()
            raw = tiny_actor()
            for task in TASKS.values():
                (supplied / task.factory).write_bytes(raw)
            out = root / 'new_session'
            record = initialize(out, supplied, hours=.5)
            self.assertEqual(record['deadline_unix'] - record['start_unix'], 1800)
            self.assertEqual(len(record['baseline_weights']), 9)
            self.assertEqual(len(record['missing_previous']), 9)
            (supplied / 'alpha_walking.onnx').write_bytes(b'changed after snapshot')
            self.assertEqual((out / 'baseline/alpha_walking.onnx').read_bytes(), raw)
            before = (out / 'session.json').read_bytes()
            with self.assertRaises(FileExistsError):
                initialize(out, supplied, hours=8)
            self.assertEqual((out / 'session.json').read_bytes(), before)

    def test_missing_inputs_do_not_create_an_apparently_initialized_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            with self.assertRaises(FileNotFoundError):
                initialize(out, tmp)
            self.assertFalse(out.exists())

    def test_invalid_budget_is_rejected_before_creating_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for hours in (0, -1, float('nan'), float('inf')):
                with self.subTest(hours=hours), self.assertRaises(ValueError):
                    initialize(Path(tmp) / 'new', tmp, hours)
            self.assertFalse((Path(tmp) / 'new').exists())

    def test_new_process_resolves_every_factory_under_the_selected_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, SIM2SIM_RESEARCH_DIR=tmp)
            code = 'from sim2sim.research.tasks import TASKS; print(TASKS["walking"].source)'
            output = subprocess.check_output([sys.executable, '-c', code], env=env, text=True)
            self.assertEqual(Path(output.strip()), Path(tmp) / 'baseline/alpha_walking.onnx')


class RelocatedBundle(unittest.TestCase):
    def make_bundle(self, root):
        (root / 'models').mkdir()
        record = dict(verification_completed_unix=1, skills={})
        raw = tiny_actor(); side = b'{"schema_version": 2}'
        for name, task in TASKS.items():
            path = root / 'models' / task.previous
            path.write_bytes(raw); path.with_suffix('.manifest.json').write_bytes(side)
            record['skills'][name] = dict(exported='/missing/original/machine/' + task.previous,
                sha256=hashlib.sha256(raw).hexdigest(), manifest_sha256=hashlib.sha256(side).hexdigest())
        (root / 'bundle.json').write_text(json.dumps(record))
        (root / 'bank.json').write_text('{"walking": "/missing/original/machine/Walk_Godot.onnx"}')

    def test_resolves_relocated_files_without_rewriting_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_bundle(root)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            bank = bundle_paths(root)
            self.assertEqual(len(bank), 9)
            self.assertTrue(all(Path(p).parent == root / 'models' for p in bank.values()))
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_tampered_model_or_manifest_is_rejected(self):
        for filename in ('Walk_Godot.onnx', 'Walk_Godot.manifest.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.make_bundle(root)
                (root / 'models' / filename).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    bundle_paths(root)

    def test_missing_skill_is_not_replaced_by_a_default_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_bundle(root)
            record = json.loads((root / 'bundle.json').read_text())
            del record['skills']['kick_right']
            (root / 'bundle.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'exactly nine'):
                bundle_paths(root)

    def test_full_holdout_requires_all_comparators_before_creating_output(self):
        from sim2sim.research.holdout import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_bundle(root)
            out = root / 'new_holdout'
            with patch('sim2sim.research.holdout.BASELINE', root / 'missing_baseline'):
                with self.assertRaisesRegex(FileNotFoundError, 'require-previous'):
                    run(root, out=out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
