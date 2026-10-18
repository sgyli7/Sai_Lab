import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1 import make_plan as factory
from experiments.lingbot_vla2_g1.benchmark import decoded_evidence, carry_metrics
from experiments.lingbot_vla2_g1.data import OWN, sha


class DecoderEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        path = root / 'decoded.jsonl'
        norm = OWN / 'train_meanstd.json'
        identities = {'norm_sha256': str(norm), 'contract_sha256': str(OWN / 'contract.json'), 'base_conversion_receipt_sha256': str(OWN / 'bf16_conversion_receipt.json'), 'runtime_overlay_receipt_sha256': str(OWN / 'runtime_overlay_receipt.json')}
        loaded = {'event': 'model_loaded', 'parameter_elements': 6375906359, 'trainable_elements': 1938090551, 'norm_path': str(norm), 'delta': None, **{key: 'hash-' + key for key in identities}}
        actions = [[0.] * 32 for _ in range(50)]
        event = {'event': 'lingbot_decoded', 'control_tick': 0, 'episode_id': 'case1', 'actions32': actions, 'actions': actions, 'action_horizon': 50, 'inference_seconds': .5}
        config = {'policy_command': [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service', '--norm-stats', str(norm), '--decoded-evidence', str(path)], 'output_dir': str(root), 'frozen_files': {value: 'hash-' + key for key, value in identities.items()}, 'case': {'case_id': 'case1'}}
        result = {'inference_calls': [{'observation_control_tick': 0, 'inference_seconds': .5}]}
        return path, loaded, event, config, result

    def test_exact_model_identity_actions_and_actual_queries_required(self):
        with tempfile.TemporaryDirectory() as root:
            path, loaded, event, config, result = self.fixture(Path(root))
            path.write_text('\n'.join(map(json.dumps, [loaded, event])))
            self.assertEqual(decoded_evidence(config, result)[0], path)
            event['control_tick'] = 50
            path.write_text('\n'.join(map(json.dumps, [loaded, event])))
            with self.assertRaises(ValueError):
                decoded_evidence(config, result)

    def test_wrong_weights_or_rejected_query_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            path, loaded, event, config, result = self.fixture(Path(root))
            for field, value in [('norm_sha256', 'different'), ('delta', '/another/delta')]:
                wrong = {**loaded, field: value}
                path.write_text('\n'.join(map(json.dumps, [wrong, event])))
                with self.assertRaises(ValueError):
                    decoded_evidence(config, result)
            event['event'] = 'request_rejected'
            path.write_text('\n'.join(map(json.dumps, [loaded, event])))
            with self.assertRaises(ValueError):
                decoded_evidence(config, result)


class SelectionTests(unittest.TestCase):
    def prepare(self, root):
        (root / 'base_bf16').mkdir()
        for name in ['train_meanstd.json', 'contract.json', 'runtime_overlay_receipt.json']:
            (root / name).write_text('{}')
        shards = []
        for index in range(6):
            path = root / 'base_bf16' / ('shard' + str(index) + '.safetensors')
            path.write_bytes(('test-only-' + str(index)).encode())
            shards.append({'output_sha256': sha(path)})
        (root / 'bf16_conversion_receipt.json').write_text(json.dumps({'status': 'completed', 'shards': shards}))
        split = root / 'split.json'
        split.write_text(json.dumps({'validation': list(range(10)), 'files': [{'episode_id': i, 'length': 51} for i in range(10)]}))
        metric = lambda value: {'frames': 510, 'chunks': 20, 'episodes': 10, 'complete': True, 'flow_loss': value, 'total_loss': value + .01}
        run = root / 'training'
        run.mkdir()
        baseline = metric(2.)
        # Base wins in this fixture. A real update still happened and is retained
        # as final, but evaluation cannot silently choose it over validation.
        result = {'schema': 'lingbot_vla2_g1_training_v1', 'status': 'completed', 'test_opened': False, 'optimizer_steps': 100, 'split_sha256': sha(split), 'supplementary_manifest': None, 'norm_path': str(root / 'train_meanstd.json'), 'norm_sha256': sha(root / 'train_meanstd.json'), 'base_receipt_sha256': sha(root / 'bf16_conversion_receipt.json'), 'contract_sha256': sha(root / 'contract.json'), 'runtime_overlay_receipt_sha256': sha(root / 'runtime_overlay_receipt.json'), 'baseline_validation': baseline, 'selected_step': 0, 'selected_validation': baseline, 'selected_delta': None}
        result_path = run / 'training_result.json'
        result_path.write_text(json.dumps(result))
        history = [{'event': 'validation_complete', 'step': 0, 'validation': baseline}]
        history += [{'event': 'optimizer_step', 'step': step} for step in range(1, 101)]
        history += [{'event': 'validation_complete', 'step': 100, 'validation': metric(3.)}, {'event': 'training_stopped', 'step': 100}, {'event': 'training_completed', 'step': 100}]
        (run / 'progress.jsonl').write_text('\n'.join(map(json.dumps, history)))
        (run / 'recipe.json').write_text('{}')
        return result_path, result, split

    def test_unmodified_base_can_win_but_step100_cannot_replace_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, result, split = self.prepare(root)
            with patch.object(factory, 'OWN', root), patch.object(factory, 'SPLIT', split), patch.object(factory, 'SPLIT_SHA', sha(split)):
                self.assertIsNone(factory.verify_selection(path)[1])
                result['selected_step'] = 100
                path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    factory.verify_selection(path)

    def test_partial_validation_never_certifies_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, result, split = self.prepare(root)
            history = path.parent / 'progress.jsonl'
            entries = [json.loads(line) for line in history.read_text().splitlines()]
            next(entry for entry in entries if entry['event'] == 'validation_complete' and entry['step'] == 100)['validation']['frames'] = 509
            history.write_text('\n'.join(map(json.dumps, entries)))
            with patch.object(factory, 'OWN', root), patch.object(factory, 'SPLIT', split), patch.object(factory, 'SPLIT_SHA', sha(split)):
                with self.assertRaises(ValueError):
                    factory.verify_selection(path)


class TransportReportTests(unittest.TestCase):
    def test_uses_audited_event_and_keeps_full_task_failure(self):
        judgment = {'carry_qualified_control_tick': 3, 'first_grasp_control_tick': 1, 'success': False}
        audit = {'recomputed_acceptance': judgment}
        rows = [{'control_tick': i, 'monotonic_ns': 1000000000 + i * 50000000} for i in range(4)]
        result = carry_metrics(audit, rows)
        self.assertTrue(result['long_carry_qualified'])
        self.assertEqual(result['time_to_qualified_carry_sim_seconds'], .06)
        self.assertEqual(result['time_to_qualified_carry_wall_seconds'], .15)
        self.assertEqual(result['grasp_to_qualified_carry_sim_seconds'], .04)
        self.assertFalse(judgment['success'])
        judgment['carry_qualified_control_tick'] = None
        self.assertFalse(carry_metrics(audit, rows)['long_carry_qualified'])


if __name__ == '__main__':
    unittest.main()
