import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1.official_baseline.prepare import (
    BASE_MANIFEST,
    BASE_SHARD_SHA256,
    EXPECTED_SHARDS,
    GENERIC_BASE_REVISION,
    HF_CKPT,
    ROBOTWIN_REPO,
    ROBOTWIN_REVISION,
    UPSTREAM,
    admit_checkpoint,
    admit_command,
    admit_precision,
    admit_robotwin_revision,
    aarch64_pip_obstacle,
    assert_url_allowed,
    base_shards,
    build_dry_run,
    classify_release_files,
    documented_flash_attn_wheel,
    fp32_command,
    local_config_hashes,
    main,
    policy_argument_defaults,
    robotwin_revision_from_readme,
    seal,
    training_yaml_for_model,
    DownloadBudget,
    current_storage_report,
)


def official_checkpoint(**overrides):
    checkpoint = {
        'repo_id': ROBOTWIN_REPO,
        'revision': 'a' * 40,
        'hf_ckpt': HF_CKPT,
        'weights_downloaded': False,
        'shards': [
            {'name': name, 'sha256': f'{index + 1:064x}', 'size': 1000 + index}
            for index, name in enumerate(EXPECTED_SHARDS)
        ],
        'index_shards': list(EXPECTED_SHARDS),
    }
    checkpoint.update(overrides)
    return checkpoint


class OfficialBaselineTests(unittest.TestCase):
    def test_current_combined_storage_supersedes_historical_own_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned, backup = root / 'owned', root / 'backup'
            owned.mkdir(); backup.mkdir()
            (owned / 'small').write_bytes(b'1234')
            (backup / 'small').write_bytes(b'12')
            commitment = root / 'commitment.json'
            commitment.write_text(json.dumps({'combined_cap_bytes': 64,
                                               'prospective_remaining_growth_commitment_bytes': 24}))
            report = current_storage_report(30, 15, commitment_path=commitment,
                                            scratch=owned, backup=backup)
            self.assertEqual(report['combined_current_bytes'], 6)
            self.assertEqual(report['combined_cap_bytes'], 64)
            self.assertTrue(report['checkpoint_fits_with_existing_future_commitment'])
            self.assertFalse(report['baseline_compressed_files_fit_with_existing_future_commitment'])
            self.assertFalse(report['historical_README_own_cap_is_current_authority'])

    def test_rejects_generic_base_repo_and_base_shard_hashes(self):
        manifest = json.loads(BASE_MANIFEST.read_text())
        self.assertEqual(base_shards(manifest), BASE_SHARD_SHA256)
        self.assertEqual(manifest['sha'], GENERIC_BASE_REVISION)
        with self.assertRaisesRegex(ValueError, 'generic base weights'):
            admit_checkpoint(official_checkpoint(repo_id='robbyant/lingbot-vla-v2-6b', revision=GENERIC_BASE_REVISION))
        copied = official_checkpoint()
        copied['shards'][0]['sha256'] = BASE_SHARD_SHA256[EXPECTED_SHARDS[0]]
        with self.assertRaisesRegex(ValueError, 'generic base weights'):
            admit_checkpoint(copied)
        with self.assertRaisesRegex(ValueError, 'hf_ckpt'):
            admit_checkpoint(official_checkpoint(hf_ckpt='.'))

    def test_rejects_bf16_and_policy_module_defaults(self):
        with self.assertRaisesRegex(ValueError, 'BF16'):
            admit_precision(True, False)
        with self.assertRaisesRegex(ValueError, 'BF16'):
            admit_precision(True, True)
        with self.assertRaisesRegex(ValueError, 'FP32'):
            admit_precision(False, False)
        admit_precision(False, True)
        source = (UPSTREAM / 'deploy' / 'lingbot_vla_v2_policy.py').read_text()
        defaults = policy_argument_defaults(source)
        self.assertTrue(defaults['use_bf16'])
        self.assertFalse(defaults['use_fp32'])
        with self.assertRaisesRegex(ValueError, 'BF16'):
            admit_precision(defaults['use_bf16'], defaults['use_fp32'])
        command = fp32_command('demo_clean')
        argv = list(command['argv'])
        argv[argv.index('--use_bf16') + 1] = 'True'
        argv[argv.index('--use_fp32') + 1] = 'False'
        command['argv'] = argv
        with self.assertRaisesRegex(ValueError, 'BF16'):
            admit_command(command)

    def test_rejects_wrong_robotwin_revision_and_base_model_path(self):
        readme = (UPSTREAM / 'experiment' / 'robotwin' / 'README.md').read_text()
        self.assertEqual(robotwin_revision_from_readme(readme), ROBOTWIN_REVISION)
        with self.assertRaisesRegex(ValueError, 'RoboTwin'):
            robotwin_revision_from_readme(readme.replace(ROBOTWIN_REVISION, 'b' * 40))
        with self.assertRaisesRegex(ValueError, 'RoboTwin'):
            admit_robotwin_revision('main')
        with self.assertRaisesRegex(ValueError, 'RoboTwin'):
            admit_robotwin_revision(GENERIC_BASE_REVISION)
        command = fp32_command('demo_clean')
        argv = list(command['argv'])
        argv[argv.index('--model_path') + 1] = '/NOT_PRESENT/lingbot-vla-v2-6b/checkpoints/global_step_50000/hf_ckpt'
        command['argv'] = argv
        with self.assertRaisesRegex(ValueError, 'generic base'):
            admit_command(command)
        self.assertEqual(
            training_yaml_for_model(Path('/repo/checkpoints/global_step_50000/hf_ckpt')),
            Path('/repo/lingbotvla_cli.yaml'),
        )

    def test_dry_run_is_not_executed_and_does_not_fetch(self):
        def boom(*_args, **_kwargs):
            raise AssertionError('network')

        with patch('experiments.lingbot_vla2_g1.official_baseline.prepare.urlopen', boom), tempfile.TemporaryDirectory() as directory:
            main(['--dry-run', '--output', directory])
            saved = json.loads((Path(directory) / 'preparation.json').read_text())
        self.assertEqual(saved['preparation_probe'], 'NOT_EXECUTED')
        self.assertEqual(saved['official_baseline_execution'], 'NOT_EXECUTED')
        self.assertFalse(saved['success_rate_reproduced'])
        self.assertFalse(saved['runnable_on_this_host'])
        self.assertNotIn('checkpoint', saved)
        self.assertNotIn('93.52%', json.dumps(saved))
        mutated = build_dry_run(Path(directory))
        mutated['success_rate_reproduced'] = True
        with self.assertRaisesRegex(ValueError, 'not reproduced'):
            seal(mutated)

    def test_aarch64_wheel_gap_and_download_guard(self):
        blocked = classify_release_files([
            'sapien-3.0.0b1-cp311-cp311-manylinux2014_x86_64.whl',
            'sapien-3.0.0b1-cp311-cp311-win_amd64.whl',
        ])
        self.assertEqual(aarch64_pip_obstacle(blocked), 'no_linux_aarch64_wheel_and_no_sdist')
        untested = classify_release_files([
            'mplib-0.2.1-cp311-cp311-manylinux2014_x86_64.whl',
            'mplib-0.2.1.tar.gz',
        ])
        self.assertEqual(aarch64_pip_obstacle(untested), 'no_linux_aarch64_wheel_sdist_untested')
        present = classify_release_files(['open3d-0.18.0-cp311-cp311-manylinux_2_27_aarch64.whl'])
        self.assertIsNone(aarch64_pip_obstacle(present))
        wheel = documented_flash_attn_wheel((UPSTREAM / 'README.md').read_text())
        self.assertIn('linux_x86_64', wheel)
        self.assertNotIn('aarch64', wheel)
        with self.assertRaises(RuntimeError):
            DownloadBudget().add('model.safetensors', b'weights')
        with self.assertRaises(RuntimeError):
            assert_url_allowed(
                'https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.safetensors'
            )

    def test_local_camera_and_action_norm_identity(self):
        hashes = local_config_hashes(UPSTREAM)
        images = hashes['robot_config']['image_origin_keys']
        self.assertEqual(images['observation.images.camera_top'], 'observation.images.cam_high')
        self.assertEqual(images['observation.images.camera_wrist_left'], 'observation.images.cam_left_wrist')
        self.assertEqual(images['observation.images.camera_wrist_right'], 'observation.images.cam_right_wrist')
        self.assertEqual(hashes['robot_config']['norm_stats'], 'assets/norm_stats/robotwin.json')
        self.assertEqual(len(hashes['action_norm_stats']['sha256']), 64)
        norm = json.loads(Path(hashes['action_norm_stats']['path']).read_text())
        self.assertIn('action.arm.position', norm['norm_stats'])
        self.assertIn('action.effector.position', norm['norm_stats'])
        admit_checkpoint(official_checkpoint())


if __name__ == '__main__':
    unittest.main()
