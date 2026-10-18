import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.lingbot_vla2_g1.archive_preserve import move_with_alias, inventory, require_acceptance
from experiments.lingbot_vla2_g1 import archive_preserve as implementation


class HistoryPreservationTests(unittest.TestCase):
    def test_byte_identical_move_preserves_original_and_nested_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'work'; source.mkdir()
            (source / 'model').write_bytes(b'uniquemodelbytes')
            (source / 'nested').mkdir(); (source / 'nested/video').write_bytes(b'videobytes')
            (source / 'model_alias').symlink_to('model')
            expected = inventory(source); target = root / 'permanent'
            self.assertEqual(move_with_alias(source, target), expected)
            self.assertTrue(source.is_symlink())
            self.assertEqual((source / 'model_alias').read_bytes(), b'uniquemodelbytes')
            self.assertEqual(inventory(target), expected)
            self.assertEqual(move_with_alias(source, target), expected)

    def test_alias_failure_rolls_back_without_losing_unique_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'work'; source.mkdir()
            (source / 'model').write_bytes(b'forever-retained')
            with patch.object(Path, 'symlink_to', side_effect=OSError('fixture')):
                with self.assertRaises(OSError):
                    move_with_alias(source, root / 'permanent')
            self.assertFalse(source.is_symlink())
            self.assertEqual((source / 'model').read_bytes(), b'forever-retained')
            self.assertFalse((root / 'permanent').exists())

    def test_wrong_existing_alias_and_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root/'a'; b = root/'b'; c = root/'c'
            a.mkdir(); b.mkdir(); c.symlink_to(a, target_is_directory=True)
            with self.assertRaises(ValueError):move_with_alias(c, b)
            with self.assertRaises(ValueError):move_with_alias(a, b)
            self.assertTrue(a.is_dir()); self.assertTrue(b.is_dir())

    def test_incomplete_or_synthetic_reports_cannot_enable_archival(self):
        report = {'status':'completed','expected_cases':20,'completed_evidence_cases':20,
                  'goal_performance_passed':True,'target_success_count':16,'success_count':16,
                  'errors':[],'actual_robot_success_claimed':False}
        require_acceptance(report)
        for key,value in [('synthetic',True),('success_count',15),('completed_evidence_cases',19),
                          ('goal_performance_passed',False),('actual_robot_success_claimed',True)]:
            with self.assertRaises(ValueError):require_acceptance({**report,key:value})

    def test_full_archive_keeps_all_original_paths_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); own = root/'workspace'; own.mkdir()
            backup = root/'ProjectBackups/model_history'; backup.mkdir(parents=True)
            (backup/'model').write_bytes(b'permanent-checkpoint')
            history = own/'training_old'; history.mkdir(); (history/'trace').write_bytes(b'rawphysics')
            for name in ('owned_runs','sim_launchers','venv','runtime_source','upstream'):(own/name).mkdir()
            (own/'gpu_budget.json').write_text(json.dumps({'runs':[]}))
            queue = root/'.scratch/vla_gpu_coordination'; queue.mkdir(parents=True)
            (queue/'queue.json').write_text(json.dumps({'tickets':[]}))
            sources = root/'experiments/lingbot_vla2_g1'; sources.mkdir(parents=True)
            (sources/'policy.py').write_text('# source fixture\n')
            report = own/'report.json'
            report.write_text(json.dumps({'status':'completed','expected_cases':20,'completed_evidence_cases':20,
                'goal_performance_passed':True,'target_success_count':16,'success_count':16,
                'errors':[],'actual_robot_success_claimed':False}))
            with patch.object(implementation,'OWN',own), patch.object(implementation,'ROOT',root), \
                 patch.object(implementation,'BACKUP',backup), \
                 patch.object(implementation.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                first = implementation.archive(report)
                second = implementation.archive(report)
            self.assertEqual(first['moved'],second['moved'])
            self.assertTrue(backup.is_symlink()); self.assertTrue(history.is_symlink())
            self.assertEqual((history/'trace').read_bytes(),b'rawphysics')
            self.assertEqual((backup/'model').read_bytes(),b'permanent-checkpoint')
            self.assertFalse((own/'venv').is_symlink())
            self.assertEqual(first['status'],'completed')
            self.assertEqual(len(first['current_task_sources_copied']),1)
