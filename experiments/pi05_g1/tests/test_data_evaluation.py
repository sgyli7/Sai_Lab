import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
prep = load('prepare_training_data')
eval = load('summarize_evaluation')
ACCEPTANCE_HASH = eval.current_acceptance_identity()['acceptance_sha256']

class DataEvaluationTests(unittest.TestCase):
    def test_duplicate_trajectory_and_video_groups_do_not_leak(self):
        records = [{'episode_id': i, 'content_sha256': str(i), 'video_sha256': {'rgb': str(i)}} for i in range(30)]
        records[1]['content_sha256'] = '0'
        records[2]['video_sha256']['rgb'] = '1'
        split = prep.grouped_split(records)
        self.assertEqual(split, prep.grouped_split(records))
        self.assertEqual(sorted(sum(split.values(), [])), list(range(30)))
        owner = {i: key for key, ids in split.items() for i in ids}
        self.assertEqual(owner[0], owner[1])
        self.assertEqual(owner[1], owner[2])
        self.assertTrue(all(split.values()))

    def test_normalization_uses_only_train_and_rejects_nan(self):
        episodes = {0: {k: np.ones((3, 32)) for k in ('state', 'actions')},
                    1: {k: np.full((2,32), 999.) for k in ('state', 'actions')}}
        self.assertEqual(prep.fit_norm(episodes, [0])['actions']['mean'], [1.]*32)
        episodes[0]['state'][0,0] = np.nan
        with self.assertRaises(ValueError):
            prep.fit_norm(episodes, [0])

    def test_shifted_timestamps_cannot_silently_pair_with_future_rgb(self):
        times = np.arange(100) / 50.
        prep.check_video_alignment(times, 100, 50.)
        for bad_times, count, fps in ((times + .1, 100, 50.), (times, 99, 50.), (times, 100, 30.)):
            with self.assertRaises(ValueError):
                prep.check_video_alignment(bad_times, count, fps)

    def test_cluster_interval_resamples_positions_not_noise_repeats(self):
        positions = [[i,0] for i in range(5) for _ in range(4)]
        baseline = [0]*20
        tuned = [1]*4 + [0]*16
        result = eval.clustered_statistics(baseline, tuned, positions)
        self.assertEqual(result['position_clusters'], 5)
        self.assertEqual(result['paired_trials'], 20)
        self.assertEqual(result['position_cluster_bootstrap_95ci'][0], 0.)
        self.assertGreaterEqual(result['position_cluster_bootstrap_95ci'][1], .6)

    def test_known_statistics(self):
        self.assertAlmostEqual(eval.wilson(0, 20)[1], .161125158, places=6)
        statistics = eval.paired_statistics([False]*20, [True]*20)
        self.assertEqual(statistics['paired_bootstrap_95ci'], [1., 1.])
        self.assertAlmostEqual(statistics['exact_mcnemar_p'], 2/2**20)
        with self.assertRaises(ValueError):
            eval.paired_statistics([np.nan], [1])

    def test_missing_runs_are_incomplete_not_zero_of_twenty(self):
        cases = [{'case_id': str(i)} for i in range(20)]
        result = eval.summarize(cases, [{'case_id':'0', 'policy':'baseline', 'success':None, 'status':'blocked'}])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['policies']['baseline']['success_rate_observed'])
        self.assertFalse(result['task_usable'])
        self.assertEqual(result['policies']['finetuned']['completed'], 0)

    def test_false_success_flags_are_rejected(self):
        cases = [{'case_id': str(i)} for i in range(20)]
        row = dict(case_id='0', policy='finetuned', success=True, grasped=False,
                   stable_placement=True, dropped=False, fell=False, timeout=False)
        with self.assertRaises(ValueError):
            eval.summarize(cases, [row])

    def test_frozen_case_mismatch_is_not_complete(self):
        cases = [{'case_id':str(i), 'seed': i, 'apple_xy_offset_m':[0,0]} for i in range(20)]
        row = dict(case_id='0', policy='finetuned', success=False, grasped=False,
                   stable_placement=False, dropped=False, fell=False, timeout=True,
                   case={'seed':99, 'apple_xy_offset_m':[0,0]})
        report = eval.summarize(cases, [row])
        self.assertIn('frozen seed mismatch: finetuned/0', report['errors'])
        self.assertFalse(report['complete'])

    def test_candidate_failure_is_not_task_completion_time(self):
        failed = {'success':False, 'control_hz':50, 'source_success_candidate_tick':293,
                  'completed_control_ticks':400, 'first_grasp_control_tick':None,
                  'inference_calls':[{'observation_control_tick':0,'inference_seconds':2.,'simulation_paused_wall_seconds':2.1},
                                     {'observation_control_tick':50,'inference_seconds':.5,'simulation_paused_wall_seconds':.6}]}
        report = eval.timing_summary({'failed':failed})
        self.assertEqual(report['source_candidate_sim_seconds']['mean'], 5.86)
        self.assertIsNone(report['task_completion_sim_seconds_success_only'])
        self.assertIsNone(report['time_to_first_grasp_sim_seconds'])
        self.assertEqual(report['episode_sim_seconds']['mean'], 8.)
        self.assertEqual(report['inference_seconds_cold_tick0']['mean'], 2.)
        self.assertEqual(report['inference_seconds_warm_ticks']['mean'], .5)
        self.assertEqual(report['simulation_pause_wall_seconds']['count'], 2)
        successful = {**failed, 'success':True, 'source_success_candidate_tick':150,
                      'first_grasp_control_tick':20, 'qualified_stability_window':{'start_tick':150,'end_tick':250,'passed':True},
                      'launch_wall_seconds':123.4}
        report = eval.timing_summary({'failed':failed, 'successful':successful})
        self.assertEqual(report['task_completion_sim_seconds_success_only']['count'], 1)
        self.assertEqual(report['task_completion_sim_seconds_success_only']['mean'], 5.)
        self.assertEqual(report['time_to_first_grasp_sim_seconds']['mean'], .4)
        self.assertEqual(report['launch_wall_seconds']['mean'], 123.4)
        bad = {**successful, 'qualified_stability_window':None}
        with self.assertRaises(ValueError):
            eval.timing_summary({'bad':bad})

    def test_recursive_results_match_only_exact_launch_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root/'p0_seed0'
            trial.mkdir()
            configs = root/'configs'
            configs.mkdir()
            result = trial/'result.json'
            result.write_text(json.dumps({'case_id':'p0_seed0','policy':'baseline','video_path':'head.mp4'}))
            (configs/'unrelated.json').write_text('not result JSON and must never be parsed')
            receipt = configs/'p0_seed0.launch.json'
            receipt.write_text(json.dumps({'case_id':'p0_seed0','policy':'baseline','elapsed_seconds':127.5}))
            rows = eval.load_result_files([root, result])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['launch_wall_seconds'], 127.5)
            self.assertEqual(rows[0]['launch_wall_receipt'], str(receipt))
            self.assertEqual(rows[0]['video_path'], str(trial/'head.mp4'))
            receipt.write_text(json.dumps({'case_id':'p0_seed0','policy':'finetuned','elapsed_seconds':127.5}))
            with self.assertRaises(ValueError):
                eval.load_result_files([root])
            receipt.unlink()
            self.assertNotIn('launch_wall_seconds', eval.load_result_files([result])[0])

    def test_gates_and_video_evidence_required(self):
        cases = [{'case_id': str(i)} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'evidence.mp4'
            video.write_bytes(b'evidence existence fixture')
            rows = [dict(case_id=str(i), policy=p, success=p=='finetuned', grasped=True,
                         stable_placement=p=='finetuned', dropped=False, fell=False, timeout=False,
                         video_path=str(video), source_interface_passed=True, replay_passed=True,
                         acceptance_version=eval.REQUIRED_ACCEPTANCE_VERSION, acceptance_sha256=ACCEPTANCE_HASH,
                         frozen_interface_sha256='f'*64, initial_physical_state_sha256='1'*64,
                         qualified_stability_window={'start_tick':100, 'end_tick':200, 'passed':True}) for p in ('baseline','finetuned') for i in range(20)]
            gates = {'source_interface_passed':True, 'replay_passed':True,
                     'acceptance_version':eval.REQUIRED_ACCEPTANCE_VERSION, 'acceptance_sha256':ACCEPTANCE_HASH,
                     'frozen_interface_sha256':'f'*64}
            report = eval.summarize(cases, rows, gates)
            self.assertTrue(report['training_benefit_proven'])
            self.assertTrue(report['task_usable'])
            self.assertEqual(report['eligible_physical_pair_count'], 20)
            mismatched_initial = [{**row} for row in rows]
            mismatched_initial[-1]['initial_physical_state_sha256'] = '2'*64
            mismatch_report = eval.summarize(cases, mismatched_initial, gates)
            self.assertEqual(mismatch_report['eligible_physical_pair_count'], 19)
            self.assertFalse(mismatch_report['task_usable'])
            self.assertFalse(mismatch_report['evidence_complete'])
            self.assertTrue(any('initial physical state hashes differ' in error for error in mismatch_report['errors']))
            missing_initial = [{**row} for row in rows]
            del missing_initial[-1]['initial_physical_state_sha256']
            missing_report = eval.summarize(cases, missing_initial, gates)
            self.assertEqual(missing_report['eligible_physical_pair_count'], 19)
            self.assertFalse(missing_report['evidence_complete'])
            self.assertFalse(missing_report['training_benefit_proven'])
            self.assertEqual(missing_report['pair_eligibility'][-1]['status'], 'missing_or_invalid_initial_state_hash')
            self.assertFalse(eval.summarize(cases, rows, {**gates, 'replay_passed':False})['task_usable'])
            mixed = [{**row} for row in rows]
            mixed[-1]['acceptance_sha256'] = 'b'*64
            self.assertFalse(eval.summarize(cases, mixed, gates)['task_usable'])
            missing = [{key:value for key,value in row.items() if key != 'acceptance_version'} for row in rows]
            self.assertFalse(eval.summarize(cases, missing, gates)['task_usable'])
            old_gates = {**gates, 'acceptance_version':'old_v1'}
            self.assertFalse(eval.summarize(cases, rows, old_gates)['task_usable'])
            video.unlink()
            self.assertFalse(eval.summarize(cases, rows, gates)['evidence_complete'])
            rows[0]['latency'] = float('nan')
            with self.assertRaises(ValueError):
                eval.summarize(cases, rows)

if __name__ == '__main__':
    unittest.main()
