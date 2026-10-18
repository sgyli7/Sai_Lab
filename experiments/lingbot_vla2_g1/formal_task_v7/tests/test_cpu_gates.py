"""Synthetic CPU gates for the R4 formal entry. No robot success is claimed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1.formal_task_v7 import aggregate as aggregate_module
from experiments.lingbot_vla2_g1.formal_task_v7 import cpu_manifest, evaluate, make_plan as make_plan_module
from experiments.lingbot_vla2_g1.formal_task_v7.gates import (
    REQUIRED_CHECKS, TARGET_SUCCESS, admit_development, admit_storage, apply_target,
    assert_case_protocol, assert_common_cohort, bind_candidate_role, recheck_turns,
    scenario_document, storage_admission)


def development_fixture(root, *, delta='/tmp/synthetic-r4/final', step=64, best=0, service=None,
                        selected_by_validation=None, audit_passed=True, fail_check=False):
    delta = str(Path(delta).absolute())
    service = service or 'experiments.lingbot_vla2_g1.service_pointmass'
    if selected_by_validation is None:
        selected_by_validation = step == best
    checks = {name: {'passed': True} for name in REQUIRED_CHECKS}
    checks['videos']['streams'] = {'head_video_path': {'synthetic': True}, 'observer_video_path': {'synthetic': True}}
    if fail_check:
        checks['acceptance']['passed'] = False
    plan_jobs, done = [], []
    for index, case_id, outcome, success in (
        (0, 'synthetic_screen', {'grasped': True, 'fell': False, 'dropped': False, 'success': False}, False),
        (1, 'synthetic_full', {'grasped': True, 'fell': False, 'dropped': False, 'success': True}, True),
    ):
        folder = root / case_id
        folder.mkdir()
        result_path = folder / 'result.json'
        result_path.write_text(json.dumps({'schema': 'pi05_g1_mobile_case_result_v1', 'status': 'completed',
                                           'success': success, 'acceptance': outcome, 'synthetic': True,
                                           'actual_robot_success_claimed': False}))
        (folder / 'raw.txt').write_text('synthetic')
        config = {'synthetic': True, 'mode': 'policy', 'policy_id': 'lingbot_vla2_selected',
                  'output_dir': str(folder), 'norm_stats': str(OWN / 'train_meanstd.json'),
                  'policy_command': [str(OWN / 'venv/bin/python'), '-m', service, '--delta', delta,
                                     '--norm-stats', str(OWN / 'train_meanstd.json'),
                                     '--decoded-evidence', str(folder / 'decoded.jsonl')],
                  'HFmodel': {'delta': delta, 'selected_by_validation': selected_by_validation}}
        config_path = root / (case_id + '.json')
        config_path.write_text(json.dumps(config))
        audit = {'synthetic': True, 'passed': audit_passed, 'autonomous_policy_trial': True,
                 'anomalies': [], 'case_id': case_id, 'result_path': str(result_path),
                 'checks': checks, 'recomputed_acceptance': outcome}
        audit_path = root / (case_id + '_audit.json')
        audit_path.write_text(json.dumps(audit))
        plan_jobs.append({'index': index, 'case_id': case_id, 'config_path': str(config_path),
                          'config_sha256': sha(config_path), 'result_file': str(result_path)})
        done.append({'index': index, 'case_id': case_id, 'status': 'completed', 'result_file': str(result_path),
                     'result_sha256': sha(result_path), 'audit': str(audit_path), 'audit_sha256': sha(audit_path),
                     'physical_outcome': outcome,
                     'raw_files_sha256': {str(path): sha(path) for path in sorted(folder.rglob('*')) if path.is_file()}})
    plan = {'schema': 'lingbot_task_v4_development_plan_v1', 'formal20_benchmark': False, 'synthetic': True,
            'selected_delta': delta, 'selected_new_step': step, 'jobs': plan_jobs}
    plan_path = root / 'development_plan.json'
    plan_path.write_text(json.dumps(plan))
    result = {'schema': 'lingbot_task_v4_development_execution_v1', 'status': 'completed', 'synthetic': True,
              'actual_robot_success_claimed': False, 'plan_sha256': sha(plan_path), 'jobs': done,
              'grasp_screen_passed': True, 'full_task_case_ran': True, 'full_task_case_success': True,
              'overall_robot_task_goal_completed': False, 'formal20_benchmark': False}
    result_path = root / 'development_result.json'
    result_path.write_text(json.dumps(result))
    return plan_path, result_path, delta


def rewrite_result(path, result):
    path.write_text(json.dumps(result))


class DevelopmentRejectionTests(unittest.TestCase):

    def test_R8_and_R7_share_R6_parent_but_require_distinct_actual_job_roles(self):
        provenance = {'source_validation_best_step': 0, 'physical_candidate_step': 180,
                      'physical_candidate_is_source_validation_best': False, 'initial_lineage_updates': 2040,
                      'job': str(OWN/'training_task_round08_job.json')}
        self.assertEqual(bind_candidate_role(provenance, 'actual_final_task_round08', False)['candidate_role'],
                         'actual_final_task_round08')
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(provenance, 'actual_final_task_round07', False)
        provenance['job'] = str(OWN/'training_task_round07_job.json')
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(provenance, 'actual_final_task_round08', False)

    def test_R8_over200_update_claim_is_rejected(self):
        provenance = {'source_validation_best_step': 0, 'physical_candidate_step': 201,
                      'physical_candidate_is_source_validation_best': False, 'initial_lineage_updates': 2040,
                      'job': str(OWN/'training_task_round08_job.json')}
        with self.assertRaisesRegex(ValueError, 'finite200'):
            bind_candidate_role(provenance, 'actual_final_task_round08', False)

    def test_coherent_adapter_must_match_plan_and_actual_policy_arguments(self):
        from experiments.lingbot_vla2_g1.coherent_sampling import ADAPTER
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, result_path, delta = development_fixture(root)
            plan = json.loads(plan_path.read_text())
            result = json.loads(result_path.read_text())
            plan['sampling_adapter'] = ADAPTER
            for job in plan['jobs']:
                config_path = Path(job['config_path'])
                config = json.loads(config_path.read_text())
                config['policy_command'][2] = 'experiments.lingbot_vla2_g1.service_coherent_noise'
                config['HFmodel']['sampling_adapter'] = ADAPTER
                config_path.write_text(json.dumps(config))
                job['config_sha256'] = sha(config_path)
            plan_path.write_text(json.dumps(plan))
            result['plan_sha256'] = sha(plan_path)
            result_path.write_text(json.dumps(result))
            admitted = admit_development(plan_path, result_path, delta, 64, 0)
            self.assertEqual(admitted['plan']['sampling_adapter'], ADAPTER)
            plan['sampling_adapter'] = None
            plan_path.write_text(json.dumps(plan))
            result['plan_sha256'] = sha(plan_path)
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'sampling adapter differs'):
                admit_development(plan_path, result_path, delta, 64, 0)

    def test_coherent_service_without_declared_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, result, delta = development_fixture(Path(directory),
                service='experiments.lingbot_vla2_g1.service_coherent_noise')
            with self.assertRaisesRegex(ValueError, 'declared LingBot sampling'):
                admit_development(plan, result, delta, 64, 0)

    def test_source_early_drop_cannot_promote_even_when_ground_drop_flag_is_false(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result_path, delta = development_fixture(root)
            result = json.loads(result_path.read_text())
            record = result['jobs'][0]
            case_path = Path(record['result_file'])
            case = json.loads(case_path.read_text())
            case.update(physical_failure=True, termination_reason='source_object_dropped_predicate')
            self.assertFalse(case['acceptance']['dropped'])
            case_path.write_text(json.dumps(case))
            record['result_sha256'] = sha(case_path)
            record['raw_files_sha256'][str(case_path)] = sha(case_path)
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Source fall/drop'):
                admit_development(plan, result_path, delta, 64, 0)

    def test_synthetic_success_cannot_generate_an_actual_formal_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, _ = development_fixture(root)
            with patch('experiments.lingbot_vla2_g1.round04_selection.admit', side_effect=AssertionError('training')):
                with self.assertRaisesRegex(ValueError, 'Synthetic fixtures'):
                    make_plan_module.make_plan(root / 'missing-training.json', plan, result, root / 'formal', root / 'episodes')
            self.assertFalse((root / 'formal').exists())

    def test_privileged_teacher_is_rejected_before_training_admission_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, _ = development_fixture(root, service='experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v2')
            with patch('experiments.lingbot_vla2_g1.round04_selection.admit', side_effect=AssertionError('training')), \
                    patch('experiments.vla_resources.gpu_queue.request', side_effect=AssertionError('gpu')):
                with self.assertRaisesRegex(ValueError, 'Privileged or scripted teacher'):
                    make_plan_module.make_plan(root / 'missing-training.json', plan, result, root / 'formal', root / 'episodes')
            self.assertFalse((root / 'formal').exists())

    def test_skipped_or_incomplete_full_development_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result_path, delta = development_fixture(root)
            result = json.loads(result_path.read_text())
            result['full_task_case_ran'] = False
            rewrite_result(result_path, result)
            with self.assertRaisesRegex(ValueError, 'incomplete full development'):
                admit_development(plan, result_path, delta, 64, 0)
            result['full_task_case_ran'] = True
            result['jobs'][1]['status'] = 'not_run_grasp_screen_failed'
            rewrite_result(result_path, result)
            with self.assertRaisesRegex(ValueError, 'Skipped or incomplete'):
                admit_development(plan, result_path, delta, 64, 0)

    def test_failed_independent_audit_and_outcome_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result_path, delta = development_fixture(root, audit_passed=False)
            with self.assertRaisesRegex(ValueError, 'Failed independent audit'):
                admit_development(plan, result_path, delta, 64, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result_path, delta = development_fixture(root, fail_check=True)
            with self.assertRaisesRegex(ValueError, 'Failed independent audit'):
                admit_development(plan, result_path, delta, 64, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result_path, delta = development_fixture(root)
            result = json.loads(result_path.read_text())
            result['jobs'][1]['physical_outcome'] = {**result['jobs'][1]['physical_outcome'], 'success': False}
            rewrite_result(result_path, result)
            with self.assertRaisesRegex(ValueError, 'Physical outcome differs'):
                admit_development(plan, result_path, delta, 64, 0)

    def test_checkpoint_mismatch_and_role_relabel_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, delta = development_fixture(root)
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                admit_development(plan, result, root / 'other-final', 64, 0)
            self.assertEqual(admit_development(plan, result, delta, 64, 0)['selected_delta'], delta)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, delta = development_fixture(root, selected_by_validation=True)
            with self.assertRaisesRegex(ValueError, 'relabelled'):
                admit_development(plan, result, delta, 64, 0)

    def test_consistent_synthetic_development_does_not_request_gpu(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('experiments.vla_resources.gpu_queue.request', side_effect=AssertionError('gpu')) as request:
            plan, result, delta = development_fixture(Path(directory))
            admitted = admit_development(plan, result, delta, 64, 0)
            self.assertEqual(admitted['selected_new_step'], 64)
            self.assertTrue(json.loads(result.read_text())['synthetic'])
            request.assert_not_called()


class RoleAndCohortTests(unittest.TestCase):
    def test_R7_role_requires_its_actual_2040_update_initial_lineage(self):
        provenance = {'source_validation_best_step': 0, 'physical_candidate_step': 240,
                      'physical_candidate_is_source_validation_best': False, 'initial_lineage_updates': 2040}
        role = bind_candidate_role(provenance, 'actual_final_task_round07', False)
        self.assertEqual(role['candidate_role'], 'actual_final_task_round07')
        for incorrect in ('actual_final_task_round04', 'actual_final_task_round05', 'actual_final_task_round06'):
            with self.assertRaisesRegex(ValueError, 'relabelled'):
                bind_candidate_role(provenance, incorrect, False)
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role({**provenance, 'initial_lineage_updates': 1800}, 'actual_final_task_round07', False)

    def test_R6_role_requires_its_actual_recovery_training_lineage(self):
        provenance = {'source_validation_best_step': 0, 'physical_candidate_step': 240,
                      'physical_candidate_is_source_validation_best': False, 'initial_lineage_updates': 1800}
        role = bind_candidate_role(provenance, 'actual_final_task_round06', False)
        self.assertEqual(role['candidate_role'], 'actual_final_task_round06')
        for incorrect in ('actual_final_task_round04', 'actual_final_task_round05'):
            with self.assertRaisesRegex(ValueError, 'relabelled'):
                bind_candidate_role(provenance, incorrect, False)
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role({**provenance, 'initial_lineage_updates': 1400}, 'actual_final_task_round06', False)

    def test_R5_role_requires_correct_actual_weight_lineage(self):
        provenance = {'source_validation_best_step': 0, 'physical_candidate_step': 400,
                      'physical_candidate_is_source_validation_best': False, 'initial_lineage_updates': 1400}
        role = bind_candidate_role(provenance, 'actual_final_task_round05', False)
        self.assertEqual(role['candidate_role'], 'actual_final_task_round05')
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(provenance, 'actual_final_task_round04', False)
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role({**provenance, 'initial_lineage_updates': 1000}, 'actual_final_task_round05', False)

    def test_final_role_cannot_be_relabelled(self):
        separate = {'source_validation_best_step': 0, 'physical_candidate_step': 64,
                    'physical_candidate_is_source_validation_best': False}
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(separate, 'source_validation_best', False)
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(separate, 'actual_final_task_round04', True)
        self.assertFalse(bind_candidate_role(separate, 'actual_final_task_round04', False)['selected_by_validation'])
        same = {'source_validation_best_step': 64, 'physical_candidate_step': 64,
                'physical_candidate_is_source_validation_best': True}
        self.assertTrue(bind_candidate_role(same, 'actual_final_task_round04', True)['selected_by_validation'])
        with self.assertRaisesRegex(ValueError, 'relabelled'):
            bind_candidate_role(same, 'actual_final_task_round04', False)

    def test_scene_order_cadence_and_threshold_changes_are_rejected(self):
        original, _ = scenario_document()
        cases = original['cases']
        assert_common_cohort(cases, original)
        reordered = copy.deepcopy(cases)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, 'order/scene'):
            assert_common_cohort(reordered, original)
        moved = copy.deepcopy(cases)
        moved[0]['scene']['box_pose']['position_xyz'][0] += 1
        with self.assertRaisesRegex(ValueError, 'order/scene'):
            assert_common_cohort(moved, original)
        cadence = copy.deepcopy(original)
        cadence['control_hz'] = 10
        with self.assertRaisesRegex(ValueError, 'cadence'):
            assert_common_cohort(cases, cadence)
        thresholds = copy.deepcopy(original)
        self._bump_number(thresholds['acceptance'])
        with self.assertRaisesRegex(ValueError, 'thresholds'):
            assert_common_cohort(cases, thresholds)
        config = {field: original[field] for field in (
            'ticks', 'episode_seconds', 'native_action_horizon', 'replan_interval_ticks',
            'observer_camera', 'acceptance', 'action_limits_file')}
        assert_case_protocol(config, original)
        config['replan_interval_ticks'] = 25
        with self.assertRaisesRegex(ValueError, 'cadence'):
            assert_case_protocol(config, original)

    def _bump_number(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if type(item) in (int, float):
                    value[key] = item + 1
                    return True
                if self._bump_number(item):
                    return True
        if isinstance(value, list):
            for index, item in enumerate(value):
                if type(item) in (int, float):
                    value[index] = item + 1
                    return True
                if self._bump_number(item):
                    return True
        return False


def completed_record(folder, index, case_id, *, success=True, status='completed', audit_passed=True,
                     policy='lingbot_vla2_selected', partial_carry_only=False, full_task_success=None):
    folder.mkdir(parents=True)
    result = folder / 'result.json'
    result.write_text('{}')
    audit = folder.parent / ('audit_' + str(index) + '.json')
    audit.write_text('{}')
    record = {'status': status, 'index': index, 'policy': policy, 'case_id': case_id, 'success': success,
              'synthetic': True, 'actual_robot_success_claimed': False, 'audit_passed': audit_passed,
              'partial_carry_only': partial_carry_only, 'case_files_sha256': evaluate.raw_files(folder),
              'cpu_audit_path': str(audit), 'cpu_audit_sha256': sha(audit)}
    if full_task_success is not None:
        record['full_task_success'] = full_task_success
    return record


class ResumeAndBudgetTests(unittest.TestCase):
    def test_contiguous_prefix_is_accepted_and_gaps_conflicts_and_foreign_plans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            records = []
            for index in range(2):
                case_id = 'synthetic_' + str(index)
                folder = root / case_id
                records.append(completed_record(folder, index, case_id, success=False))
                jobs.append({'policy': 'lingbot_vla2_selected', 'case_id': case_id, 'result_path': str(folder / 'result.json')})
            jobs.extend({'policy': 'lingbot_vla2_selected', 'case_id': 'later_' + str(index), 'result_path': str(root / 'later')} for index in range(2, 20))
            document = {'schema': 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1', 'plan_sha256': 'plan',
                        'synthetic': True, 'model_payloads_verified_before': True, 'model_payloads_verified_end': True,
                        'jobs': records}
            receipt = root / 'prefix.json'
            receipt.write_text(json.dumps(document))
            plan = {'jobs': jobs}
            admitted = evaluate.previous_records([receipt], 'plan', plan)
            self.assertEqual(set(admitted), {0, 1})
            gapped = json.loads(json.dumps(document))
            gapped['jobs'] = [records[1]]
            gap_path = root / 'gap.json'
            gap_path.write_text(json.dumps(gapped))
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                evaluate.previous_records([gap_path], 'plan', plan)
            other = json.loads(json.dumps(records[0]))
            other['success'] = True
            conflict = root / 'conflict.json'
            conflict.write_text(json.dumps({**document, 'jobs': [other]}))
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                evaluate.previous_records([receipt, conflict], 'plan', plan)
            with self.assertRaisesRegex(ValueError, 'differs from frozen plan'):
                evaluate.previous_records([receipt], 'foreign', plan)

    def test_rejected_resume_storage_and_short_budget_do_not_request_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / 'receipt.json'
            foreign = root / 'foreign.json'
            foreign.write_text(json.dumps({'schema': 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1',
                                           'plan_sha256': 'foreign', 'synthetic': True,
                                           'model_payloads_verified_before': True, 'model_payloads_verified_end': True,
                                           'jobs': []}))
            plan_path = root / 'plan.json'
            plan_path.write_text('{}')
            loaded = {'jobs': [{'max_seconds': 900, 'policy': 'p', 'case_id': 'c'} for _ in range(3)],
                      'storage_admission': storage_admission()}
            with patch.object(evaluate, 'OWN', root), patch.object(evaluate, 'load_formal_plan', return_value=loaded), \
                    patch.object(evaluate, 'admit_allocation', side_effect=AssertionError('budget')), \
                    patch.object(evaluate.gpu_queue, 'request', side_effect=AssertionError('gpu')) as request:
                with self.assertRaisesRegex(ValueError, 'differs from frozen plan'):
                    evaluate.run(plan_path, receipt, 1, [foreign])
                request.assert_not_called()
            with patch.object(evaluate, 'OWN', root), \
                    patch.object(evaluate, 'load_formal_plan', return_value={'storage_admission': {'schema': 'lingbot_storage_admission_v2', 'own_cap_bytes': 36 * 1024 ** 3}}), \
                    patch.object(evaluate.gpu_queue, 'request', side_effect=AssertionError('gpu')) as request:
                with self.assertRaisesRegex(ValueError, '64GiB'):
                    evaluate.run(plan_path, receipt, 1, [])
                request.assert_not_called()
            ledger = root / 'ledger.json'
            ledger.write_text(json.dumps({'runs': [{'stage': 'evaluation', 'elapsed_seconds': 0}]}))
            allocation = {'old_wall_deadline_enforced': False, 'total_limit_seconds': 1000, 'used_total_seconds': 500,
                          'stage_limits_seconds': {'evaluation': 1000}}
            with patch.object(evaluate, 'OWN', root), patch.object(evaluate, 'LEDGER', ledger), \
                    patch.object(evaluate, 'load_formal_plan', return_value=loaded), \
                    patch.object(evaluate, 'admit_allocation', return_value=allocation), \
                    patch.object(evaluate.gpu_queue, 'request', side_effect=AssertionError('gpu')) as request:
                with self.assertRaisesRegex(ValueError, 'full next turn'):
                    evaluate.run(plan_path, receipt, 3, [])
                request.assert_not_called()

    def test_continuation_budget_and_combined_storage_replace_obsolete_limits(self):
        source = Path(evaluate.__file__).read_text()
        self.assertIn('budget_reallocated_v7', source)
        self.assertNotIn('budget_12h', source)
        self.assertNotIn('budget_allocated', source)
        self.assertNotIn('wall_remaining', source)
        command = evaluate.supervisor_argv(Path('/tmp/synthetic/result.json'), Path('/tmp/synthetic/config.json'))
        self.assertIn('experiments.lingbot_vla2_g1.budget_reallocated_v7', command)
        self.assertNotIn('budget_12h', command)
        self.assertNotIn('budget_allocated', command)
        self.assertEqual(command[command.index('--max-seconds') + 1], '900')
        self.assertEqual(command[command.index('--memory-high-gib') + 1], '80')
        self.assertEqual(command[command.index('--memory-max-gib') + 1], '90')
        self.assertEqual(command[command.index('--host-reserve-gib') + 1], '16')
        self.assertEqual(evaluate.reservation({'max_seconds': 900})[1:], (evaluate.OWNER, 'evaluation', 900))
        self.assertEqual(evaluate.OWNER, '01a145c6-0b86-7623-a869-086b33292199')
        admit_storage(storage_admission())
        with self.assertRaisesRegex(ValueError, '64GiB|Obsolete'):
            admit_storage({'schema': 'lingbot_storage_admission_v2', 'own_cap_bytes': 25 * 1024 ** 3, 'combined_cap_bytes': 48 * 1024 ** 3})
        with self.assertRaisesRegex(ValueError, 'expired wall'):
            admit_storage({**storage_admission(), 'wall_deadline_enforced': True})
        with self.assertRaisesRegex(ValueError, 'Expired wall'):
            recheck_turns({'old_wall_deadline_enforced': True}, {'runs': []}, 1)
        with self.assertRaisesRegex(ValueError, 'seven'):
            evaluate.verify_payloads({'immutable_payloads': {}})
        with self.assertRaisesRegex(ValueError, 'one artifact root'):
            cpu_manifest.publish(Path('/tmp/not-the-cpu-root'))


def performance_report(success_count, *, status='completed', cases=20, errors=None, forged=False):
    return {'status': status, 'formal_verdict_available': status == 'completed' and cases == 20 and not errors,
            'completed_evidence_cases': cases, 'expected_cases': 20, 'success_count': success_count,
            'errors': [] if errors is None else errors, 'synthetic': True, 'goal_performance_passed': forged,
            'actual_robot_success_claimed': False}


def twenty(root, *, successes=16, status='completed', audit_passed=True, policy='lingbot_vla2_selected',
           partial_carry_only=False, full_task_success=None):
    jobs = []
    for index in range(20):
        success = index < successes
        jobs.append(completed_record(root / ('case_' + str(index)), index, 'synthetic_' + str(index),
                                     success=success, status=status if success or status == 'completed' else status,
                                     audit_passed=audit_passed, policy=policy, partial_carry_only=partial_carry_only,
                                     full_task_success=full_task_success))
    if status != 'completed':
        for job in jobs:
            job['status'] = status
    return jobs


class AggregateTargetTests(unittest.TestCase):
    def test_real_plan_loader_remains_callable_inside_aggregate_adapter(self):
        import experiments.lingbot_vla2_g1.benchmark_pose_speed as benchmark
        from experiments.lingbot_vla2_g1.formal_task_v7.gates import load_formal_plan
        original = benchmark.load_plan
        with tempfile.TemporaryDirectory() as directory:
            benchmark.load_plan = load_formal_plan
            try:
                with self.assertRaises(FileNotFoundError):
                    load_formal_plan(Path(directory) / 'missing_plan.json')
            finally:
                benchmark.load_plan = original

    def test_adapter_is_restored_on_success_and_exception(self):
        import experiments.lingbot_vla2_g1.benchmark_pose_speed as benchmark
        original = (benchmark.load_plan, benchmark.decoded_evidence)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, receipt = root / 'plan.json', root / 'receipt.json'
            plan.write_text('{}')
            receipt.write_text(json.dumps({'synthetic': True, 'jobs': twenty(root / 'rows', successes=16)}))
            output = root / 'summary.json'

            def boom(*args, **kwargs):
                self.assertIs(benchmark.load_plan, aggregate_module.load_formal_plan)
                self.assertIs(benchmark.decoded_evidence, aggregate_module.pointmass_decoder)
                raise RuntimeError('synthetic boundary')

            with patch.object(benchmark, 'aggregate', side_effect=boom):
                with self.assertRaisesRegex(RuntimeError, 'synthetic boundary'):
                    aggregate_module.aggregate(plan, [receipt], output)
            self.assertEqual((benchmark.load_plan, benchmark.decoded_evidence), original)
            self.assertFalse(output.exists())

            def succeed(plan_path, receipt_paths, output_path):
                self.assertIs(benchmark.load_plan, aggregate_module.load_formal_plan)
                self.assertIs(benchmark.decoded_evidence, aggregate_module.pointmass_decoder)
                return performance_report(16)

            with patch.object(benchmark, 'aggregate', side_effect=succeed):
                report = aggregate_module.aggregate(plan, [receipt], output)
            self.assertEqual((benchmark.load_plan, benchmark.decoded_evidence), original)
            self.assertTrue(report['goal_performance_passed'])
            self.assertEqual(report['target_success_count'], TARGET_SUCCESS)
            self.assertTrue(report['synthetic'])
            self.assertFalse(report['actual_robot_success_claimed'])
            self.assertEqual(report['acceptance_conclusion_owner'], 'parent')

    def test_incomplete_audit_infrastructure_teacher_carry_and_fifteen_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifteen = [{'synthetic': True, 'jobs': twenty(root / 'fifteen', successes=15)}]
            report = apply_target(performance_report(15, forged=True), fifteen)
            self.assertFalse(report['goal_performance_passed'])
            forged = apply_target(performance_report(16, forged=True), fifteen)
            self.assertFalse(forged['goal_performance_passed'])
            incomplete = apply_target(performance_report(None, status='incomplete', cases=10),
                                      [{'synthetic': True, 'jobs': twenty(root / 'partial', successes=10)[:10]}])
            self.assertFalse(incomplete['goal_performance_passed'])
            failed = apply_target(performance_report(16), [{'synthetic': True, 'jobs': twenty(root / 'audit', audit_passed=False)}])
            self.assertFalse(failed['goal_performance_passed'])
            infrastructure = apply_target(performance_report(16), [{'synthetic': True, 'jobs': twenty(root / 'infra', status='infrastructure_failed')}])
            self.assertFalse(infrastructure['goal_performance_passed'])
            teacher = apply_target(performance_report(20), [{'synthetic': True, 'teacher_data': True,
                                                             'jobs': twenty(root / 'teacher', successes=20, policy='privileged_scripted_TRAIN_teacher')}])
            self.assertFalse(teacher['goal_performance_passed'])
            carry = apply_target(performance_report(16), [{'synthetic': True, 'jobs': twenty(
                root / 'carry', successes=16, partial_carry_only=True, full_task_success=False)}])
            self.assertFalse(carry['goal_performance_passed'])
            valid = apply_target(performance_report(16), [{'synthetic': True, 'jobs': twenty(root / 'valid', successes=16)}])
            self.assertTrue(valid['goal_performance_passed'])
            self.assertFalse(valid['actual_robot_success_claimed'])


if __name__ == '__main__':
    unittest.main()
