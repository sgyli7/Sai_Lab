"""Freeze a finite, explicitly TRAIN-only recovery collection in shared FIFO."""
import copy
import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.vla_resources import gpu_queue
from experiments.lingbot_vla2_g1.budget_reallocated_v7 import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, SHARED, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write
from experiments.pi05_g1_mobile.simulator import validate_config
from .privileged_teacher_v6 import PREFIX_CONTROLS, validate_source


def main():
    destination = OWN / 'placement_recovery_teacher06'
    if destination.exists():
        raise FileExistsError('Preserve every previous preparation and trial')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used = sum(r['elapsed_seconds'] for r in ledger['runs'] if r['stage'] == 'evaluation')
    if (allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 900
            or allocation['stage_limits_seconds']['evaluation'] - used < 900):
        raise ValueError('Finite900seconds must fit the current total/stage allocation')
    gate_path = OWN / 'diagnostics/recovery_teacher_v6_CPU_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 7:
        raise ValueError('Actual-prefix handoff and bilateral correction checks required')
    for name, expected in gate['source_files_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Collector changed after CPU verification: ' + name)
    prior_audit = OWN / 'placement_recovery_teacher05/independent_CPU_audit.json'
    prior_result = OWN / 'placement_recovery_teacher05/episode/result.json'
    prior = json.loads(prior_audit.read_text())
    if (prior.get('passed') is not True or prior.get('counts_as_LingBot_success') is not False
            or prior.get('recomputed_acceptance', {}).get('success') is not False
            or any(check.get('passed') is not True for check in prior['checks'].values())):
        raise ValueError('Preserve independently audited preceding recovery failure')
    if gate.get('actual_R6_negative_trace_sha256') != sha(OWN / 'round06_development_plan/episodes/round06_fresh_full60/trajectory.jsonl'):
        raise ValueError('CPU scheduling regression belongs to another physical trace')
    source = OWN / 'round06_development_plan/episodes/round06_fresh_full60'
    config = copy.deepcopy(json.loads((source / 'config.json').read_text()))
    closure_path = OWN / 'round06_development_preparation/Root_actual_negative_closure.json'
    closure = json.loads(closure_path.read_text())
    if (closure.get('status') != 'completed_valid_negative'
            or closure.get('all7_checks_both_cases_passed') is not True
            or closure.get('full_task_success') is not False):
        raise ValueError('Independently audited actual long-carry failure required')
    scenarios = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    if (config['case']['seed'] in {case['seed'] for case in scenarios['cases']}
            or any(np.linalg.norm(np.asarray(config['scene']['box_pose']['position_xyz'])[:2]
                                  - np.asarray(case['scene']['box_pose']['position_xyz'])[:2]) < 1e-6
                   for case in scenarios['cases'])):
        raise ValueError('A formal position or seed cannot become TRAIN data')
    reclassification = OWN / 'diagnostics/development_to_TRAIN_recovery_v6.json'
    classification = json.loads(reclassification.read_text())
    if (classification.get('source_trace_sha256') != sha(source / 'trajectory.jsonl')
            or classification.get('fresh_development_cohort_required') is not True
            or classification.get('formal20_unchanged_sha256') != sha(SHARED / 'scenarios_frozen_v3.json')):
        raise ValueError('Explicit prospective TRAIN reclassification required')
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    existing = sum(p.stat().st_size for base in (OWN, backup) for p in base.rglob('*') if p.is_file())
    growth = 2 * 1024**3
    if existing + growth > 64 * 1024**3 or shutil.disk_usage(OWN).free - growth < 32 * 1024**3:
        raise RuntimeError('Combined64GiB cap and global32GiB floor required')
    destination.mkdir()
    folder, config_path = destination / 'episode', destination / 'config.json'
    config.update(policy_id='privileged_scripted_TRAIN_teacher', HFmodel=None,
                  case={**config['case'], 'case_id': 'privileged_TRAIN_recovery06', 'cluster_id': 'TRAIN_reclassified_nonformal_R6_2026101803'},
                  output_dir=str(folder), max_run_wall_seconds=780, save_full_rgb=True,
                  policy_command=[str(OWN / 'venv/bin/python'), '-m',
                                  'experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v6',
                                  '--config', str(config_path), '--decoded-evidence', str(folder / 'decoded.jsonl')])
    declaration = {'schema': 'lingbot_privileged_scripted_TRAIN_collector_v6', 'collector_version': 6,
                   'TRAIN_only': True, 'counts_as_LingBot_success': False,
                   'autonomous_model_policy': False, 'formal_benchmark_trial': False,
                   'physics_contacts_only_object_motion': True, 'runtime_teleport_or_attachment': False,
                   'new_student_observation_channels': False, 'source_full_task_success': False,
                   'lowering_schedule': {'center_error_below_m': .045, 'footprint_margin_above_m': .015, 'held_ready_controls': 50, 'continues_bilateral_XY_correction_while_lowering': True},
                   'floor_release_schedule': {'freeze_body_height_at_floor_clearance_m': .025, 'minimum_footprint_margin_m': .025, 'maximum_center_error_m': .040, 'held_ready_controls': 50},
                   'initial_replay_controls': PREFIX_CONTROLS,
                   'former_development_scene_now_TRAIN': True, 'fresh_development_cohort_required': True,
                   'source_trace': str(source / 'trajectory.jsonl'), 'source_result': str(source / 'result.json'),
                   'object_geometry': config['geometry_definition'], 'robot_geometry': config['robot_geometry'],
                   'reclassification': str(reclassification)}
    for key in ('source_trace', 'source_result', 'object_geometry', 'robot_geometry', 'reclassification'):
        declaration[key + '_sha256'] = sha(declaration[key])
    validate_source(declaration)
    config['privileged_TRAIN_teacher'] = declaration
    write(destination / 'cohort.json', {'schema': 'lingbot_privileged_TRAIN_collector_cohort_v1',
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'declaration': declaration, 'scene': config['scene'], 'seed': config['case']['seed'],
          'full60s_physics_thresholds_unchanged': True,
          'only_after_independent_complete_physical_success_may_admit_full_demonstration': True,
          'LingBot_task_goal_completed': False, 'finite_GPU_seconds': 900,
          'storage_growth_bound_bytes': growth, 'combined_owned_bytes_before': existing,
          'formal20_scenarios_sha256': sha(SHARED / 'scenarios_frozen_v3.json')})
    frozen = dict(config['frozen_files'])
    starts = [Path(__file__), Path(__file__).with_name('privileged_teacher_v6.py'),
              Path(__file__).with_name('run_teacher_case_v6.py'), Path(__file__).with_name('teacher_v6_audit.py'),
              Path(__file__).with_name('teacher_sim_entry_v6.py'), Path(__file__).with_name('teacher_measurement_v6.py'),
              ROOT / 'experiments/lingbot_vla2_g1/job.py', ROOT / 'experiments/lingbot_vla2_g1/budget_reallocated_v7.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    for path in (gate_path, closure_path, reclassification, prior_audit, prior_result,
                 OWN / 'round06_development_plan/episodes/round06_fresh_full60/trajectory.jsonl', destination / 'cohort.json',
                 source / 'config.json', source / 'result.json', source / 'trajectory.jsonl',
                 OWN / 'placement_recovery_teacher05/episode/trajectory.jsonl',
                 OWN / 'placement_recovery_teacher05/Root_actual_negative_closure.json'):
        frozen[str(path)] = sha(path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Previously frozen input changed: ' + name)
    config['frozen_files'] = dict(frozen)
    write(config_path, validate_config(config))
    frozen[str(config_path)] = sha(config_path)
    py, job_path, result = str(OWN / 'venv/bin/python'), destination / 'job.json', folder / 'result.json'
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', '01a145c6-0b86-7623-a869-086b33292199', 'evaluation', 900)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_reallocated_v7', '--ledger', str(OWN / 'gpu_budget.json'),
                      '--stage', 'evaluation', '--max-seconds', '900', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result), '--result-schema', 'pi05_g1_mobile_case_result_v1',
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        write(job_path, {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'evaluation', 'max_seconds': 900,
              'fifo_ticket': ticket['ticket'], 'result_file': str(result), 'result_schema': 'pi05_g1_mobile_case_result_v1',
              'workload_argv': [py, '-m', 'experiments.lingbot_vla2_g1.diagnostics.run_teacher_case_v6', '--config', str(config_path)],
              'supervisor_argv': supervisor, 'frozen_files': frozen, 'immutable_payloads': {},
              'budget_allocation': allocation, 'formal_benchmark_trial': False, 'LingBot_success_claim': False})
        copied = {}
        for name, expected in frozen.items():
            if Path(name).suffix == '.py' or Path(name).is_relative_to(destination):
                target = destination / 'source_snapshot' / name.lstrip('/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(name, target)
                if sha(target) != expected:
                    raise ValueError('Source changed during snapshot')
                copied[str(target)] = expected
        write(destination / 'source_snapshot_manifest.json', {'status': 'completed', 'job_sha256': sha(job_path),
              'copied_files_sha256': copied, 'large_model_or_old_trace_payloads_copied': False})
        write(destination / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path),
              'coordinator_unit': 'lingbot-TRAIN-recovery06-' + ticket['ticket'] + '.service',
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'prepared': str(destination), 'ticket': ticket['ticket'], 'job_sha256': sha(job_path), 'frozen_files': len(frozen)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
