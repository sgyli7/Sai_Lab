"""Freeze one finite TRAIN-only physical placement collection, then FIFO queue."""
import copy
import datetime
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from experiments.lingbot_vla2_g1.budget_continuation import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, SHARED, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write
from experiments.pi05_g1_mobile.simulator import validate_config


def main():
    destination = OWN / 'placement_teacher01'
    if destination.exists():
        raise FileExistsError('Preserve every prior preparation and trial')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used = sum(r['elapsed_seconds'] for r in ledger['runs'] if r['stage'] == 'evaluation')
    if allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 900 or allocation['stage_limits_seconds']['evaluation'] - used < 900:
        raise ValueError('Finite900seconds must fit total/stage allocation')
    gate_path = OWN / 'diagnostics/privileged_teacher_CPU_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 3:
        raise ValueError('Actual TRAIN replay/navigation/attribution CPU checks required')
    for name, expected in gate['source_files_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Collector changed after CPU gate: ' + name)
    source = OWN / 'resumed_turn01/episodes/resume01_native_teacher01'
    config = copy.deepcopy(json.loads((source / 'config.json').read_text()))
    result = json.loads((source / 'result.json').read_text())
    if result['case']['seed'] != 2026101702 or result['success'] is not False or sha(source / 'trajectory.jsonl') != result['trace_sha256']:
        raise ValueError('Independently admitted failed TRAIN source required')
    jacobian_gate = json.loads((OWN / 'diagnostics/training_teacher_measured_Jacobian_CPU_gate.json').read_text())
    if (jacobian_gate.get('passed') is not True
            or jacobian_gate.get('actual_native_trace_sha256') != result['trace_sha256']
            or jacobian_gate.get('source_sha256') != sha(Path(__file__).with_name('training_teacher_kinematics.py'))
            or jacobian_gate.get('geometry_sha256') != sha(config['robot_geometry'])):
        raise ValueError('Measured-link kinematics gate is not bound to actual source')
    protocol_path = OWN / 'diagnostics/teacher_CPU_protocol_gate.json'
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get('passed') is not True or protocol.get('GPU_started') is not False
            or protocol.get('teacher_sha256') != sha(Path(__file__).with_name('privileged_teacher.py'))):
        raise ValueError('Actual foreground collector JSONL protocol CPU gate required')
    scenarios = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    if any(config['scene'] == case['scene'] for case in scenarios['cases']):
        raise ValueError('TRAIN physical reset must not equal a formal position')
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    existing = sum(p.stat().st_size for base in (OWN, backup) for p in base.rglob('*') if p.is_file())
    growth = 2 * 1024 ** 3
    if existing + growth > 48 * 1024 ** 3 or shutil.disk_usage(OWN).free - growth < 32 * 1024 ** 3:
        raise RuntimeError('Combined48GiB cap and global32GiB floor required')
    destination.mkdir()
    folder = destination / 'episode'
    config_path = destination / 'config.json'
    config.update(policy_id='privileged_scripted_TRAIN_teacher', HFmodel=None,
                  case={**config['case'], 'case_id': 'privileged_TRAIN_placement01', 'cluster_id': 'TRAIN_native2026101702_recollection'},
                  output_dir=str(folder), max_run_wall_seconds=780, save_full_rgb=True,
                  policy_command=[str(OWN / 'venv/bin/python'), '-m',
                                  'experiments.lingbot_vla2_g1.diagnostics.privileged_teacher',
                                  '--config', str(config_path), '--decoded-evidence', str(folder / 'decoded.jsonl')])
    declaration = {'schema': 'lingbot_privileged_scripted_TRAIN_collector_v1',
                   'TRAIN_only': True, 'counts_as_LingBot_success': False,
                   'autonomous_model_policy': False, 'formal_benchmark_trial': False,
                   'physics_contacts_only_object_motion': True,
                   'runtime_teleport_or_attachment': False,
                   'new_student_observation_channels': False,
                   'source_full_task_success': False,
                   'initial_replay_controls': 1462,
                   'source_trace': str(source / 'trajectory.jsonl'),
                   'source_result': str(source / 'result.json'),
                   'object_geometry': config['geometry_definition'],
                   'robot_geometry': config['robot_geometry']}
    for key in ('source_trace', 'source_result', 'object_geometry', 'robot_geometry'):
        declaration[key + '_sha256'] = sha(declaration[key])
    config['privileged_TRAIN_teacher'] = declaration
    cohort = {'schema': 'lingbot_privileged_TRAIN_collector_cohort_v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'declaration': declaration, 'scene': config['scene'],
              'seed': config['case']['seed'], 'full60s_physics_thresholds_unchanged': True,
              'only_after_independent_complete_physical_success_may_admit_full_demonstration': True,
              'LingBot_task_goal_completed': False, 'finite_GPU_seconds': 900,
              'storage_growth_bound_bytes': growth, 'combined_owned_bytes_before': existing,
              'host_reserve_GiB': 16, 'formal20_scenarios_sha256': sha(SHARED / 'scenarios_frozen_v3.json'),
              'kinematics_CPU_gate': str(OWN / 'diagnostics/training_teacher_measured_Jacobian_CPU_gate.json')}
    write(destination / 'cohort.json', cohort)
    frozen = dict(config['frozen_files'])
    frozen.update(gate['source_files_sha256'])
    starts = [ROOT / 'experiments/lingbot_vla2_g1/diagnostics' / name for name in (
        'privileged_teacher.py', 'teacher_measurement.py', 'teacher_sim_entry.py',
        'run_teacher_case.py', 'teacher_audit.py', 'prepare_teacher_turn.py')]
    starts += [ROOT / 'experiments/lingbot_vla2_g1/job.py', ROOT / 'experiments/lingbot_vla2_g1/budget_continuation.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    for path in (gate_path, protocol_path, source / 'config.json', source / 'result.json', source / 'trajectory.jsonl',
                 OWN / 'diagnostics/training_teacher_measured_Jacobian_CPU_gate.json', destination / 'cohort.json'):
        frozen[str(path)] = sha(path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Previously frozen input changed: ' + name)
    config['frozen_files'] = dict(frozen)
    config = validate_config(config)
    write(config_path, config)
    frozen[str(config_path)] = sha(config_path)
    py = str(OWN / 'venv/bin/python')
    job_path = destination / 'job.json'
    result_path = folder / 'result.json'
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', '01a145c6-0b86-7623-a869-086b33292199', 'evaluation', 900)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_continuation', '--ledger', str(OWN / 'gpu_budget.json'),
                      '--stage', 'evaluation', '--max-seconds', '900', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result_path), '--result-schema', 'pi05_g1_mobile_case_result_v1',
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        job = {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'evaluation', 'max_seconds': 900,
               'fifo_ticket': ticket['ticket'], 'result_file': str(result_path), 'result_schema': 'pi05_g1_mobile_case_result_v1',
               'workload_argv': [py, '-m', 'experiments.lingbot_vla2_g1.diagnostics.run_teacher_case', '--config', str(config_path)],
               'supervisor_argv': supervisor, 'frozen_files': frozen, 'immutable_payloads': {},
               'budget_allocation': allocation, 'formal_benchmark_trial': False, 'LingBot_success_claim': False}
        write(job_path, job)
        copied = {}
        for name, expected in frozen.items():
            if Path(name).suffix == '.py' or Path(name).is_relative_to(destination):
                target = destination / 'source_snapshot' / name.lstrip('/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(name, target)
                if sha(target) != expected:
                    raise ValueError('Frozen source changed during copy')
                copied[str(target)] = expected
        write(destination / 'source_snapshot_manifest.json', {'status': 'completed', 'job_sha256': sha(job_path),
              'copied_files_sha256': copied, 'actual_copied_bytes': sum(Path(n).stat().st_size for n in copied),
              'large_model_or_old_trace_payloads_copied': False})
        write(destination / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path),
              'coordinator_unit': 'lingbot-TRAIN-placement01-' + ticket['ticket'] + '.service',
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'prepared': str(destination), 'ticket': ticket['ticket'], 'job_sha256': sha(job_path),
                          'frozen_files': len(frozen), 'actual_snapshot_files': len(copied)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
