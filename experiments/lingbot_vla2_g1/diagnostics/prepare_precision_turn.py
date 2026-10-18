"""Freeze a finite paired inference control only after previous GPU cleanup."""
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from experiments.lingbot_vla2_g1.budget_continuation import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write


def main():
    destination = OWN / 'paired_precision_control01'
    if destination.exists():
        raise FileExistsError('Preserve previous numerical controls')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used = sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage'] == 'preflight')
    if (allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 900
            or allocation['stage_limits_seconds']['preflight'] - used < 900):
        raise ValueError('Finite preflight control must fit actual remaining allocation')
    manifest_path = OWN / 'diagnostics/paired_precision_manifest_v1.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('TRAIN_only') is not True or len(manifest['samples']) != 8:
        raise ValueError('Frozen eight-sample TRAIN arithmetic control required')
    prior = json.loads((OWN / 'training_round03_dual_job.json').read_text())
    frozen = {**prior['frozen_files'], **manifest['frozen_inputs']}
    starts = [Path(__file__), Path(__file__).with_name('precision_probe.py'),
              ROOT / 'experiments/lingbot_vla2_g1/budget_continuation.py',
              ROOT / 'experiments/lingbot_vla2_g1/job.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    frozen[str(manifest_path)] = sha(manifest_path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Numerical-control dependency changed: ' + name)
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    size = sum(path.stat().st_size for root in (OWN, backup) for path in root.rglob('*') if path.is_file())
    if size + 1024 ** 3 > 64 * 1024 ** 3 or shutil.disk_usage(OWN).free < 33 * 1024 ** 3:
        raise RuntimeError('Existing shared storage cap and global floor required')
    destination.mkdir()
    result = destination / 'execution/result.json'
    job_path = destination / 'job.json'
    py = str(OWN / 'venv/bin/python')
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination',
                               '01a145c6-0b86-7623-a869-086b33292199', 'preflight', 900)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_continuation',
                      '--ledger', str(OWN / 'gpu_budget.json'), '--stage', 'preflight',
                      '--max-seconds', '900', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result),
                      '--result-schema', 'lingbot_paired_precision_TRAIN_result_v1', '--',
                      py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        job = {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'preflight', 'max_seconds': 900,
               'fifo_ticket': ticket['ticket'], 'result_file': str(result),
               'result_schema': 'lingbot_paired_precision_TRAIN_result_v1',
               'workload_argv': [py, '-m', 'experiments.lingbot_vla2_g1.diagnostics.precision_probe',
                                 '--manifest', str(manifest_path), '--output', str(result.parent)],
               'supervisor_argv': supervisor, 'frozen_files': frozen,
               'immutable_payloads': prior['immutable_payloads'], 'budget_allocation': allocation,
               'official_benchmark_reproduced': False, 'robot_goal_success_claim': False}
        write(job_path, job)
        write(destination / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path),
              'job_sha256': sha(job_path),
              'coordinator_unit': 'lingbot-paired-precision01-' + ticket['ticket'] + '.service',
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run',
                          '--ticket', ticket['ticket'], '--ledger', str(OWN / 'gpu_budget.json'),
                          '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'ticket': ticket['ticket'], 'job': str(job_path),
                          'job_sha256': sha(job_path), 'frozen_files': len(frozen)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'],
                         '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
