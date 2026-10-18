"""CPU-only finite FIFO preparation after new training and selection are closed."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from .budget_continuation import admit_allocation
from .data import OWN, ROOT, sha
from .make_plan import experiment_sources, write


def prepare(plan_path, output):
    plan_path, output = Path(plan_path).absolute(), Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned validation preparation required')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    plan = json.loads(plan_path.read_text())
    if (plan.get('schema') != 'lingbot_round03_development_plan_v1'
            or plan.get('formal20_benchmark') is not False or len(plan['jobs']) != 2
            or plan.get('max_GPU_seconds') != 1800 or plan.get('selected_new_step', 0) <= 0):
        raise ValueError('Reviewed genuinely new two-case development candidate required')
    frozen = dict(plan['frozen_files'])
    for path in experiment_sources([Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/job.py']):
        frozen[str(path)] = sha(path)
    frozen[str(plan_path)] = sha(plan_path)
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Frozen development input changed: ' + name)
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    existing = sum(p.stat().st_size for folder in (OWN, backup) for p in folder.rglob('*') if p.is_file())
    if existing + 2 * 1024 ** 3 > 48 * 1024 ** 3 or shutil.disk_usage(OWN).free < 27 * 1024 ** 3:
        raise RuntimeError('Preserve combined48GiB cap and24GiB global floor plus complete-case buffers')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used_eval = sum(r['elapsed_seconds'] for r in ledger['runs'] if r['stage'] == 'evaluation')
    if allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 1800 or allocation['stage_limits_seconds']['evaluation'] - used_eval < 1800:
        raise ValueError('Complete finite1800second validation turn must fit allocation')
    output.mkdir()
    execution = output / 'execution'
    result, job_path = execution / 'result.json', output / 'job.json'
    py = str(OWN / 'venv/bin/python')
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', '01a145c6-0b86-7623-a869-086b33292199', 'evaluation', 1800)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_continuation', '--ledger', str(OWN / 'gpu_budget.json'),
                      '--stage', 'evaluation', '--max-seconds', '1800', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result),
                      '--result-schema', 'lingbot_round03_development_execution_v1',
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        job = {'schema': 'lingbot_vla2_verified_job_v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'stage': 'evaluation', 'max_seconds': 1800, 'fifo_ticket': ticket['ticket'],
               'result_file': str(result), 'result_schema': 'lingbot_round03_development_execution_v1',
               'workload_argv': [py, '-m', 'experiments.lingbot_vla2_g1.run_development_round03', '--plan', str(plan_path), '--output', str(execution)],
               'supervisor_argv': supervisor, 'frozen_files': frozen, 'immutable_payloads': plan['immutable_payloads'],
               'budget_allocation': allocation, 'original_physics_acceptance_unchanged': True}
        write(job_path, job)
        snapshot = output / 'source_snapshot'
        copied = {}
        for name, digest in frozen.items():
            if Path(name).suffix == '.py':
                target = snapshot / name.lstrip('/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(name, target)
                if sha(target) != digest:
                    raise ValueError('Development source changed during copy')
                copied[str(target)] = digest
        write(output / 'source_snapshot_manifest.json', {'status': 'completed', 'job_sha256': sha(job_path),
              'copied_files_sha256': copied, 'actual_copied_bytes': sum(Path(n).stat().st_size for n in copied),
              'large_immutable_inputs_copied': False})
        write(output / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path),
              'coordinator_unit': 'lingbot-eval-round03-dev-' + ticket['ticket'] + '.service',
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path)}), flush=True)
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.output)


if __name__ == '__main__':
    main()
