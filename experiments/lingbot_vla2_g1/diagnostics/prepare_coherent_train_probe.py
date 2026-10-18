"""CPU admission for a finite coherent TRAIN action control, no robot rollout."""
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.vla_resources import gpu_queue
from experiments.lingbot_vla2_g1.budget_reallocated_v5 import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write
from .coherent_train_probe import SCHEMA


def main():
    output = OWN / 'coherent_TRAIN_action_probe01'
    if output.exists():
        raise FileExistsError('Fresh control output required')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used = sum(r['elapsed_seconds'] for r in ledger['runs'] if r['stage'] == 'preflight')
    if (allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 300
            or allocation['stage_limits_seconds']['preflight'] - used < 300):
        raise ValueError('Finite300-second control must fit the unchanged allocation')
    manifest_path = OWN / 'diagnostics/coherent_TRAIN_action_probe_manifest_v1.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != SCHEMA or not manifest['TRAIN_only'] or len(manifest['samples']) != 20:
        raise ValueError('Twenty TRAIN observations required')
    first = manifest['samples'][0]
    rows = [json.loads(line) for line in Path(first['trace']).read_text().splitlines()]
    targets = np.asarray([[r['applied_action32'] for r in rows[x['tick']+1:x['tick']+51]]
                          for x in manifest['samples'][:8]], np.float32)
    with np.load(manifest['R2_baseline_npz'], allow_pickle=False) as arrays:
        if not np.array_equal(targets, arrays['teacher_next_actions32']):
            raise ValueError('Exact R2/R4 TRAIN reference alignment failed')
    starts = [Path(__file__), Path(__file__).with_name('coherent_train_probe.py'),
              ROOT / 'experiments/lingbot_vla2_g1/budget_reallocated_v5.py',
              ROOT / 'experiments/lingbot_vla2_g1/job.py']
    frozen = dict(manifest['frozen_inputs'])
    for path in [*experiment_sources(starts), *(OWN / 'runtime_source').rglob('*.py'),
                 *(OWN / 'qwen_processor').glob('*'), *(OWN / 'base_bf16').glob('*.json'),
                 OWN / 'bf16_conversion_receipt.json', OWN / 'runtime_overlay_receipt.json',
                 OWN / 'environment_receipt.json', manifest_path,
                 OWN / 'diagnostics/coherent_placement_hypotheses.json']:
        if path.is_file():
            frozen[str(path)] = sha(path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Control input changed: ' + name)
    payloads = json.loads((OWN / 'round05_development_plan/plan.json').read_text())['immutable_payloads']
    if len(payloads) != 7 or str(Path(manifest['delta']) / 'trainable.safetensors') not in payloads:
        raise ValueError('Actual R5 final plus six base shards required')
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    size = sum(p.stat().st_size for root in (OWN, backup) for p in root.rglob('*') if p.is_file())
    if size + 1024**3 > 64 * 1024**3 or shutil.disk_usage(OWN).free < 25 * 1024**3:
        raise RuntimeError('Committed storage cap and global floor required')
    output.mkdir()
    result, job_path = output / 'execution/result.json', output / 'job.json'
    py = str(OWN / 'venv/bin/python')
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination',
                               '01a145c6-0b86-7623-a869-086b33292199', 'preflight', 300)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_reallocated_v5',
                      '--ledger', str(OWN / 'gpu_budget.json'), '--stage', 'preflight',
                      '--max-seconds', '300', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result), '--result-schema', SCHEMA,
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        write(job_path, {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'preflight',
              'max_seconds': 300, 'fifo_ticket': ticket['ticket'], 'result_file': str(result),
              'result_schema': SCHEMA, 'workload_argv': [py, '-m',
                  'experiments.lingbot_vla2_g1.diagnostics.coherent_train_probe', '--manifest', str(manifest_path),
                  '--output', str(result.parent)], 'supervisor_argv': supervisor,
              'frozen_files': frozen, 'immutable_payloads': payloads, 'budget_allocation': allocation,
              'actual_robot_success_claimed': False})
        copied = {}
        for name, digest in frozen.items():
            if Path(name).suffix != '.py':
                continue
            target = output / 'source_snapshot' / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
            if sha(target) != digest:
                raise ValueError('Source snapshot changed')
            copied[str(target)] = digest
        write(output / 'CPU_gate.json', {'status': 'passed', 'GPU_used': False,
              'paired_native_target_arrays_exact_equal': True, 'samples': 20,
              'source_images_verified_against_measured_row_pixel_hash': True,
              'TRAIN_only': True, 'source_snapshots_sha256': copied, 'job_sha256': sha(job_path)})
        write(output / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path),
              'job_sha256': sha(job_path), 'coordinator_unit': 'lingbot-coherent-TRAIN-probe01-' + ticket['ticket'] + '.service',
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'ticket': ticket['ticket'], 'job_sha256': sha(job_path), 'frozen_files': len(frozen)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
