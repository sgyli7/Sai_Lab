"""Preserve actual R5 checkpoints after owned training has fully closed.

This is a checkpoint backup, never a physical-performance completion receipt.
The unchanged six base shards remain separately referenced by their hashes.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

from .data import OWN, ROOT, sha
from .make_plan import write
from .round05_selection import admit

BACKUP = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')


def write_pointer(destination):
    expected = {'schema': 'lingbot_R5_checkpoint_backup_pointer_v1', 'status': 'completed',
                'manifest': str(destination / 'manifest.json'),
                'manifest_sha256': sha(destination / 'manifest.json'), 'physical_goal_completed': False}
    path = OWN / 'round05_checkpoint_backup_receipt.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Existing backup pointer differs; do not overwrite it')
    else:
        write(path, expected)


def copy_checked(source, target, expected):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, target.open('xb') as writer:
        shutil.copyfileobj(reader, writer, 16 * 1024 ** 2)
        writer.flush()
        os.fsync(writer.fileno())
    if sha(target) != expected:
        raise ValueError('Permanent checkpoint copy differs: ' + str(target))
    for path in (source, target):
        with path.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def backup(training_result):
    result, final_delta, frozen, payloads, provenance = admit(training_result)
    final_identity = payloads[str(final_delta / 'trainable.safetensors')]
    destination = BACKUP / ('training_task_round05_final_' + final_identity['sha256'][:12])
    staging = destination.with_name('.' + destination.name + '.incomplete')
    if destination.exists():
        report = json.loads((destination / 'manifest.json').read_text())
        if report.get('physical_candidate_weights_sha256') != final_identity['sha256']:
            raise ValueError('Existing permanent checkpoint identity differs')
        for name, expected in report['saved_files_sha256'].items():
            if sha(name) != expected:
                raise ValueError('Existing permanent backup changed: ' + name)
        for role in report['checkpoint_roles'].values():
            path = Path(role['path'])
            if (str(path / 'trainable.safetensors') not in report['saved_files_sha256']
                    and sha(path / 'trainable.safetensors') != role['weights_sha256']):
                raise ValueError('Referenced historical permanent checkpoint changed')
            if sha(path / 'metadata.json') != role['metadata_sha256']:
                raise ValueError('Permanent checkpoint metadata changed')
        write_pointer(destination)
        return report
    if staging.exists():
        raise FileExistsError('Keep partial prior backup evidence; use a fresh reviewed attempt')
    best = Path(result['selected_delta']).absolute()
    best_meta = json.loads((best / 'metadata.json').read_text())
    best_existing = BACKUP / 'training/training_round02/selected/trainable.safetensors'
    reuse_best = best == OWN / 'training_round02/selected'
    if reuse_best and (not best_existing.exists() or sha(best_existing) != best_meta['weights_sha256']):
        raise ValueError('Previously permanent R2 source-best copy must remain intact')
    if reuse_best and sha(best_existing.with_name('metadata.json')) != sha(best / 'metadata.json'):
        raise ValueError('Permanent R2 metadata differs from the actual source-best checkpoint')
    copies = {'physical_final': final_delta}
    if best != final_delta and not reuse_best:
        copies['source_validation_best'] = best
    required = sum((path / 'trainable.safetensors').stat().st_size for path in copies.values()) + 32 * 1024 ** 2
    combined = sum(path.stat().st_size for root in (OWN, BACKUP) for path in root.rglob('*') if path.is_file())
    if combined + required > 64 * 1024 ** 3 or shutil.disk_usage(BACKUP).free - required < 32 * 1024 ** 3:
        raise RuntimeError('Current combined64GiB commitment and global32GiB floor must fit backup')
    staging.mkdir()
    saved = {}
    roles = {}
    for role, source in copies.items():
        metadata = json.loads((source / 'metadata.json').read_text())
        for name in ('metadata.json', 'trainable.safetensors'):
            digest = metadata['weights_sha256'] if name.endswith('safetensors') else sha(source / name)
            copy_checked(source / name, staging / role / name, digest)
            saved[str(destination / role / name)] = digest
        roles[role] = {'path': str(destination / role), 'step': metadata['step'],
                       'weights_sha256': metadata['weights_sha256'], 'origin_run': source.parent.name,
                       'round05_new_updates': metadata['step'],
                       'model_weight_lineage_updates': metadata['model_weight_lineage_optimizer_updates'],
                       'metadata_sha256': sha(source / 'metadata.json')}
    if best == final_delta:
        roles['source_validation_best'] = roles['physical_final']
    elif reuse_best:
        roles['source_validation_best'] = {'path': str(best_existing.parent),
                                           'step': best_meta['step'], 'weights_sha256': best_meta['weights_sha256'],
                                           'origin_run': 'training_round02', 'round05_new_updates': 0,
                                           'model_weight_lineage_updates': 1000,
                                           'metadata_sha256': sha(best / 'metadata.json')}
    run = Path(training_result).absolute().parent
    for source in [run / name for name in ('training_result.json', 'recipe.json', 'progress.jsonl')]:
        digest = sha(source)
        copy_checked(source, staging / 'training' / source.name, digest)
        saved[str(destination / 'training' / source.name)] = digest
    for source in (OWN / 'training_task_round05_job.json',
                   OWN / 'source_snapshots/training_task_round05/manifest.json',
                   OWN / 'round05_CPU_preparation/decision.json', OWN / 'round05_CPU_gate.json'):
        digest = sha(source)
        copy_checked(source, staging / 'provenance' / source.name, digest)
        saved[str(destination / 'provenance' / source.name)] = digest
    for name in ('runtime.py', 'service_pointmass.py', 'constant_inverse.py', 'contract.py', 'backends.py',
                 'training_task_round05.py', 'robust_training_v5.py'):
        source = ROOT / 'experiments/lingbot_vla2_g1' / name
        digest = sha(source)
        copy_checked(source, staging / 'runtime_sources' / name, digest)
        saved[str(destination / 'runtime_sources' / name)] = digest
    for name in ('train_meanstd.json', 'contract.json', 'bf16_conversion_receipt.json', 'runtime_overlay_receipt.json', 'environment_receipt.json'):
        source = OWN / name
        digest = sha(source)
        copy_checked(source, staging / 'runtime_identity' / name, digest)
        saved[str(destination / 'runtime_identity' / name)] = digest
    report = {'schema': 'lingbot_R5_actual_checkpoint_permanent_backup_v1', 'status': 'completed',
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'training_result': str(Path(training_result).absolute()),
              'training_result_sha256': sha(training_result), 'actual_optimizer_updates': result['optimizer_steps'],
              'physical_candidate_weights_sha256': final_identity['sha256'],
              'checkpoint_roles': roles, 'saved_files_sha256': saved,
              'unchanged_base_payloads_referenced_not_duplicated':
                  {name: identity for name, identity in payloads.items() if Path(name).parent == OWN / 'base_bf16'},
              'actual_training_and_cleanup_provenance': provenance,
              'physical_evaluation_completed_by_this_command': False, 'overall_robot_goal_completed': False}
    write(staging / 'manifest.json', report)
    with (staging / 'manifest.json').open('rb') as stream:
        os.fsync(stream.fileno())
    os.rename(staging, destination)
    write_pointer(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', type=Path, default=OWN / 'training_task_round05/training_result.json')
    args = parser.parse_args()
    report = backup(args.training_result)
    print(json.dumps({'status': report['status'], 'updates': report['actual_optimizer_updates'],
                      'checkpoint_roles': report['checkpoint_roles'], 'robot_goal_completed': False}))


if __name__ == '__main__':
    main()
