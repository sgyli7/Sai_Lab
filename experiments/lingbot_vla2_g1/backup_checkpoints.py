"""Verified checkpoint backup with bounded clean file cache; keep original files."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / '.scratch/lingbot_vla2_g1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_verified(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    flush_from = 0
    with source.open('rb', buffering=0) as src, target.open('xb', buffering=0) as dst:
        while block := src.read(8 * 1024**2):
            dst.write(block)
            digest.update(block)
            size += len(block)
            if size - flush_from >= 64 * 1024**2:
                if shutil.disk_usage(target).free < 24 * 1024**3:
                    raise RuntimeError('Keep24GiB global free reserve; partial backup retained')
                os.fsync(dst.fileno())
                for stream in [src, dst]:
                    os.posix_fadvise(stream.fileno(), flush_from, size-flush_from, os.POSIX_FADV_DONTNEED)
                flush_from = size
        os.fsync(dst.fileno())
        for stream in [src, dst]:
            os.posix_fadvise(stream.fileno(), flush_from, size-flush_from, os.POSIX_FADV_DONTNEED)
    actual = digest.hexdigest()
    if expected and expected != actual:
        raise ValueError('Source differs from selected metadata: ' + str(source))
    check = hashlib.sha256()
    with target.open('rb', buffering=0) as stream:
        offset = 0
        while block := stream.read(8 * 1024**2):
            check.update(block)
            os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
    if check.hexdigest() != actual:
        raise ValueError('Actual saved backup SHA differs')
    return {'source': str(source), 'target': str(target), 'bytes': size,
            'sha256': actual, 'readback_sha256': check.hexdigest()}


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = Path('/home/ethan/ProjectBackups') / ('LingBotVLA2_G1_checkpoints_' + stamp)
    if destination.exists():
        raise FileExistsError(destination)
    files = {}
    for run in ['training_round01', 'training_round02']:
        folder = OWN / run
        for name in ['training_result.json', 'recipe.json', 'progress.jsonl', 'carry_admission.json']:
            p = folder / name
            if p.is_file():
                files[p] = Path('training') / run / name
        for name in ['metadata.json', 'trainable.safetensors']:
            files[folder / 'selected' / name] = Path('training') / run / 'selected' / name
    for name in ['train_meanstd.json', 'contract.json', 'bf16_conversion_receipt.json',
                 'environment_receipt.json', 'runtime_overlay_receipt.json', 'hardware_and_scope.json',
                 'benchmark_pose_speed_summary_v2.json', 'benchmark_pose_speed_summary_v2.csv',
                 'experiment_completion_receipt.json', 'round02_training_completion_review.json',
                 'original_goal_wall_deadline.json', 'round02_resource_stop_rule.json',
                 'round02_actual_resource_stop_signal.json']:
        files[OWN / name] = Path('metadata') / name
    for p in sorted((ROOT / 'experiments/lingbot_vla2_g1').rglob('*')):
        if p.is_file() and p.suffix in ['.py', '.md']:
            files[p] = Path('source/experiments/lingbot_vla2_g1') / p.relative_to(ROOT / 'experiments/lingbot_vla2_g1')
    for p in sorted((OWN / 'runtime_source').rglob('*.py')):
        files[p] = Path('source/runtime_source') / p.relative_to(OWN / 'runtime_source')
    for name in ['plan.json', 'selection_receipt.json']:
        files[OWN / 'evaluation_plan_round02' / name] = Path('evaluation') / name
    for p in sorted((OWN / 'evaluation_plan_round02/configs').glob('*.json')):
        files[p] = Path('evaluation/configs') / p.name
    backup_bytes = sum(p.stat().st_size for p in files)
    own_bytes = sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file())
    bound = own_bytes + backup_bytes + 1024**3 + 256 * 1024**2
    if bound > 36 * 1024**3 or shutil.disk_usage(OWN).free - backup_bytes < 24 * 1024**3:
        raise RuntimeError('Combined owned outputs and backup exceed admitted bounds')
    destination.mkdir(parents=True)
    records = []
    for source, relative in files.items():
        expected = None
        if source.name == 'trainable.safetensors':
            expected = json.loads(source.with_name('metadata.json').read_text())['weights_sha256']
        records.append(copy_verified(source, destination / relative, expected))
    plan = json.loads((OWN / 'evaluation_plan_round02/plan.json').read_text())
    manifest = {'schema': 'lingbot_selected_parameters_verified_backup_v1', 'status': 'completed',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'directory': str(destination), 'scope': 'Both selected trained-parameter snapshots and reproduction metadata; original files retained',
                'full_model_self_contained': False,
                'external_base_model_payloads_not_duplicated': {p: v for p, v in plan['immutable_payloads'].items() if Path(p).parent == OWN / 'base_bf16'},
                'external_original_dataset': {'repo': 'nvidia/Arena-G1-Loco-Manipulation-Task', 'revision': '97f7b5a4135e4e6a206d394c9b6ec0253f1558a7', 'local': str(ROOT / '.scratch/pi05_g1/round03_mobile/data/lerobot')},
                'original_raw_videos_and_traces_retained_in_workspace': True,
                'new_physical_pilot_not_yet_included': True,
                'files': records, 'backup_bytes': backup_bytes,
                'combined_owned_outputs_future_bound_bytes': bound, 'own_output_cap_bytes': 36 * 1024**3,
                'all_saved_files_SHA_verified_by_readback': True,
                'clean_cache_policy': 'Own source and destination files only; fsync64MiB batches, advise consumed clean ranges; no global cache eviction'}
    path = destination / 'manifest.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {'schema': manifest['schema'], 'status': 'completed', 'manifest': str(path),
               'manifest_sha256': sha(path), 'file_count': len(records),
               'backup_bytes': backup_bytes, 'combined_owned_outputs_future_bound_bytes': bound,
               'full_model_self_contained': False, 'external_base_model_preserved': True}
    with (OWN / 'selected_checkpoints_backup_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
