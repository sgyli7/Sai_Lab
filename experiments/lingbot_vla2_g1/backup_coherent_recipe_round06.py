"""Permanently preserve a qualified coherent inference recipe, without weights duplication."""
import json
import os
from pathlib import Path
import shutil

from .backup_round06 import BACKUP, copy_checked
from .data import OWN, ROOT, sha
from .formal_task_v4.gates import admit_development
from .make_plan import write


def record_pointer(destination):
    value = {'manifest': str(destination / 'manifest.json'), 'manifest_sha256': sha(destination / 'manifest.json'),
             'formal20_completed': False, 'overall_robot_goal_completed': False}
    path = OWN / 'round06_coherent_recipe_backup_receipt.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing recipe backup pointer differs')
    else:
        write(path, value)


def backup():
    plan_path = OWN / 'round06_development_plan/plan.json'
    result_path = OWN / 'round06_development_preparation/execution/result.json'
    admitted = admit_development(plan_path, result_path, require_actual=True)
    if admitted['plan'].get('sampling_adapter') != 'coherent_antithetic_two_predictions_v1':
        raise ValueError('Prospectively declared coherent sampling recipe required')
    pointer = json.loads((OWN / 'round06_checkpoint_backup_receipt.json').read_text())
    if sha(pointer['manifest']) != pointer['manifest_sha256']:
        raise ValueError('Permanent R6 checkpoint manifest changed')
    model = json.loads(Path(pointer['manifest']).read_text())['checkpoint_roles']['physical_final']
    expected = admitted['plan']['immutable_payloads'][str(Path(admitted['selected_delta']) / 'trainable.safetensors')]['sha256']
    if model['weights_sha256'] != expected:
        raise ValueError('Same actual R6 checkpoint required')
    if (sha(Path(model['path']) / 'trainable.safetensors') != model['weights_sha256']
            or sha(Path(model['path']) / 'metadata.json') != model['metadata_sha256']):
        raise ValueError('Referenced permanent model checkpoint changed')
    destination = BACKUP / ('inference_coherent_R6_' + expected[:12])
    if destination.exists():
        report = json.loads((destination / 'manifest.json').read_text())
        for path, expected in report['saved_files_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Permanent recipe changed')
        record_pointer(destination)
        return report
    staging = destination.with_name('.' + destination.name + '.incomplete')
    if staging.exists():
        raise FileExistsError('Preserve partial prior recipe backup')
    combined = sum(p.stat().st_size for folder in (OWN, BACKUP) for p in folder.rglob('*') if p.is_file())
    if combined + 128 * 1024**2 > 64 * 1024**3 or shutil.disk_usage(BACKUP).free < 32 * 1024**3:
        raise RuntimeError('Preserve combined storage cap and global free floor')
    staging.mkdir()
    saved = {}
    files = [ROOT / 'experiments/lingbot_vla2_g1' / name for name in
             ('coherent_sampling.py', 'service_coherent_noise.py', 'constant_inverse.py',
              'runtime.py', 'contract.py', 'backends.py')]
    files += [ROOT / 'experiments/lingbot_vla2_g1/diagnostics/development_audit_v2.py',
              OWN / 'coherent_noise01_CPU_gate.json', plan_path, result_path]
    files += [Path(record['audit']) for record in admitted['result']['jobs']]
    for source in files:
        target = 'sources/' + source.name if source.suffix == '.py' else 'evidence/' + source.name
        # Plan and result have distinct names; all other evidence is case-scoped.
        if source == plan_path:
            target = 'evidence/development_plan.json'
        elif source == result_path:
            target = 'evidence/development_result.json'
        digest = sha(source)
        copy_checked(source, staging / target, digest)
        saved[str(destination / target)] = digest
    report = {'schema': 'lingbot_coherent_inference_permanent_recipe_v1', 'status': 'completed',
              'sampling_adapter': admitted['plan']['sampling_adapter'], 'common_noise_seed': 42,
              'predictions_per_query': 2, 'combination': 'Float32 mean of two actual predictions from fixed opposite noise vectors.',
              'model_checkpoint_reference': model, 'checkpoint_manifest': pointer,
              'model_weights_duplicated': False, 'saved_files_sha256': saved,
              'actual_autonomous_full_development_success_verified': True,
              'formal20_completed': False, 'overall_robot_goal_completed': False}
    write(staging / 'manifest.json', report)
    os.rename(staging, destination)
    record_pointer(destination)
    return report


if __name__ == '__main__':
    print(json.dumps(backup()))
