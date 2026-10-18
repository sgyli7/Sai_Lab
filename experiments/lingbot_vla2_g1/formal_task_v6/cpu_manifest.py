"""Snapshot this new pipeline without creating a formal plan or receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from experiments.lingbot_vla2_g1.data import sha
from .gates import CPU_ROOT


def publish(root=None):
    root = CPU_ROOT if root is None else Path(root).absolute()
    if root != CPU_ROOT:
        raise ValueError('CPU preparation has one artifact root')
    if (root / 'manifest.json').exists() or (root / 'source_snapshot').exists():
        raise FileExistsError('Preserve prior CPU preparation; choose a versioned artifact root in code')
    package = Path(__file__).resolve().parent
    files = sorted(path for path in package.rglob('*.py') if '__pycache__' not in path.parts)
    root.mkdir(parents=True, exist_ok=True)
    copied = {}
    snapshot = root / 'source_snapshot'
    for path in files:
        target = snapshot / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied[str(target)] = sha(target)
        if sha(target) != sha(path):
            raise ValueError('Pipeline source changed during CPU snapshot: ' + str(path))
    manifest = {'schema': 'lingbot_formal_task_v6_cpu_preparation_v1', 'formal_plan_generated': False,
                'formal_receipts_generated': False, 'gpu_requested': False, 'simulator_launched': False,
                'training_modified': False, 'synthetic_fixture': False, 'actual_robot_success_claimed': False,
                'reason': 'This artifact records CPU software preparation only. A formal plan requires separately verified training and actual autonomous development success.',
                'copied_files_sha256': copied,
                'entrypoints': {
                    'make_plan': 'python -m experiments.lingbot_vla2_g1.formal_task_v6.make_plan --training-result TRAIN --development-plan PLAN --development-result RESULT --output OUTPUT --episodes-root EPISODES',
                    'evaluate': 'python -m experiments.lingbot_vla2_g1.formal_task_v6.evaluate --plan PLAN --receipt RECEIPT --max-jobs 3 --resume-receipt PRIOR',
                    'aggregate': 'python -m experiments.lingbot_vla2_g1.formal_task_v6.aggregate --plan PLAN --receipt RECEIPT --output OUTPUT'}}
    path = root / 'manifest.json'
    temporary = path.with_name('manifest.json.incomplete')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    temporary.replace(path)
    return manifest


def main():
    manifest = publish()
    print(json.dumps({'formal_plan_generated': manifest['formal_plan_generated'], 'files': len(manifest['copied_files_sha256']),
                      'gpu_requested': manifest['gpu_requested']}))


if __name__ == '__main__':
    main()
