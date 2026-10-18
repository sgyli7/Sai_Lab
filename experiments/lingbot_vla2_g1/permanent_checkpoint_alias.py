"""Keep an already verified historical checkpoint in permanent storage once.

The working directory becomes an alias to its byte-identical permanent copy.
No unique model data is removed. The CLI requires every owned GPU turn closed.
"""
import argparse
import json
import os
from pathlib import Path
import uuid

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .data import OWN, ROOT, sha
from .backup_round06 import BACKUP


def inventory(folder):
    folder = Path(folder)
    files = {p.name: p for p in folder.iterdir()}
    if set(files) != {'metadata.json', 'trainable.safetensors'} or any(not p.is_file() for p in files.values()):
        raise ValueError('An exact immutable two-file checkpoint is required')
    values = {name: sha(path) for name, path in files.items()}
    metadata = json.loads(files['metadata.json'].read_text())
    if metadata.get('weights_sha256') != values['trainable.safetensors']:
        raise ValueError('Checkpoint metadata and actual weights differ')
    return values


def create_alias(target, link):
    link.symlink_to(target, target_is_directory=True)


def use_permanent_copy(local, permanent):
    local, permanent = [Path(p).absolute() for p in (local, permanent)]
    if not local.is_relative_to(OWN) or not permanent.is_relative_to(BACKUP):
        raise ValueError('Only owned working and permanent checkpoint paths are allowed')
    if local.is_symlink():
        if local.resolve() != permanent.resolve():
            raise ValueError('Existing checkpoint alias points elsewhere')
        return inventory(local)
    if not local.parent.resolve().is_relative_to(OWN.resolve()):
        raise ValueError('A regular owned working parent is required')
    expected = inventory(local)
    if inventory(permanent) != expected:
        raise ValueError('Permanent checkpoint is not byte-identical')
    staging = local.with_name(local.name + '.verified_duplicate_' + uuid.uuid4().hex)
    link = local.with_name(local.name + '.permanent_alias_' + uuid.uuid4().hex)
    local.rename(staging)
    try:
        create_alias(permanent, link)
        os.replace(link, local)
        if inventory(local) != expected or inventory(permanent) != expected:
            raise ValueError('Checkpoint changed during alias transition')
    except BaseException:
        if link.is_symlink():
            link.unlink()
        if local.is_symlink():
            local.unlink()
        if not local.exists():
            staging.rename(local)
        raise
    # Only the two verified duplicate files are removed; the permanent model
    # and every original working path remain readable with identical bytes.
    for name in expected:
        (staging / name).unlink()
    staging.rmdir()
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('local', 'permanent', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or not args.receipt.absolute().is_relative_to(OWN):
        raise FileExistsError('Fresh owned operation receipt required')
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    if any(t['owner'] == '01a145c6-0b86-7623-a869-086b33292199' and t['status'] in ('waiting', 'running')
           for t in queue['tickets']):
        raise ValueError('All owned GPU turns must close before changing working aliases')
    if any(not cleanup_confirmed(row) for row in json.loads((OWN / 'gpu_budget.json').read_text())['runs']):
        raise ValueError('Owned GPU cleanup is incomplete')
    values = use_permanent_copy(args.local, args.permanent)
    with args.receipt.open('x') as stream:
        json.dump({'status': 'completed', 'working_path': str(args.local.absolute()),
                   'permanent_path': str(args.permanent.absolute()), 'files_sha256': values,
                   'all_unique_checkpoint_bytes_retained': True,
                   'working_path_is_alias_to_permanent_copy': True, 'goal_completed': False}, stream, indent=2)
    print(json.dumps({'status': 'completed', 'all_unique_checkpoint_bytes_retained': True}))


if __name__ == '__main__':
    main()
