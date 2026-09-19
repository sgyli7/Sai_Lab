"""Verify and open a relocated candidate bundle without rewriting its evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sim2sim.paths import sim2sim_root
from sim2sim.policy import OnnxPolicy
from .tasks import TASKS


def bundle_paths(directory):
    directory = Path(directory).expanduser().resolve()
    record = json.loads((directory / 'bundle.json').read_text())
    if not record.get('verification_completed_unix') or set(record['skills']) != set(TASKS):
        raise ValueError('Expected a verified bundle containing exactly nine skills')
    paths = {}
    for skill, task in TASKS.items():
        path = directory / 'models' / task.previous
        item = record['skills'][skill]
        for file, expected in [(path, item['sha256']),
                               (path.with_suffix('.manifest.json'), item['manifest_sha256'])]:
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Bundle checksum mismatch: {file}')
        paths[skill] = str(path)
    # Historical bank.json contains original-machine paths. Resolve only files
    # checked above; leave that provenance and the frozen directory unchanged.
    return paths


def verify(directory):
    paths = bundle_paths(directory)
    for path in paths.values():
        actor = OnnxPolicy(Path(path))
        actor.check_dims(14)
        action = actor.infer(np.zeros(61, np.float32))
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f'Invalid inference: {path}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--write-bank', type=Path, help='Create a relocated bank outside the frozen bundle')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--roller', action='store_true')
    args = parser.parse_args()
    paths = verify(args.directory)
    if args.write_bank:
        if args.write_bank.resolve().is_relative_to(args.directory.expanduser().resolve()):
            parser.error('--write-bank must be outside the frozen bundle')
        with args.write_bank.open('x') as stream:
            json.dump(paths, stream, indent=2)
    print('Verified nine models, manifests, dimensions and finite inference.', flush=True)
    if args.play:
        os.environ['MICRODUCK_POLICIES'] = str(args.directory.expanduser().resolve() / 'models')
        from sim2sim.play import main as play
        options = ['--robot', str(sim2sim_root() / 'robots/microduck_ball_stand_fix.json')]
        if args.roller:
            options.append('--roller')
        raise SystemExit(play(options))


if __name__ == '__main__':
    main()
