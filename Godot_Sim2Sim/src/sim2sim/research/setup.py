"""Prepare explicit external inputs for a new, isolated research session."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from sim2sim.paths import load_robot_json, sim2sim_root
from .tasks import TASKS, SESSION


def read_session():
    path = SESSION / 'session.json'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Run python -m sim2sim.research.setup init '
            '--out NEW_DIRECTORY --policies FACTORY_DIRECTORY, then set '
            'SIM2SIM_RESEARCH_DIR to that directory before training.')
    return json.loads(path.read_text())


def initialize(out, policies, hours=8., require_previous=False):
    """Snapshot supplied weights; never reset an existing experiment's budget."""
    import onnx
    out, policies = Path(out).expanduser().resolve(), Path(policies).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Research directory already exists: {out}; choose a new directory')
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('hours must be finite and positive')
    required = {task.factory for task in TASKS.values()}
    previous = {task.previous for task in TASKS.values()}
    if require_previous:
        required |= previous
    missing = sorted(name for name in required if not (policies / name).is_file())
    if missing:
        raise FileNotFoundError('Missing supplied policies: ' + ', '.join(missing))
    names = required | {name for name in previous if (policies / name).is_file()}
    # Validate before creating a session, so a missing/bad input cannot leave a
    # seemingly initialized experiment or start its deadline.
    snapshots = {}
    for name in sorted(names):
        source = policies / name
        model = onnx.load(source)
        onnx.checker.check_model(model)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError(f'{name}: expected one observation and one action tensor')
        for tensor, width in [(model.graph.input[0], 61), (model.graph.output[0], 14)]:
            shape = tensor.type.tensor_type.shape.dim
            if len(shape) != 2 or shape[-1].dim_value != width:
                raise ValueError(f'{name}: expected rank-2 tensor with width {width}')
        # Persist any ONNX external tensor data inside the snapshot itself.
        onnx.external_data_helper.convert_model_from_external_data(model)
        original = source.read_bytes()
        raw = model.SerializeToString()
        if raw == original:
            raw = original
        side = source.with_suffix('.manifest.json')
        side_raw = side.read_bytes() if side.is_file() else None
        if side_raw is not None:
            json.loads(side_raw)
        snapshots[name] = (source, raw, side_raw)
    out.mkdir(parents=True, exist_ok=False)
    baseline = out / 'baseline'
    baseline.mkdir()
    records = {}
    for name, (source, raw, side_raw) in snapshots.items():
        target = baseline / name
        target.write_bytes(raw)
        records[name] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             snapshot_sha256=hashlib.sha256(raw).hexdigest())
        if side_raw is not None:
            target.with_suffix('.manifest.json').write_bytes(side_raw)
    start = time.time()
    record = dict(schema_version=1, start_unix=start, deadline_unix=start + hours * 3600,
                  objective='New research session; independent of the sealed 2026-09-10 experiment',
                  baseline_weights=records, missing_previous=sorted(previous - names))
    (out / 'session.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def prepare_scenes():
    from mjcf2godot.convert import convert
    from sim2sim.godot_proc import godot_bin
    root = sim2sim_root()
    results = []
    for name in ('microduck', 'microduck_ball', 'microduck_ball_stand_fix', 'microduck_roller'):
        cfg = load_robot_json(root / 'robots' / (name + '.json'))
        spec = Path(cfg['godot_spec'])
        if spec.is_file() and spec.with_name('robot.tscn').is_file():
            results.append(dict(robot=name, status='kept_existing'))
            continue
        if spec.parent.exists():
            raise FileExistsError(f'Incomplete generated scene: {spec.parent}; use a clean checkout')
        convert(Path(cfg['mjcf']), spec.parent)
        results.append(dict(robot=name, status='generated'))
    subprocess.run([godot_bin(), '--headless', '--path', str(root / 'godot'),
                    '--editor', '--import', '--quit'], check=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    init = sub.add_parser('init', help='Create a fresh session and snapshot external weights')
    init.add_argument('--out', required=True, type=Path)
    init.add_argument('--policies', required=True, type=Path)
    init.add_argument('--hours', type=float, default=8.)
    init.add_argument('--require-previous', action='store_true', help='Require all nine previous Godot models for A/B')
    sub.add_parser('scenes', help='Generate missing scenes; retain existing generated assets')
    refs = sub.add_parser('references', help='Collect MuJoCo observations required by --variant anchor')
    refs.add_argument('--skill', choices=list(TASKS), required=True)
    refs.add_argument('--seeds', type=int, default=3)
    refs.add_argument('--seed-start', type=int, default=40000)
    refs.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'init':
        result = initialize(args.out, args.policies, args.hours, args.require_previous)
        print(json.dumps(dict(directory=str(args.out.resolve()), **result), indent=2))
    elif args.action == 'scenes':
        print(json.dumps(prepare_scenes(), indent=2))
    else:
        if min(args.seeds, args.workers) < 1:
            parser.error('seed and worker counts must be positive')
        read_session()
        dest = SESSION / 'references' / args.skill
        dest.mkdir(parents=True, exist_ok=False)
        from .evaluate import run_suite
        result = run_suite(args.skill, TASKS[args.skill].source, 'mujoco',
                           range(args.seed_start, args.seed_start + args.seeds), args.workers,
                           dest, entry='both')
        if result['errors']:
            raise RuntimeError(f'Reference collection had errors; retained in {dest}')
        print(dest / 'summary.json')


if __name__ == '__main__':
    main()
