"""Replay captured physical evidence through the unchanged task acceptance.

This CPU diagnostic never generates actions and never counts as a new trial.
Exit 2 means the required physical outcome is absent; a completed producer alone
cannot make the loop green. Keep output outside the immutable raw case directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.pi05_g1_mobile.acceptance_pose_speed import evaluate_episode


def replay(case):
    case = Path(case).absolute()
    trace = case / 'trajectory.jsonl'
    raw = trace.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = json.loads((case / 'result.json').read_text())
    if digest != result['trace_sha256']:
        raise ValueError('Recorded trace digest differs from producer receipt')
    config = json.loads((case / 'config.json').read_text())
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if result['completed_control_ticks'] != len(rows) - 1:
        raise ValueError('Recorded completion counter differs from trace')
    context = {**config, 'audit_body_names': result['body_names'],
               'audit_contact_metadata': result['contact_view']}
    actual = evaluate_episode(rows, context)
    for key in ('success', 'grasped', 'stable_placement', 'fell', 'dropped',
                'max_carried_box_displacement_m', 'carry_turn_degrees'):
        if actual[key] != result['acceptance'][key]:
            raise ValueError('Independent acceptance differs: ' + key)
    samples = actual.pop('samples')
    return {
        'schema': 'lingbot_actual_task_failure_replay_v1',
        'case': str(case), 'trace_sha256': digest,
        'new_physical_trial': False, 'GPU_used': False,
        'physical_outcome': actual,
        'contact_diagnostics': {
            'left_contact_samples': sum(s['left_hand_contact'] for s in samples),
            'right_contact_samples': sum(s['right_hand_contact'] for s in samples),
            'bilateral_contact_samples': sum(s['left_hand_contact'] and s['right_hand_contact'] for s in samples),
            'held_proxy_samples': sum(s['grasp_contact_lift_proxy'] for s in samples),
            'maximum_box_actor_lift_m': max(r['acceptance_truth_only']['brown_box']['pose'][2] for r in rows)
                - rows[0]['acceptance_truth_only']['brown_box']['pose'][2],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--require', choices=('full', 'grasp'), default='full')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.absolute().is_relative_to(args.case.absolute()):
        raise ValueError('Do not add diagnostics to immutable raw evidence')
    started = time.monotonic()
    report = replay(args.case)
    passed = bool(report['physical_outcome']['success' if args.require == 'full' else 'grasped'])
    report.update(required_outcome=args.require, passed=passed,
                  replay_seconds=time.monotonic() - started)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report, allow_nan=False), flush=True)
    print('REPRO_GREEN' if passed else 'REPRO_RED: required physical task outcome absent', flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
