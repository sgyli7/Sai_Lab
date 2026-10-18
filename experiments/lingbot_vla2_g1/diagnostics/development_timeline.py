"""Read-only failure localization after an independently audited development run.

This report does not alter policy actions, physical gates, or benchmark scores.
World/contact truth is used only for post-run explanation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.lingbot_vla2_g1.contract import UPPER_NAMES
from experiments.lingbot_vla2_g1.data import OWN, sha


def intervals(mask):
    spans, start = [], None
    for tick, active in enumerate([*mask, False]):
        if active and start is None:
            start = tick
        if not active and start is not None:
            spans.append([start, tick - 1])
            start = None
    return spans


def analyze(config_path, audit_path):
    config_path, audit_path = map(lambda p: Path(p).absolute(), (config_path, audit_path))
    config, audit = [json.loads(p.read_text()) for p in (config_path, audit_path)]
    folder = Path(config['output_dir'])
    result_path, trace_path = folder / 'result.json', folder / 'trajectory.jsonl'
    result = json.loads(result_path.read_text())
    if (result.get('status') != 'completed' or audit.get('passed') is not True
            or audit.get('autonomous_policy_trial') is not True
            or Path(audit['result_path']).absolute() != result_path.absolute()
            or audit.get('case_id') != result.get('case_id')
            or sha(trace_path) != result['trace_sha256']):
        raise ValueError('An actually completed, independently audited autonomous episode is required')
    for name, digest in audit['inputs_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Audited evidence changed: ' + name)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    outcome = audit['recomputed_acceptance']
    samples = result['acceptance']['samples']
    if any(result['acceptance'].get(key) != value for key, value in outcome.items()):
        raise ValueError('Recorded verdict differs from the independent recomputation')
    if (len(rows) != result['completed_control_ticks'] + 1 or len(samples) != len(rows)
            or [r['control_tick'] for r in rows] != list(range(len(rows)))):
        raise ValueError('Complete physical timeline required')
    first = outcome['first_grasp_control_tick']
    box = np.asarray([r['acceptance_truth_only']['brown_box']['pose'][:3] for r in rows])
    root = np.asarray([r['root_link_pose_w'][:3] for r in rows])
    arm_indices = [rows[0]['source_joint_names'].index(name) for name in UPPER_NAMES[:14]]
    tracking = []
    for row in rows[1:]:
        measured = np.asarray(row['joint_positions'])[arm_indices]
        target = np.asarray(row['processed_upper28_targets'])[:14]
        tracking.append(measured - target)
    boundaries = {'pregrasp': (1, len(rows) if first is None else first + 1),
                  'after_first_grasp': (len(rows), len(rows)) if first is None else (first + 1, len(rows))}
    phases = {}
    for phase, (start, stop) in boundaries.items():
        if stop <= start:
            phases[phase] = None
            continue
        actions = np.asarray([r['applied_action32'] for r in rows[start:stop]])
        phases[phase] = {
            'controls': stop - start,
            'body_xy_command_RMS_m_s': float(np.sqrt(np.mean(actions[:, 29:31] ** 2))),
            'body_yaw_command_RMS_rad_s': float(np.sqrt(np.mean(actions[:, 31] ** 2))),
            'measured_arm_to_servo_target_RMS_rad': float(np.sqrt(np.mean(np.asarray(tracking)[start-1:stop-1] ** 2))),
            'box_XY_path_m': float(np.linalg.norm(np.diff(box[start-1:stop, :2], axis=0), axis=1).sum()),
            'robot_XY_path_m': float(np.linalg.norm(np.diff(root[start-1:stop, :2], axis=0), axis=1).sum()),
        }
    end_pregrasp = len(rows) if first is None else first + 1
    proxy = [s['grasp_contact_lift_proxy'] for s in samples]
    ready = [s['placement_ready'] for s in samples]
    return {
        'schema': 'lingbot_audited_development_timeline_v1', 'status': 'completed',
        'case_id': result['case_id'], 'scope': 'Post-run localization only; same independently recomputed physical verdict.',
        'physical_verdict': {key: outcome[key] for key in (
            'success', 'grasped', 'first_grasp_control_tick', 'carry_qualified_control_tick',
            'max_carried_box_displacement_m', 'carry_turn_degrees', 'stable_placement',
            'fell', 'dropped', 'termination_reasons')},
        'first_grasp_sim_seconds': None if first is None else first / 50,
        'maximum_box_XY_displacement_before_first_grasp_m': float(np.linalg.norm(box[:end_pregrasp, :2]-box[0, :2], axis=1).max()),
        'grasp_contact_lift_proxy_spans_inclusive_ticks': intervals(proxy),
        'placement_ready_spans_inclusive_ticks': intervals(ready),
        'phase_measurements': phases,
        'servo_tracking_error_is_not_model_prediction_error': True,
        'actual_robot_success_claimed': False,
        'inputs_sha256': {str(path): sha(path) for path in (config_path, audit_path, result_path, trace_path)},
        'source_sha256': sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.absolute().is_relative_to(OWN):
        raise FileExistsError('Fresh owned diagnostic report required')
    report = analyze(args.config, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('case_id', 'first_grasp_sim_seconds', 'physical_verdict')}))


if __name__ == '__main__':
    main()
