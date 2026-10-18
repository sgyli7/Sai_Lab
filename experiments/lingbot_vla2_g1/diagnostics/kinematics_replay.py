"""CPU replay of the unchanged rest-velocity guard; no physical-stability verdict."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.lingbot_vla2_g1.data import sha


def stats(value):
    value = np.asarray(value, float)
    return {'minimum': float(value.min()), 'median': float(np.median(value)), 'maximum': float(value.max())}


def pose_span(poses):
    positions = poses[:, :3]
    q = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1)[:, None]
    displacement = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    angle = np.rad2deg(2 * np.arccos(np.clip(np.abs(q @ q.T), 0, 1)))
    return {'all_pairs_position_span_m': float(displacement.max()), 'all_pairs_rotation_span_degrees': float(angle.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-minimal', type=Path)
    parser.add_argument('--assert-velocity-guard', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or (args.export_minimal and args.export_minimal.exists()):
        raise FileExistsError('Preserve diagnostic evidence')
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()][-101:]
    if len(rows) != 101 or any(b['control_tick'] - a['control_tick'] != 1 for a, b in zip(rows, rows[1:])):
        raise ValueError('Two-second inclusive101sample window at50Hz required')
    t = np.asarray([row['sim_time_seconds'] for row in rows], float)
    if not np.allclose(np.diff(t), .02, rtol=0, atol=1e-9):
        raise ValueError('Recorded simulation timeline is not50Hz')
    box = np.asarray([row['acceptance_truth_only']['brown_box']['pose'] for row in rows], float)
    bin_pose = np.asarray([row['acceptance_truth_only']['blue_sorting_bin']['pose'] for row in rows], float)
    velocity = np.asarray([row['acceptance_truth_only']['brown_box']['velocity'] for row in rows], float)
    if box.shape != (101, 7) or bin_pose.shape != (101, 7) or velocity.shape != (101, 6) or not all(np.isfinite(a).all() for a in [box, bin_pose, velocity]):
        raise ValueError('Finite native world pose7/velocity6 required')
    linear = np.linalg.norm(velocity[:, :3], axis=1)
    angular = np.linalg.norm(velocity[:, 3:], axis=1)
    admitted = (linear < .02) & (angular < .1)
    dt = np.diff(t)
    fd = np.diff(box[:, :3], axis=0) / dt[:, None]
    integrated = np.sum((velocity[1:, :3] + velocity[:-1, :3]) * .5 * dt[:, None], axis=0)
    report = {'schema': 'lingbot_vla2_preplaced_kinematics_replay_v1', 'window_control_ticks': [rows[0]['control_tick'], rows[-1]['control_tick']], 'window_sim_seconds': float(t[-1] - t[0]), 'samples': len(rows), 'reported_linear_speed_m_s': stats(linear), 'reported_angular_speed_rad_s': stats(angular), 'reported_vertical_velocity_m_s': stats(velocity[:, 2]), 'box_pose': pose_span(box), 'bin_pose': pose_span(bin_pose), 'pose_finite_difference_speed_m_s': stats(np.linalg.norm(fd, axis=1)), 'pose_finite_difference_vertical_velocity_m_s': stats(fd[:, 2]), 'integrated_reported_linear_displacement_m': integrated.tolist(), 'actual_pose_displacement_m': (box[-1, :3] - box[0, :3]).tolist(), 'velocity_guard_true_samples': int(admitted.sum()), 'two_second_velocity_guard_passed': bool(admitted.all()), 'guard_unchanged': '|linear|<0.02m/s and |angular|<0.1rad/s at all101samples', 'physical_stability_verdict': None, 'scope': 'Recorded50Hz samples only. A pose/velocity discrepancy can arise from sample timing, frame/COM definition, solver motion or aliasing; this replay does not select a cause or modify a threshold.', 'trace_path': str(args.trace.absolute()), 'trace_sha256': sha(args.trace), 'tool_sha256': sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if args.export_minimal:
        with args.export_minimal.open('x') as stream:
            for row in rows:
                stream.write(json.dumps({key: row[key] for key in ['control_tick', 'sim_time_seconds', 'acceptance_truth_only']}) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)
    if args.assert_velocity_guard and not admitted.all():
        print('REPRO_RED: recorded two-second rest velocity guard failed', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
