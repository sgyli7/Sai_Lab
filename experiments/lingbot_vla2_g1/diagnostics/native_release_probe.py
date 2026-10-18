"""CPU geometry and command-tracking facts for a failed native TRAIN collection.

No actions or thresholds are changed. These facts distinguish failure stages;
they do not identify why the neural policy chose a command.
"""
import json
from pathlib import Path
import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from experiments.lingbot_vla2_g1.data import OWN, sha


def main():
    case = OWN / 'resumed_turn01/episodes/resume01_native_teacher01'
    config = json.loads((case / 'config.json').read_text())
    result = json.loads((case / 'result.json').read_text())
    rows = [json.loads(line) for line in (case / 'trajectory.jsonl').read_text().splitlines() if line.strip()]
    assert sha(case / 'trajectory.jsonl') == result['trace_sha256']
    assert sha(config['geometry_definition']) == config['geometry_definition_sha256']
    definition = json.loads(Path(config['geometry_definition']).read_text())
    objects = {v['kind']: v for v in definition['objects']}
    corners = np.concatenate([v['points'] for v in objects['t2_box']['convex_parts']])
    floor = np.asarray(objects['t2_bin']['convex_parts'][2]['points'])
    box_local_center = (corners.min(0) + corners.max(0)) / 2
    floor_local_center = (floor.min(0) + floor.max(0)) / 2
    facts = []
    for row, sample in zip(rows, result['acceptance']['samples']):
        bp, tp = (np.asarray(row['acceptance_truth_only'][key]['pose']) for key in ('brown_box', 'blue_sorting_bin'))
        rb, rt = rotation_xyzw(bp), rotation_xyzw(tp)
        box_center = box_local_center @ rb.T + bp[:3]
        bin_floor_center = floor_local_center @ rt.T + tp[:3]
        clearance = float((corners @ rb.T + bp[:3])[:, 2].min() - (floor @ rt.T + tp[:3])[:, 2].max())
        facts.append({'tick': row['control_tick'], 'box_mesh_center_world': box_center.tolist(),
                      'bin_floor_center_world': bin_floor_center.tolist(),
                      'horizontal_center_distance_m': float(np.linalg.norm(box_center[:2] - bin_floor_center[:2])),
                      'box_bottom_minus_bin_floor_m': clearance, **sample})
    closest = min(facts, key=lambda v: v['horizontal_center_distance_m'])
    best_margin = max(facts, key=lambda v: v['bin_floor_margin_m'])
    root = np.asarray([r['root_link_pose_w'] for r in rows])
    yaw = np.unwrap([np.arctan2(rotation_xyzw(p)[1, 0], rotation_xyzw(p)[0, 0]) for p in root])
    actual, commanded, tracking = [], [], []
    for start in range(0, len(rows) - 50, 50):
        stop = start + 50
        mid = (yaw[start] + yaw[stop]) / 2
        world_v = root[stop, :2] - root[start, :2]  # Exactly1s.
        body_v = np.array([[np.cos(mid), np.sin(mid)], [-np.sin(mid), np.cos(mid)]]) @ world_v
        command = np.mean([r['applied_action32'][29:32] for r in rows[start + 1:stop + 1]], axis=0)
        measured = np.r_[body_v, yaw[stop] - yaw[start]]
        commanded.append(command); actual.append(measured)
        tracking.append({'start_tick': start, 'end_tick': stop, 'mean_body_command_vx_vy_yaw': command.tolist(),
                         'pose_difference_body_velocity_vx_vy_yaw': measured.tolist(),
                         'box_center_target_distance_at_end_m': facts[stop]['horizontal_center_distance_m']})
    commanded, actual = np.asarray(commanded), np.asarray(actual)
    correlations = [float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else None
                    for a, b in zip(commanded.T, actual.T)]
    late = facts[2000:]
    report = {'schema': 'lingbot_fresh_native_release_stage_facts_v1', 'GPU_used': False,
              'trace_sha256': result['trace_sha256'], 'full_task_success': result['success'],
              'maximum_full_footprint_bin_margin': best_margin,
              'closest_mesh_center_to_bin_floor_center': closest,
              'full_footprint_XY_inside_samples': sum(f['bin_floor_margin_m'] >= 0 for f in facts),
              'box_inside_bin_prism_samples': sum(f['box_inside_bin_prism'] for f in facts),
              'bin_upward_support_samples': sum(f['bin_upward_force_n'] > 0 for f in facts),
              'placement_ready_samples': sum(f['placement_ready'] for f in facts),
              'late20s_min_horizontal_center_distance_m': min(f['horizontal_center_distance_m'] for f in late),
              'late20s_contact_counts': {'left': sum(f['left_hand_contact'] for f in late), 'right': sum(f['right_hand_contact'] for f in late)},
              '1s_body_command_vs_pose_velocity_correlations_vx_vy_yaw': correlations,
              '1s_tracking_RMSE_vx_vy_yaw': np.sqrt(np.mean((commanded - actual) ** 2, axis=0)).tolist(),
              'last10s_mean_body_command_vx_vy_yaw': commanded[-10:].mean(0).tolist(),
              'last10s_mean_pose_body_velocity_vx_vy_yaw': actual[-10:].mean(0).tolist(),
              'tracking_1s_windows': tracking,
              'scope': 'Exact collision geometry and sampled rigid/root poses. Tracking correlations do not by themselves identify policy failure or controller causality; no command/contract changes.'}
    path = OWN / 'diagnostics/fresh_native_release_stage_facts.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('tracking_1s_windows',)}), flush=True)


if __name__ == '__main__':
    main()
