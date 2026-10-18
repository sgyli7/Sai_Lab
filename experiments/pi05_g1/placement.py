"""Acceptance measurement revision v2: sampled, released placement stability.

The original geometry/contact audit remains the basis for containment, support,
robot separation and standing. Its instantaneous velocity thresholds are kept in
the separate legacy diagnostic. This module changes no simulator configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ACCEPTANCE_VERSION = 'pi05_g1_placement_v2'
APPLE_NAME = 'apple_01_objaverse_robolab'
CRITERIA = {
    'control_hz': 50, 'physics_hz': 200, 'physics_steps_per_control': 4,
    'consecutive_samples': 101, 'window_seconds': 2.0,
    'position_reference': 'SDK apple actor/link origin; not mass or geometric centroid',
    'maximum_actor_origin_pairwise_distance_m': 0.005,
    'maximum_quaternion_pairwise_angle_deg': 5.0,
    'quaternion_order': 'xyzw', 'quaternion_sign_invariant': True,
    'geometry_each_sample': ['whole_apple_footprint_inside_plate', 'positive_plate_support',
                             'positive_robot_aabb_separation_and_no_robot_contact', 'standing'],
    'window_start': 'at_or_after_source_success_candidate_and_strictly_after_first_grasp',
    'source_success_candidate_latest_tick': 300,
    'episode_fall_or_drop': 'reject',
    'measurement_scope': '101 consecutive 50Hz physical samples; not proof of continuous-time absolute stillness',
    'legacy_instantaneous_speed_thresholds': 'diagnostic_only_retained_separately',
}
POSE_PROVENANCE = {
    'isaaclab_commit': 'e57379c634b42db5a0fe9f754341be6e2a7c7c43',
    'root_pose_w_alias': {
        'path': 'source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object_data.py',
        'line': 586,
        'sha256': '5c6d3df477d5a8b13845c3470e602ad614310e657c53d378b34a5042f9627800',
        'meaning': 'root_pose_w is shorthand for root_link_pose_w; actor frame origin',
    },
    'physx_root_link_pose_w': {
        'path': 'source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object_data.py',
        'line': 179,
        'sha256': '63df713f2134d37b9a3d74b521e2844f434ec95bc8d1752d7bc58df5794c7cd2',
        'meaning': 'wp.transformf directly from PhysX transforms; orientation (x,y,z,w)',
    },
}


def acceptance_identity():
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    body = {'version': ACCEPTANCE_VERSION, 'criteria': CRITERIA, 'pose_provenance': POSE_PROVENANCE,
            'module_sha256': module_sha}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'acceptance_version': ACCEPTANCE_VERSION, 'acceptance_sha256': sha,
            'module_sha256': module_sha, 'criteria': dict(CRITERIA), 'pose_provenance': POSE_PROVENANCE}


def _maximum_pairwise_position_distance(positions):
    differences = positions[:, None, :] - positions[None, :, :]
    return float(np.sqrt(np.max(np.sum(differences * differences, axis=-1))))


def _maximum_pairwise_quaternion_angle(quaternions):
    # Both q and -q encode the same rotation. Inputs have already been normalized.
    cosine = np.clip(np.abs(quaternions @ quaternions.T), 0., 1.)
    return float(2 * np.arccos(np.min(cosine)))


def evaluate(legacy_audit, rows, *, first_grasp_tick, source_success_candidate_tick,
             fell=False, dropped=False):
    """Assess unchanged physical evidence using the frozen v2 window rule.

    ``legacy_audit`` must be the original source geometry/contact audit of these
    very rows. Never filter rows or reuse its ``ready`` field: that field includes
    the retained legacy velocity thresholds.
    """
    if legacy_audit.get('schema') != 'g1_original_source_conservative_placement_audit_v1':
        raise ValueError('Original source geometry/contact audit is required')
    samples = legacy_audit.get('samples', [])
    if not rows or len(samples) != len(rows):
        raise ValueError('Original audit and physical trace must have identical nonempty coverage')
    if legacy_audit.get('completed_source_controls') != rows[-1]['control_tick']:
        raise ValueError('Geometry audit does not cover the same physical episode')
    positions, quaternions, geometry_valid, reasons = [], [], [], []
    episode_fell, episode_dropped = bool(fell), bool(dropped)
    for index, (sample, row) in enumerate(zip(samples, rows)):
        tick = row['control_tick']
        if tick != index or sample.get('tick') != tick or row['physics_steps_since_reset'] != tick * 4:
            raise ValueError('Physical samples must be consecutive from reset with exactly four integrations per control')
        pose = np.asarray(row['acceptance_truth_only'][APPLE_NAME]['pose'], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('Invalid finite apple pose')
        quaternion = pose[3:7]
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError('Zero or nonnormalizable apple quaternion')
        positions.append(pose[:3])
        quaternions.append(quaternion / norm)
        margin = float(sample['minimum_footprint_margin_m'])
        if not math.isfinite(margin):
            raise ValueError('Nonfinite original footprint evidence')
        failed = []
        if margin < 0:
            failed.append('footprint')
        for key in ('plate_support', 'separation_proved', 'standing'):
            if not isinstance(sample[key], bool):
                raise ValueError(f'Original {key} must be an explicit boolean')
            if not sample[key]:
                failed.append(key)
        geometry_valid.append(not failed)
        reasons.append(failed)
        upright = float(row['upright'])
        root = np.asarray(row['root_link_pose_w'], dtype=np.float64)
        if root.shape != (7,) or not np.isfinite(root).all() or not math.isfinite(upright):
            raise ValueError('Invalid original robot standing/fall evidence')
        episode_fell |= upright < .5 or root[2] + .795 < .35
        episode_dropped |= bool(row.get('original_terms', {}).get('object_dropped', False))
    identity = acceptance_identity()
    result = {'schema': 'pi05_g1_sampled_placement_audit_v2', **identity,
              'stable_placement_passed': False, 'qualified_window': None,
              'first_grasp_tick': first_grasp_tick,
              'source_success_candidate_tick': source_success_candidate_tick,
              'episode_fell': bool(episode_fell), 'episode_dropped': bool(episode_dropped),
              'legacy_speed_audit_passed': bool(legacy_audit['source_diagnostic_placement_passed']),
              'legacy_max_stable_seconds': legacy_audit['max_stable_seconds'],
              'physical_samples': len(rows), 'windows': [],
              'geometry_failures_by_tick': [{'tick': i, 'reasons': x} for i, x in enumerate(reasons) if x]}
    valid_identity = (type(first_grasp_tick) is int and type(source_success_candidate_tick) is int
                      and 0 <= first_grasp_tick < source_success_candidate_tick <= 300)
    if not valid_identity or episode_fell or episode_dropped:
        result['ineligible_reason'] = ('fall_or_drop' if episode_fell or episode_dropped
                                       else 'missing_or_invalid_grasp_before_source_success_within_6s')
        return result
    positions, quaternions = np.asarray(positions), np.asarray(quaternions)
    count = CRITERIA['consecutive_samples']
    start_tick = max(source_success_candidate_tick, first_grasp_tick + 1)
    for start in range(start_tick, len(rows) - count + 1):
        end = start + count - 1
        window = {'start_tick': start, 'end_tick': end, 'duration_seconds': (end - start) / 50.,
                  'geometry_all_samples_passed': bool(all(geometry_valid[start:end + 1])),
                  'passed': False}
        if window['geometry_all_samples_passed']:
            diameter = _maximum_pairwise_position_distance(positions[start:end + 1])
            angle = _maximum_pairwise_quaternion_angle(quaternions[start:end + 1])
            window.update(maximum_actor_origin_pairwise_distance_m=diameter,
                          maximum_quaternion_pairwise_angle_deg=math.degrees(angle))
            window['passed'] = bool(diameter <= CRITERIA['maximum_actor_origin_pairwise_distance_m']
                                     and angle <= math.radians(CRITERIA['maximum_quaternion_pairwise_angle_deg']))
        result['windows'].append(window)
        if window['passed'] and result['qualified_window'] is None:
            result['qualified_window'] = window.copy()
            result['stable_placement_passed'] = True
    return result
