"""TRAIN-only physical collector: recorded TRAIN approach, truth-guided placement.

This is explicitly a privileged scripted demonstrator, never LingBot inference
or a benchmark policy. It reads the current raw trace on purpose. The simulator,
Homie controller, physical dynamics and success predicates remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from experiments.lingbot_vla2_g1.contract import UPPER_NAMES
from .training_teacher_kinematics import MeasuredJacobian


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PlacementTeacher:
    """Truth is restricted to this labelled TRAIN demonstrator process."""
    def __init__(self, source_rows, objects, robot_geometry, body_names):
        self.replay = np.asarray([r['applied_action32'] for r in source_rows[1:1463]], np.float32)
        if self.replay.shape != (1462, 32) or not np.isfinite(self.replay).all():
            raise ValueError('Complete independently admitted TRAIN carry prefix required')
        definitions = {x['kind']: x for x in objects['objects']}
        self.corners = np.concatenate([p['points'] for p in definitions['t2_box']['convex_parts']])
        self.floor = np.asarray(definitions['t2_bin']['convex_parts'][2]['points'])
        self.box_center = (self.corners.min(0) + self.corners.max(0)) / 2
        self.floor_center = (self.floor.min(0) + self.floor.max(0)) / 2
        self.kinematics = MeasuredJacobian(robot_geometry, body_names)
        self.phase = 'replay_TRAIN_prefix'
        self.entered = 0
        self.aligned_since = None
        self.upper = self.replay[-1, :28].copy()
        self.height = .75
        self.release_poses = None
        self.last = None

    def geometry(self, row):
        box, target = [np.asarray(row['acceptance_truth_only'][key]['pose'])
                       for key in ('brown_box', 'blue_sorting_bin')]
        rb, rt = rotation_xyzw(box), rotation_xyzw(target)
        corners = self.corners @ rb.T + box[:3]
        floor = self.floor @ rt.T + target[:3]
        center = self.box_center @ rb.T + box[:3]
        goal = self.floor_center @ rt.T + target[:3]
        in_target = (corners - target[:3]) @ rt
        margin = float(min((in_target[:, :2].min(0) - self.floor[:, :2].min(0)).min(),
                           (self.floor[:, :2].max(0) - in_target[:, :2].max(0)).min()))
        return center, goal, float(corners[:, 2].min() - floor[:, 2].max()), margin

    def change_phase(self, phase, tick):
        self.phase, self.entered = phase, tick

    def actions(self, row):
        tick = row['control_tick']
        if tick < len(self.replay):
            # Chunks crossing the boundary hold their last valid TRAIN command;
            # placement starts at the next native50 query, tick1500.
            valid = self.replay[tick:min(tick + 50, len(self.replay))]
            chunk = np.concatenate([valid, np.repeat(valid[-1:], 50 - len(valid), axis=0)])
            if tick + 50 > len(self.replay):
                chunk[len(valid):, 29:] = 0
            self.last = chunk[-1].copy()
            return chunk, {'phase': self.phase, 'privileged_truth_used': False,
                           'recorded_TRAIN_actions_used': True}

        if self.phase == 'replay_TRAIN_prefix':
            self.change_phase('align_box_over_bin', tick)
        center, goal, clearance, margin = self.geometry(row)
        error = goal[:2] - center[:2]
        distance = float(np.linalg.norm(error))
        root_rotation = rotation_xyzw(row['root_link_pose_w'])
        heading = np.arctan2(root_rotation[1, 0], root_rotation[0, 0])
        body_from_world = np.array([[np.cos(heading), np.sin(heading)],
                                    [-np.sin(heading), np.cos(heading)]])
        upper = self.upper.copy()
        velocity = np.zeros(3)
        height_start = self.height

        if self.phase in ('align_box_over_bin', 'lower_held_box'):
            world_velocity = error * (.65 if self.phase == 'align_box_over_bin' else .4)
            speed_limit = .30 if self.phase == 'align_box_over_bin' else .07
            norm = np.linalg.norm(world_velocity)
            if norm > speed_limit:
                world_velocity *= speed_limit / norm
            velocity[:2] = body_from_world @ world_velocity
            if self.phase == 'align_box_over_bin':
                if distance < .035 and margin > .015:
                    self.aligned_since = tick if self.aligned_since is None else self.aligned_since
                    if tick - self.aligned_since >= 50:
                        self.change_phase('lower_held_box', tick)
                else:
                    self.aligned_since = None
            if self.phase == 'lower_held_box':
                self.height = max(.45, self.height - .045)
                if (clearance <= .045 and margin >= .005) or (self.height <= .450001 and tick - self.entered >= 400):
                    self.release_poses = {side: self.kinematics.palm_pose(row, side).copy()
                                          for side in ('left', 'right')}
                    self.change_phase('open_palms', tick)

        if self.phase in ('open_palms', 'retreat', 'settle'):
            velocity[:] = 0
            left, right = [self.release_poses[side][:3] for side in ('left', 'right')]
            outward = left - right
            outward[2] = 0
            outward /= max(np.linalg.norm(outward), 1e-6)
            release_age = max(0, tick - self.release_tick) if hasattr(self, 'release_tick') else 0
            if not hasattr(self, 'release_tick'):
                self.release_tick = tick
            opening = min(.14, .035 * (release_age / 50 + 1))
            if self.phase == 'open_palms':
                for side, sign, offset in (('left', 1, 0), ('right', -1, 7)):
                    pose = self.release_poses[side]
                    desired = pose[:3] + sign * opening * outward
                    target = self.kinematics.position_orientation_step(
                        row, side, desired, rotation_xyzw(pose), max_joint_step=.18)
                    upper[offset:offset + 7] = target
            else:
                upper = self.retreat_upper.copy()
            if self.phase == 'open_palms' and release_age >= 200:
                self.retreat_upper = upper.copy()
                self.change_phase('retreat', tick)
            if self.phase == 'retreat':
                velocity[0] = -.12
                if tick - self.entered >= 150:
                    self.change_phase('settle', tick)
                    velocity[:] = 0

        end = np.r_[upper, self.height, velocity].astype(np.float32)
        start = self.last if self.last is not None else end
        fraction = (np.arange(50, dtype=np.float32) + 1) / 50
        chunk = start[None] + fraction[:, None] * (end - start)[None]
        # Locomotion is feedback command, rather than a full second of delayed ramp.
        chunk[:, 29:] = end[29:]
        chunk[:, 28] = height_start + fraction * (self.height - height_start)
        self.last = chunk[-1].copy()
        return chunk.astype(np.float32), {
            'phase': self.phase, 'privileged_truth_used': True,
            'recorded_TRAIN_actions_used': False,
            'horizontal_center_error_m': distance, 'box_floor_clearance_m': clearance,
            'full_footprint_XY_margin_m': margin, 'height_command_m': self.height,
            'body_command_vx_vy_yaw': velocity.tolist(),
        }


class TraceReader:
    def __init__(self, path):
        self.stream = Path(path).open()
        self.latest = None

    def at(self, tick):
        for line in self.stream:
            self.latest = json.loads(line)
        if self.latest is None or self.latest['control_tick'] != tick:
            raise ValueError('Privileged trace must be flushed at exact paused query tick')
        return self.latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--decoded-evidence', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    declaration = config['privileged_TRAIN_teacher']
    if declaration.get('TRAIN_only') is not True or declaration.get('counts_as_LingBot_success') is not False:
        raise ValueError('Explicit privileged TRAIN-only non-LingBot collector required')
    for key in ('source_trace', 'source_result', 'object_geometry', 'robot_geometry'):
        if digest(declaration[key]) != declaration[key + '_sha256']:
            raise ValueError('TRAIN collector input changed: ' + key)
    rows = [json.loads(line) for line in Path(declaration['source_trace']).read_text().splitlines()]
    source = json.loads(Path(declaration['source_result']).read_text())
    if source['case']['seed'] != 2026101702 or source['success'] is not False:
        raise ValueError('Reviewed TRAIN-only source required, not a formal/test trajectory')
    runtime = json.loads((Path(config['output_dir']) / 'runtime_identity.json').read_text())
    teacher = PlacementTeacher(rows, json.loads(Path(declaration['object_geometry']).read_text()),
                               json.loads(Path(declaration['robot_geometry']).read_text()), runtime['body_names'])
    reader = TraceReader(Path(config['output_dir']) / 'trajectory.jsonl')
    with args.decoded_evidence.open('x', buffering=1) as evidence:
        def record(value):
            evidence.write(json.dumps(value, allow_nan=False) + '\n')
        record({'event': 'privileged_TRAIN_teacher_loaded', 'teacher_sha256': digest(__file__),
                'TRAIN_only': True, 'autonomous_model_policy': False,
                'counts_as_LingBot_success': False, 'declaration': declaration})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('episode_id') != config['case']['case_id'] or request.get('prompt') != config['prompt']:
                raise ValueError('Collector request identity differs')
            started = time.monotonic()
            row = reader.at(request['control_tick'])
            if row['source_joint_names'] != request['measured_joint_names43'] or row['joint_positions'] != request['measured_joint_positions43']:
                raise ValueError('Current privileged trace is not the actual measured policy tick')
            chunk, diagnostics = teacher.actions(row)
            if chunk.shape != (50, 32) or not np.isfinite(chunk).all():
                raise ValueError('Finite source action contract required')
            elapsed = time.monotonic() - started
            record({'event': 'privileged_TRAIN_action', 'control_tick': request['control_tick'],
                    'actions32': chunk.tolist(), **diagnostics})
            print(json.dumps({'episode_id': request['episode_id'], 'control_tick': request['control_tick'],
                              'actions': chunk.tolist(), 'inference_seconds': elapsed,
                              'stage_seconds': {'scripted_CPU_collector': elapsed},
                              'autonomous_model_policy': False}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
