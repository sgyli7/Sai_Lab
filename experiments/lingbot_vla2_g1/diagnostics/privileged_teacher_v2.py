"""Versioned TRAIN-only placement collector with bilateral arm correction.

Actor/link truth is used only by this declared demonstration collector. The
student observation/action contract and every prior collector remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from experiments.lingbot_vla2_g1.contract import UPPER_NAMES
from .privileged_teacher import PlacementTeacher as PreviousTeacher, TraceReader, digest


class PlacementTeacher(PreviousTeacher):
    """Stop the base near the bin, then translate both gripping palms together."""

    def __init__(self, *args, action_bounds=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_bounds is None:
            joints = self.kinematics.joints
            bounds = np.asarray([joints[name]['limits'] for name in UPPER_NAMES])
            self.lower, self.upper_limit = bounds[:, 0], bounds[:, 1]
        else:
            self.lower = np.asarray(action_bounds['lower'][:28], float)
            self.upper_limit = np.asarray(action_bounds['upper'][:28], float)

    def arm_translation(self, row, displacement, max_joint_step=.30):
        displacement = np.asarray(displacement, float)
        norm = np.linalg.norm(displacement)
        displacement *= min(1., .065 / max(norm, 1e-12))
        upper = self.upper.copy()
        predictions = []
        for side, offset in (('left', 0), ('right', 7)):
            jacobian, _ = self.kinematics.jacobian(row, side)
            weighted = jacobian.copy()
            weighted[3:] *= .2
            error = np.r_[displacement, np.zeros(3)]
            change = weighted.T @ np.linalg.solve(
                weighted @ weighted.T + .03 ** 2 * np.eye(6), error)
            # Scale the whole increment to preserve its Cartesian direction.
            change *= min(1., max_joint_step / max(np.max(np.abs(change)), 1e-12))
            # Integrate from the last commanded pose. Starting from measured q
            # would discard the servo offset that maintains the gripping force.
            upper[offset:offset + 7] = np.clip(
                upper[offset:offset + 7] + change,
                self.lower[offset:offset + 7] + 1e-5,
                self.upper_limit[offset:offset + 7] - 1e-5)
            predictions.append((jacobian[:3] @ change).tolist())
        return upper, predictions

    def actions(self, row):
        tick = row['control_tick']
        if tick < len(self.replay):
            return super().actions(row)
        if self.phase == 'replay_TRAIN_prefix':
            self.change_phase('align_box_over_bin', tick)
        center, goal, clearance, margin = self.geometry(row)
        error = goal[:2] - center[:2]
        distance = float(np.linalg.norm(error))
        root_rotation = rotation_xyzw(row['root_link_pose_w'])
        heading = np.arctan2(root_rotation[1, 0], root_rotation[0, 0])
        body_from_world = np.array([[np.cos(heading), np.sin(heading)],
                                    [-np.sin(heading), np.cos(heading)]])
        velocity = np.zeros(3)
        upper = self.upper.copy()
        height_start = self.height
        predictions = None

        if self.phase == 'align_box_over_bin':
            if distance > .16:
                world_velocity = error * .9
                speed = np.linalg.norm(world_velocity)
                world_velocity *= min(1., .30 / max(speed, 1e-12))
                velocity[:2] = body_from_world @ world_velocity
            else:
                self.change_phase('bilateral_arm_alignment', tick)

        if self.phase == 'bilateral_arm_alignment':
            upper, predictions = self.arm_translation(row, np.r_[error, 0.])
            if distance < .028 and margin > .010:
                self.aligned_since = tick if self.aligned_since is None else self.aligned_since
                if tick - self.aligned_since >= 50:
                    self.change_phase('lower_held_box', tick)
            else:
                self.aligned_since = None

        if self.phase == 'lower_held_box':
            # Body lowering and arm lowering share the available vertical range.
            # Horizontal feedback remains active while both hands retain bias.
            self.height = max(.45, self.height - .035)
            vertical = -min(.035, max(0., clearance - .020))
            upper, predictions = self.arm_translation(row, np.r_[error, vertical])
            if clearance <= .025 and margin >= .005:
                self.release_poses = {side: self.kinematics.palm_pose(row, side).copy()
                                      for side in ('left', 'right')}
                self.release_tick = tick
                self.change_phase('open_palms', tick)

        if self.phase in ('open_palms', 'retreat', 'settle'):
            left, right = [self.release_poses[side][:3] for side in ('left', 'right')]
            outward = left - right
            outward[2] = 0
            outward /= max(np.linalg.norm(outward), 1e-6)
            release_age = tick - self.release_tick
            if self.phase == 'open_palms':
                opening = min(.16, .045 * (release_age / 50 + 1))
                for side, sign, offset in (('left', 1, 0), ('right', -1, 7)):
                    pose = self.release_poses[side]
                    upper[offset:offset + 7] = self.kinematics.position_orientation_step(
                        row, side, pose[:3] + sign * opening * outward,
                        rotation_xyzw(pose), max_joint_step=.25)
                if release_age >= 150:
                    self.retreat_upper = upper.copy()
                    self.change_phase('retreat', tick)
            else:
                upper = self.retreat_upper.copy()
            if self.phase == 'retreat':
                velocity[0] = -.12
                if tick - self.entered >= 100:
                    self.change_phase('settle', tick)
                    velocity[:] = 0

        upper = np.clip(upper, self.lower, self.upper_limit)
        self.upper = upper.copy()
        end = np.r_[upper, self.height, velocity].astype(np.float32)
        start = self.last if self.last is not None else end
        fraction = (np.arange(50, dtype=np.float32) + 1) / 50
        chunk = start[None] + fraction[:, None] * (end - start)[None]
        chunk[:, 29:] = end[29:]
        chunk[:, 28] = height_start + fraction * (self.height - height_start)
        self.last = chunk[-1].copy()
        return chunk.astype(np.float32), {
            'phase': self.phase, 'privileged_truth_used': True,
            'recorded_TRAIN_actions_used': False,
            'horizontal_center_error_m': distance,
            'box_floor_clearance_m': clearance,
            'full_footprint_XY_margin_m': margin,
            'height_command_m': self.height,
            'body_command_vx_vy_yaw': velocity.tolist(),
            'bilateral_Jacobian_predicted_translation_m': predictions,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--decoded-evidence', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    declaration = config['privileged_TRAIN_teacher']
    if (declaration.get('TRAIN_only') is not True
            or declaration.get('counts_as_LingBot_success') is not False
            or declaration.get('collector_version') != 2):
        raise ValueError('Explicit TRAIN-only collector v2 required')
    for key in ('source_trace', 'source_result', 'object_geometry', 'robot_geometry'):
        if digest(declaration[key]) != declaration[key + '_sha256']:
            raise ValueError('Frozen collector input changed: ' + key)
    rows = [json.loads(line) for line in Path(declaration['source_trace']).read_text().splitlines()]
    source = json.loads(Path(declaration['source_result']).read_text())
    if source['case']['seed'] != 2026101702 or source['success'] is not False:
        raise ValueError('Reviewed TRAIN source required')
    runtime = json.loads((Path(config['output_dir']) / 'runtime_identity.json').read_text())
    interface = json.loads((Path(config['output_dir']) / 'interface.json').read_text())
    teacher = PlacementTeacher(rows, json.loads(Path(declaration['object_geometry']).read_text()),
                               json.loads(Path(declaration['robot_geometry']).read_text()), runtime['body_names'],
                               action_bounds=interface['actual_action_bounds'])
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
            if (row['source_joint_names'] != request['measured_joint_names43']
                    or row['joint_positions'] != request['measured_joint_positions43']):
                raise ValueError('Privileged row is not the actual measured query tick')
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
