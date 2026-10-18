"""TRAIN-only recovery collector after a declared, nonformal student prefix.

The prefix is replayed through the original physics. Only this demonstration
process reads actor/link truth; its successful outcome is never a model win.
Bilateral corrections stay unchanged. Once the box reaches the floor, body height is held and the box is centered before opening; all physical acceptance remains unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from .privileged_teacher import TraceReader, digest
from .privileged_teacher_v3 import PlacementTeacher as PreviousTeacher


PREFIX_CONTROLS = 1800
SOURCE_CASE = 'round06_fresh_full60'
SOURCE_SEED = 2026101803


class GuardedFloorRelease(PreviousTeacher):
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
            floor_reached = clearance <= .025
            # Finish body descent before opening. Keep the original target
            # height while gently centering the supported box on the floor.
            if not floor_reached:
                self.height = max(.45, self.height - .035)
            vertical = 0. if floor_reached else -min(.035, max(0., clearance - .020))
            upper, predictions = self.arm_translation(row, np.r_[error, vertical])
            safe = floor_reached and margin >= .025 and distance <= .040
            if safe:
                if self.floor_release_ready_since is None:
                    self.floor_release_ready_since = tick
            else:
                self.floor_release_ready_since = None
            if safe and tick - self.floor_release_ready_since >= 50:
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
            'floor_centering_pending': self.phase == 'lower_held_box' and clearance <= .025,
            'release_minimum_margin_m': .025,
            'release_maximum_center_error_m': .040,
            'body_command_vx_vy_yaw': velocity.tolist(),
            'bilateral_Jacobian_predicted_translation_m': predictions,
        }



class PlacementTeacher(GuardedFloorRelease):
    def __init__(self, source_rows, *args, **kwargs):
        super().__init__(source_rows, *args, **kwargs)
        self.replay = np.asarray([r['applied_action32'] for r in source_rows[1:PREFIX_CONTROLS + 1]], np.float32)
        if self.replay.shape != (PREFIX_CONTROLS, 32) or not np.isfinite(self.replay).all():
            raise ValueError('Complete1800control R6 physical prefix required')
        self.upper = self.replay[-1, :28].copy()
        self.height = float(self.replay[-1, 28])
        self.recovery_ready_since = None
        self.floor_release_ready_since = None

    def actions(self, row):
        tick = row['control_tick']
        if tick >= len(self.replay) and self.phase in (
                'replay_TRAIN_prefix', 'align_box_over_bin', 'bilateral_arm_alignment'):
            center, goal, _, margin = self.geometry(row)
            ready = np.linalg.norm(goal[:2]-center[:2]) < .045 and margin > .015
            if ready:
                if self.recovery_ready_since is None:
                    self.recovery_ready_since = tick
                if tick-self.recovery_ready_since >= 50:
                    # Both palms continue horizontal correction during lowering.
                    # This changes the demonstration controller's scheduling,
                    # never any physical acceptance or release requirement.
                    self.change_phase('lower_held_box', tick)
            else:
                self.recovery_ready_since = None
        return super().actions(row)


def validate_source(declaration):
    if (declaration.get('TRAIN_only') is not True
            or declaration.get('counts_as_LingBot_success') is not False
            or declaration.get('collector_version') != 6
            or declaration.get('initial_replay_controls') != PREFIX_CONTROLS
            or declaration.get('former_development_scene_now_TRAIN') is not True
            or declaration.get('fresh_development_cohort_required') is not True):
        raise ValueError('Explicit reclassified TRAIN recovery declaration required')
    if declaration.get('lowering_schedule') != {
            'center_error_below_m': .045, 'footprint_margin_above_m': .015,
            'held_ready_controls': 50, 'continues_bilateral_XY_correction_while_lowering': True}:
        raise ValueError('Exact TRAIN demonstration scheduling declaration required')
    if declaration.get('floor_release_schedule') != {'freeze_body_height_at_floor_clearance_m': .025, 'minimum_footprint_margin_m': .025, 'maximum_center_error_m': .040, 'held_ready_controls': 50}:
        raise ValueError('Exact declared floor-before-opening schedule required')
    for key in ('source_trace', 'source_result', 'object_geometry', 'robot_geometry', 'reclassification'):
        if digest(declaration[key]) != declaration[key + '_sha256']:
            raise ValueError('Frozen recovery collector input changed: ' + key)
    result = json.loads(Path(declaration['source_result']).read_text())
    if (result['case_id'] != SOURCE_CASE or result['case']['seed'] != SOURCE_SEED
            or result['success'] is not False or result['completed_control_ticks'] != 3000
            or result['acceptance']['fell'] or result['acceptance']['dropped']
            or result['acceptance']['max_carried_box_displacement_m'] < 4
            or result['trace_sha256'] != digest(declaration['source_trace'])):
        raise ValueError('Only the reviewed nonformal long-carry failure is admitted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--decoded-evidence', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    declaration = config['privileged_TRAIN_teacher']
    validate_source(declaration)
    rows = [json.loads(line) for line in Path(declaration['source_trace']).read_text().splitlines()]
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
            if row['source_joint_names'] != request['measured_joint_names43'] or row['joint_positions'] != request['measured_joint_positions43']:
                raise ValueError('Trace is not the actual current measured tick')
            chunk, diagnostics = teacher.actions(row)
            if chunk.shape != (50, 32) or not np.isfinite(chunk).all():
                raise ValueError('Finite original effective32 commands required')
            elapsed = time.monotonic() - started
            record({'event': 'privileged_TRAIN_action', 'control_tick': request['control_tick'],
                    'actions32': chunk.tolist(), **diagnostics})
            print(json.dumps({'episode_id': request['episode_id'], 'control_tick': request['control_tick'],
                              'actions': chunk.tolist(), 'inference_seconds': elapsed,
                              'stage_seconds': {'scripted_CPU_collector': elapsed},
                              'autonomous_model_policy': False}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
