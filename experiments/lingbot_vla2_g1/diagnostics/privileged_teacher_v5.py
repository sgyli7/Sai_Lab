"""TRAIN-only recovery collector after a declared, nonformal student prefix.

The prefix is replayed through the original physics. Only this demonstration
process reads actor/link truth; its successful outcome is never a model win.
The bilateral correction, lowering and release controller is unchanged v2.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .privileged_teacher import TraceReader, digest
from .privileged_teacher_v3 import PlacementTeacher as PreviousTeacher


PREFIX_CONTROLS = 1800
SOURCE_CASE = 'round06_fresh_full60'
SOURCE_SEED = 2026101803


class PlacementTeacher(PreviousTeacher):
    def __init__(self, source_rows, *args, **kwargs):
        super().__init__(source_rows, *args, **kwargs)
        self.replay = np.asarray([r['applied_action32'] for r in source_rows[1:PREFIX_CONTROLS + 1]], np.float32)
        if self.replay.shape != (PREFIX_CONTROLS, 32) or not np.isfinite(self.replay).all():
            raise ValueError('Complete1800control R6 physical prefix required')
        self.upper = self.replay[-1, :28].copy()
        self.height = float(self.replay[-1, 28])
        self.recovery_ready_since = None

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
            or declaration.get('collector_version') != 5
            or declaration.get('initial_replay_controls') != PREFIX_CONTROLS
            or declaration.get('former_development_scene_now_TRAIN') is not True
            or declaration.get('fresh_development_cohort_required') is not True):
        raise ValueError('Explicit reclassified TRAIN recovery declaration required')
    if declaration.get('lowering_schedule') != {
            'center_error_below_m': .045, 'footprint_margin_above_m': .015,
            'held_ready_controls': 50, 'continues_bilateral_XY_correction_while_lowering': True}:
        raise ValueError('Exact TRAIN demonstration scheduling declaration required')
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
