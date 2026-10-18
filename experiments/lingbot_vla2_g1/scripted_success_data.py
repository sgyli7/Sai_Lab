"""Admit physically successful privileged TRAIN demonstrations, never VLA wins.

Keep this schema separate from autonomous native teacher data. Actor/link truth
is bound as external audit evidence only; model samples contain RGB/q/prompt and
the next applied32 action, using unchanged original80TRAIN mean/std statistics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .contract import PROMPT, encode_state, decode_to_source
from .data import OWN, SHARED, SPLIT_SHA, sha
from .success_data import write

SCHEMA = 'lingbot_privileged_physical_full_success_TRAIN_episode_v1'


def require_success(audit, result, config, cohort):
    canonical = {k: v for k, v in audit.items() if k != 'audit_sha256'}
    if audit.get('audit_sha256') != hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest():
        raise ValueError('Independent collector audit digest differs')
    if (audit.get('schema') != 'lingbot_privileged_TRAIN_teacher_CPU_audit_v1'
            or audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not False
            or audit.get('counts_as_LingBot_success') is not False or not audit.get('checks')
            or any(c.get('passed') is not True for c in audit['checks'].values())):
        raise ValueError('Every independent collector evidence check must pass with truthful attribution')
    outcome = audit.get('recomputed_acceptance') or {}
    if (result.get('status') != 'completed' or result.get('success') is not True
            or outcome.get('success') is not True or outcome.get('stable_placement') is not True
            or outcome.get('fell') or outcome.get('dropped')
            or outcome.get('autonomous_model_policy') is not False
            or outcome.get('counts_as_LingBot_success') is not False
            or outcome.get('policy_external_interventions', 0) != result['completed_control_ticks']):
        raise ValueError('Actual complete physical transport and stable release are mandatory')
    if (config.get('policy_id') != 'privileged_scripted_TRAIN_teacher'
            or result.get('policy') != config['policy_id'] or config.get('save_full_rgb') is not True
            or config.get('prompt') != PROMPT
            or cohort.get('schema') != 'lingbot_privileged_TRAIN_collector_cohort_v1'
            or config.get('privileged_TRAIN_teacher') != cohort.get('declaration')
            or config['scene'] != cohort.get('scene') or config['case']['seed'] != cohort.get('seed')
            or cohort.get('only_after_independent_complete_physical_success_may_admit_full_demonstration') is not True):
        raise ValueError('Prospective TRAIN collection identity differs')
    return outcome


def prepare(case, audit_path, cohort_path, output):
    case, audit_path, cohort_path, output = [Path(p).absolute() for p in (case, audit_path, cohort_path, output)]
    if output.exists() or not case.is_relative_to(OWN) or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned TRAIN preparation required')
    audit, cohort = [json.loads(p.read_text()) for p in (audit_path, cohort_path)]
    result, config = [json.loads((case / name).read_text()) for name in ('result.json', 'config.json')]
    outcome = require_success(audit, result, config, cohort)
    if Path(audit['result_path']) != case / 'result.json':
        raise ValueError('Collector audit belongs to another case')
    formal = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    if (config['case']['seed'] in {c['seed'] for c in formal['cases']}
            or any(np.linalg.norm(np.asarray(config['scene']['box_pose']['position_xyz'])[:2]
                                  - np.asarray(c['scene']['box_pose']['position_xyz'])[:2]) < 1e-6
                   for c in formal['cases'])):
        raise ValueError('Formal seeds/physical positions cannot become TRAIN data')
    frozen = {**audit['inputs_sha256'], str(audit_path): sha(audit_path), str(cohort_path): sha(cohort_path)}
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Audited collector evidence changed: ' + name)
    rows = [json.loads(line) for line in (case / 'trajectory.jsonl').read_text().splitlines()]
    length = len(rows) - 1
    if length != result['completed_control_ticks'] or result['actual_sdk_integrations'] != 4 * length:
        raise ValueError('Actual control/physics coverage differs')
    if sha(case / 'trajectory.jsonl') != result['trace_sha256']:
        raise ValueError('Actual raw trace changed')
    states, actions, joints, frames = [], [], [], []
    names = result['source_joint_names']
    for tick, row in enumerate(rows[:-1]):
        q = np.asarray(row['joint_positions'], np.float32)
        action = np.asarray(rows[tick + 1]['applied_action32'], np.float32)
        if (row['control_tick'] != tick or row['physics_steps_since_reset'] != 4 * tick
                or row['source_joint_names'] != names or q.shape != (43,) or action.shape != (32,)
                or not np.isfinite(action).all()
                or not np.array_equal(decode_to_source(action, q, names), np.asarray(rows[tick + 1]['applied_source_action50'], np.float32))):
            raise ValueError('RGB/current measured state/next applied target alignment differs')
        image = case / 'head_rgb' / f'{tick:05}.png'
        frame = {'path': str(image), 'byte_sha256': sha(image), 'rgb_sha256': row['head_rgb_sha256'],
                 'control_tick': tick, 'timestamp_seconds': tick / 50}
        raw_png(frame)
        frozen[str(image)] = frame['byte_sha256']
        states.append(encode_state(q, names)); actions.append(action); joints.append(q); frames.append(frame)
    boundary = outcome['first_grasp_control_tick'] + 100
    if not 0 < boundary < 1500 < length:
        raise ValueError('Actual grasp/transport/placement supervision phases required')
    output.mkdir(parents=True)
    with (output / 'episode.npz').open('xb') as stream:
        np.savez(stream, state=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32),
                 measured_joint_positions43=np.asarray(joints, np.float32), timestamps=np.arange(length) / 50)
    frozen[str(output / 'episode.npz')] = sha(output / 'episode.npz')
    manifest = {'schema': SCHEMA, 'status': 'completed', 'train_only': True, 'validation': [], 'test': [],
                'task_success_teacher': True, 'source_autonomous_model_policy': False,
                'counts_as_LingBot_success': False, 'teacher_type': 'explicit_privileged_scripted_physical_collector',
                'source_case_id': result['case_id'], 'seed': config['case']['seed'], 'length': length,
                'source_joint_names': names, 'raw_rgb_frames': frames, 'npz': 'episode.npz',
                'npz_sha256': sha(output / 'episode.npz'), 'independent_audit': str(audit_path),
                'independent_audit_sha256': sha(audit_path), 'physical_outcome': outcome,
                'extra_frozen_files': frozen, 'source_split_sha256': SPLIT_SHA,
                'LingBot_original80train_norm_sha256': sha(OWN / 'train_meanstd.json'),
                'source_test_payloads_opened': False, 'formal_trajectories_used_for_training': False,
                'input_ticks': [0, length - 1], 'target_applied_rows': [1, length],
                'phase_boundaries': {'approach_grasp': [0, boundary], 'transport': [boundary, 1500], 'placement': [1500, length]},
                'normalization_policy': 'Original80TRAIN statistics unchanged; teacher excluded from statistic fitting.'}
    write(output / 'manifest.json', manifest)
    return manifest


class FullScriptedSuccess:
    def __init__(self, manifest, normalizer):
        path = Path(manifest).absolute()
        data = json.loads(path.read_text())
        if (not path.is_relative_to(OWN) or data.get('schema') != SCHEMA
                or data.get('task_success_teacher') is not True or data.get('train_only') is not True
                or data.get('source_autonomous_model_policy') is not False
                or data.get('counts_as_LingBot_success') is not False or data.get('validation') or data.get('test')
                or data.get('source_split_sha256') != SPLIT_SHA
                or data.get('LingBot_original80train_norm_sha256') != sha(OWN / 'train_meanstd.json')):
            raise ValueError('Explicit audited full physical TRAIN collector data required')
        audit = json.loads(Path(data['independent_audit']).read_text())
        if (sha(data['independent_audit']) != data['independent_audit_sha256'] or audit.get('passed') is not True
                or audit.get('autonomous_policy_trial') is not False or audit.get('counts_as_LingBot_success') is not False
                or audit.get('recomputed_acceptance') != data['physical_outcome']
                or data['physical_outcome'].get('success') is not True):
            raise ValueError('Actual full physical success audit differs')
        for name, expected in data['extra_frozen_files'].items():
            if sha(name) != expected:
                raise ValueError('Admitted full demonstration changed: ' + name)
        with np.load(path.parent / data['npz'], allow_pickle=False) as arrays:
            self.states = arrays['state'].copy(); actions = arrays['actions'].copy()
            q = arrays['measured_joint_positions43'].copy(); times = arrays['timestamps'].copy()
        length = data['length']
        if self.states.shape != (length, 32) or actions.shape != (length, 32) or q.shape != (length, 43):
            raise ValueError('Teacher tensor shapes differ')
        if not np.array_equal(self.states, np.asarray([encode_state(row, data['source_joint_names']) for row in q])) or not np.array_equal(times, np.arange(length) / 50):
            raise ValueError('Measured-state/time tensors differ')
        self.frames = data['raw_rgb_frames']
        if len(self.frames) != length:
            raise ValueError('Every teacher input requires synchronized raw RGB')
        for tick, frame in enumerate(self.frames):
            if frame['control_tick'] != tick or frame['timestamp_seconds'] != tick / 50:
                raise ValueError('Teacher image time alignment differs')
            raw_png(frame)
        self.actions = normalizer.normalize(actions, 'actions')
        self.indices = range(length)
        self.episodes = [{'id': data['source_case_id']}]
        self.boundaries = data['phase_boundaries']
        self.phase_draw_counts = {'approach_grasp': 0, 'transport': 0, 'placement': 0}

    def sample(self, index):
        value = random.random()
        phase = 'approach_grasp' if value < .45 else 'transport' if value < .65 else 'placement'
        start, stop_phase = self.boundaries[phase]
        index = random.randrange(start, stop_phase)
        self.phase_draw_counts[phase] += 1
        stop = min(index + 50, len(self.actions)); count = stop - index
        targets = np.repeat(self.actions[stop - 1:stop], 50, axis=0)
        targets[:count] = self.actions[index:stop]
        return raw_png(self.frames[index]), self.states[index], targets, np.arange(50) < count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'audit', 'cohort', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.case, args.audit, args.cohort, args.output)
    print(json.dumps({'status': result['status'], 'frames': result['length'],
                      'full_physical_success_teacher': True, 'LingBot_success_claim': False}))


if __name__ == '__main__':
    main()
