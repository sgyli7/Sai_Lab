"""Explicit admission of fresh, independently successful autonomous TRAIN demos.

Never reads the original dataset test payloads or formal-test trajectories.
Supervision is RGB/q at tick t -> the action actually applied at row t+1.
Truth is retained only in external audit evidence, never model input tensors.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .contract import PROMPT, encode_state, decode_to_source
from .data import OWN, SHARED, SPLIT, SPLIT_SHA, sha


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(case, audit_path, declaration_path, output):
    case, output = Path(case).absolute(), Path(output).absolute()
    if output.exists() or not case.is_relative_to(OWN) or not output.is_relative_to(OWN):
        raise ValueError('Fresh LingBot-owned TRAIN admission required')
    audit_path, declaration_path = Path(audit_path).absolute(), Path(declaration_path).absolute()
    audit, declaration = (json.loads(p.read_text()) for p in (audit_path, declaration_path))
    result = json.loads((case / 'result.json').read_text())
    config = json.loads((case / 'config.json').read_text())
    case_id = result['case_id']
    if declaration.get('schema') != 'lingbot_fresh_native_TRAIN_declaration_v1' or case_id not in declaration['train_case_ids']:
        raise ValueError('Prospective TRAIN declaration required')
    if (audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True
            or Path(audit['result_path']).absolute() != case / 'result.json'
            or not audit.get('checks') or any(v.get('passed') is not True for v in audit['checks'].values())):
        raise ValueError('Every independent physical/evidence audit check must pass')
    outcome = audit.get('recomputed_acceptance', {})
    if (result.get('status') != 'completed' or result.get('success') is not True
            or outcome.get('success') is not True or outcome.get('stable_placement') is not True
            or outcome.get('fell') or outcome.get('dropped')
            or config.get('policy_id') != 'gr00t_n16' or result.get('policy') != 'gr00t_n16'
            or config.get('mode') != 'policy' or config.get('save_full_rgb') is not True
            or config.get('prompt') != PROMPT):
        raise ValueError('An actual complete autonomous native success with full raw RGB is required')
    formal = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    if (config['case']['seed'] in {c['seed'] for c in formal['cases']}
            or case_id in {c['case_id'] for c in formal['cases']}
            or any(np.linalg.norm(np.asarray(config['scene']['box_pose']['position_xyz'])[:2]
                                  - np.asarray(c['scene']['box_pose']['position_xyz'])[:2]) < 1e-6
                   for c in formal['cases'])):
        raise ValueError('Formal seeds/physical positions cannot become TRAIN teachers')
    if config['case']['seed'] not in declaration['train_seeds'] or config['scene'] not in declaration['train_scenes']:
        raise ValueError('Actual teacher differs from its prospective TRAIN declaration')
    evidence = {str(p): sha(p) for p in (case / 'result.json', case / 'config.json',
                case / 'trajectory.jsonl', case / 'runtime_identity.json', audit_path, declaration_path)}
    for name, digest in audit['inputs_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Independent audit evidence changed: ' + name)
    for p in (case / 'result.json', case / 'config.json', case / 'trajectory.jsonl', case / 'runtime_identity.json'):
        if audit['inputs_sha256'].get(str(p)) != evidence[str(p)]:
            raise ValueError('Actual teacher audit identity mismatch')
    rows = [json.loads(line) for line in (case / 'trajectory.jsonl').read_text().splitlines() if line.strip()]
    length = len(rows) - 1
    if length != result['completed_control_ticks'] or length <= 100 or result['actual_sdk_integrations'] != 4 * length:
        raise ValueError('Actual complete control/physics coverage differs')
    states, actions, joints, frames = [], [], [], []
    names = result['source_joint_names']
    for tick, row in enumerate(rows[:-1]):
        if row['control_tick'] != tick or row['physics_steps_since_reset'] != 4 * tick or row['source_joint_names'] != names:
            raise ValueError('Teacher state/physics time gap')
        q = np.asarray(row['joint_positions'], np.float32)
        action = np.asarray(rows[tick + 1]['applied_action32'], np.float32)
        if q.shape != (43,) or action.shape != (32,) or not np.isfinite(action).all():
            raise ValueError('Teacher input/target shape differs')
        if not np.array_equal(decode_to_source(action, q, names), np.asarray(rows[tick + 1]['applied_source_action50'], np.float32)):
            raise ValueError('Next-row target/source50 alignment differs')
        path = case / 'head_rgb' / f'{tick:05d}.png'
        entry = {'path': str(path), 'byte_sha256': sha(path), 'rgb_sha256': row['head_rgb_sha256'],
                 'control_tick': tick, 'timestamp_seconds': tick / 50}
        raw_png(entry)
        states.append(encode_state(q, names)); actions.append(action); joints.append(q); frames.append(entry)
    if sha(SPLIT) != SPLIT_SHA:
        raise ValueError('Original whole-episode split changed')
    output.mkdir(parents=True)
    with (output / 'episode.npz').open('xb') as stream:
        np.savez(stream, state=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32),
                 measured_joint_positions43=np.asarray(joints, np.float32), timestamps=np.arange(length) / 50)
    manifest = {'schema': 'lingbot_fresh_full_success_TRAIN_episode_v1', 'status': 'completed',
                'train_only': True, 'validation': [], 'test': [], 'task_success_teacher': True,
                'source_case_id': case_id, 'seed': config['case']['seed'], 'length': length,
                'source_joint_names': names, 'raw_rgb_frames': frames, 'npz': 'episode.npz',
                'npz_sha256': sha(output / 'episode.npz'), 'independent_audit': str(audit_path),
                'independent_audit_sha256': sha(audit_path), 'evidence_sha256': evidence,
                'full_audit_inputs_sha256': audit['inputs_sha256'], 'source_split_sha256': SPLIT_SHA,
                'LingBot_original80train_norm_sha256': sha(OWN / 'train_meanstd.json'),
                'source_test_payloads_opened': False, 'formal_trajectories_used_for_training': False,
                'input_ticks': [0, length - 1], 'target_applied_rows': [1, length],
                'physical_outcome': outcome, 'normalization_policy': 'Original80TRAIN statistics unchanged; this new teacher is excluded from fitting.'}
    write(output / 'manifest.json', manifest)
    return manifest


class FullSuccessEpisode:
    def __init__(self, manifest, normalizer):
        path = Path(manifest).absolute()
        if not path.is_relative_to(OWN):
            raise ValueError('LingBot-owned admitted teacher required')
        data = json.loads(path.read_text())
        if (data.get('schema') != 'lingbot_fresh_full_success_TRAIN_episode_v1'
                or data.get('status') != 'completed' or data.get('task_success_teacher') is not True
                or data.get('train_only') is not True or data.get('validation') or data.get('test')
                or data.get('source_split_sha256') != SPLIT_SHA
                or data.get('LingBot_original80train_norm_sha256') != sha(OWN / 'train_meanstd.json')):
            raise ValueError('Explicit audited fresh TRAIN-only full-success data required')
        audit = json.loads(Path(data['independent_audit']).read_text())
        if (sha(data['independent_audit']) != data['independent_audit_sha256']
                or audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True
                or audit.get('recomputed_acceptance') != data['physical_outcome']
                or data['physical_outcome'].get('success') is not True):
            raise ValueError('Successful teacher actual independent physical audit differs')
        for name, digest in {**data['evidence_sha256'], **data['full_audit_inputs_sha256']}.items():
            if sha(name) != digest:
                raise ValueError('Successful teacher evidence changed: ' + name)
        payload = path.parent / data['npz']
        if sha(payload) != data['npz_sha256']:
            raise ValueError('Successful teacher tensors changed')
        with np.load(payload, allow_pickle=False) as arrays:
            self.states = arrays['state'].copy()
            raw_actions = arrays['actions'].copy()
            q = arrays['measured_joint_positions43'].copy()
            times = arrays['timestamps'].copy()
        n = data['length']
        if self.states.shape != (n, 32) or raw_actions.shape != (n, 32) or q.shape != (n, 43):
            raise ValueError('Successful teacher tensor shape differs')
        if not np.array_equal(self.states, np.asarray([encode_state(row, data['source_joint_names']) for row in q])) or not np.allclose(times, np.arange(n) / 50, rtol=0, atol=1e-9):
            raise ValueError('Successful teacher measured-state/timing differs')
        self.frames = data['raw_rgb_frames']
        if len(self.frames) != n:
            raise ValueError('Every supervised input requires a synchronized lossless image')
        for index, entry in enumerate(self.frames):
            if entry['control_tick'] != index or entry['timestamp_seconds'] != index / 50:
                raise ValueError('Successful teacher RGB timing differs')
            raw_png(entry)
        self.actions = normalizer.normalize(raw_actions, 'actions')
        self.indices = range(n)
        self.episodes = [{'id': data['source_case_id']}]

    def sample(self, index):
        stop = min(index + 50, len(self.actions))
        count = stop - index
        targets = np.repeat(self.actions[stop - 1:stop], 50, axis=0)
        targets[:count] = self.actions[index:stop]
        return raw_png(self.frames[index]), self.states[index], targets, np.arange(50) < count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'audit', 'declaration', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.case, args.audit, args.declaration, args.output)
    print(json.dumps({'status': result['status'], 'task_success_teacher': True, 'frames': result['length']}), flush=True)


if __name__ == '__main__':
    main()
