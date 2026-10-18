"""A fresh native TRAIN-only carry prefix, with failed placement kept explicit.

Reuse the existing synchronized RGB/q43/next-row action loader. A new independent
audit and root admission are required; no complete-success label is introduced.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import CarryPrefix, raw_png
from .contract import PROMPT, encode_state, decode_to_source
from .data import OWN, SHARED, SPLIT_SHA, sha

SCHEMA = 'lingbot_audited_native_carry_TRAIN_prefix_v2'


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(case, audit_path, authorization_path, output):
    case, output = Path(case).absolute(), Path(output).absolute()
    audit_path, authorization_path = Path(audit_path).absolute(), Path(authorization_path).absolute()
    if output.exists() or not case.is_relative_to(OWN) or not output.is_relative_to(OWN):
        raise ValueError('Fresh owned prefix output and actual owned source required')
    audit = json.loads(audit_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    result, config = (json.loads((case / name).read_text()) for name in ('result.json', 'config.json'))
    acceptance = audit.get('recomputed_acceptance') or {}
    stop = acceptance.get('carry_qualified_control_tick')
    if (authorization.get('schema') != 'lingbot_fresh_carry_only_TRAIN_admission_v1'
            or authorization.get('source_case_id') != result['case_id']
            or authorization.get('audit_sha256') != sha(audit_path)
            or authorization.get('supervise_through_applied_row') != stop
            or authorization.get('full_task_success_teacher') is not False):
        raise ValueError('Explicit root carry-only boundary admission required')
    if (audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True
            or Path(audit['result_path']).absolute() != case / 'result.json'
            or any(v.get('passed') is not True for v in audit['checks'].values())
            or acceptance.get('success') is not False or acceptance.get('grasped') is not True
            or acceptance.get('fell') or acceptance.get('dropped')
            or type(stop) is not int or stop <= 0):
        raise ValueError('Independent clean qualified carry from a full-task failure required')
    if (result.get('status') != 'completed' or result.get('success') is not False
            or config.get('policy_id') != 'gr00t_n16' or config.get('mode') != 'policy'
            or config.get('save_full_rgb') is not True or config.get('prompt') != PROMPT):
        raise ValueError('Actual native source and every raw TRAIN RGB frame required')
    formal = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    if (config['case']['seed'] in {c['seed'] for c in formal['cases']}
            or any(np.linalg.norm(np.asarray(config['scene']['box_pose']['position_xyz'])[:2]
                                  - np.asarray(c['scene']['box_pose']['position_xyz'])[:2]) < 1e-6
                   for c in formal['cases'])):
        raise ValueError('Formal seed/position cannot become a new TRAIN prefix')
    frozen = dict(audit['inputs_sha256'])
    frozen.update({str(audit_path): sha(audit_path), str(authorization_path): sha(authorization_path)})
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Audited source changed: ' + name)
    rows = [json.loads(line) for line in (case / 'trajectory.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) <= stop or result['completed_control_ticks'] != len(rows) - 1:
        raise ValueError('Actual source does not cover the admitted boundary')
    names = result['source_joint_names']
    states, targets, joints, frames = [], [], [], []
    for tick, row in enumerate(rows[:stop]):
        q = np.asarray(row['joint_positions'], np.float32)
        action = np.asarray(rows[tick + 1]['applied_action32'], np.float32)
        if (row['control_tick'] != tick or row['physics_steps_since_reset'] != 4 * tick
                or row['source_joint_names'] != names or q.shape != (43,) or action.shape != (32,)
                or not np.array_equal(decode_to_source(action, q, names), np.asarray(rows[tick + 1]['applied_source_action50'], np.float32))):
            raise ValueError('Actual state/next-row applied target alignment differs')
        path = case / 'head_rgb' / f'{tick:05d}.png'
        frame = {'path': str(path), 'byte_sha256': sha(path), 'rgb_sha256': row['head_rgb_sha256'],
                 'control_tick': tick, 'timestamp_seconds': tick / 50}
        raw_png(frame)
        frames.append(frame); states.append(encode_state(q, names)); targets.append(action); joints.append(q)
        frozen[str(path)] = frame['byte_sha256']
    first = acceptance['first_grasp_control_tick']
    if not 0 < first + 100 < stop:
        raise ValueError('Both first-contact/stabilization and later carry phases required')
    output.mkdir(parents=True)
    with (output / 'prefix.npz').open('xb') as stream:
        np.savez(stream, state=np.asarray(states, np.float32), actions=np.asarray(targets, np.float32),
                 measured_joint_positions43=np.asarray(joints, np.float32), timestamps=np.arange(stop) / 50)
    frozen[str(output / 'prefix.npz')] = sha(output / 'prefix.npz')
    manifest = {'schema': SCHEMA, 'status': 'completed', 'train_only': True, 'validation': [], 'test': [],
                'source_case_id': result['case_id'], 'seed': config['case']['seed'], 'length': stop,
                'source_task_success': False, 'task_success_teacher': False, 'positive_carry_only': True,
                'source_joint_names': names, 'npz': 'prefix.npz', 'npz_sha256': sha(output / 'prefix.npz'),
                'raw_rgb_frames': frames, 'first_grasp_contact_lift_tick': first,
                'approach_and_stabilization_stop_exclusive': first + 100,
                'input_ticks': [0, stop - 1], 'target_applied_rows': [1, stop], 'excluded_after_tick': stop,
                'independent_audit': str(audit_path), 'independent_audit_sha256': sha(audit_path),
                'source_split_sha256': SPLIT_SHA, 'LingBot_norm_sha256': sha(OWN / 'train_meanstd.json'),
                'extra_frozen_files': frozen, 'formal_or_original_test_training': False,
                'scope': 'Carry-positive prefix only. No arrival/release/placement success labels; later failed source retained and excluded from supervision.'}
    write(output / 'manifest.json', manifest)
    return manifest


def admit(manifest):
    path = Path(manifest).absolute()
    data = json.loads(path.read_text())
    if (not path.is_relative_to(OWN) or data.get('schema') != SCHEMA or data.get('status') != 'completed'
            or data.get('train_only') is not True or data.get('validation') or data.get('test')
            or data.get('positive_carry_only') is not True or data.get('source_task_success') is not False
            or data.get('task_success_teacher') is not False or data.get('source_split_sha256') != SPLIT_SHA
            or data.get('LingBot_norm_sha256') != sha(OWN / 'train_meanstd.json')):
        raise ValueError('Explicit new TRAIN-only carry-positive/task-failed identity required')
    for name, digest in data['extra_frozen_files'].items():
        if sha(name) != digest:
            raise ValueError('Admitted new carry evidence changed: ' + name)
    audit = json.loads(Path(data['independent_audit']).read_text())
    if (audit.get('passed') is not True or audit.get('recomputed_acceptance', {}).get('success') is not False
            or audit.get('case_id') != data['source_case_id'] or audit.get('seed') != data['seed']
            or audit['recomputed_acceptance']['first_grasp_control_tick'] != data['first_grasp_contact_lift_tick']
            or data['approach_and_stabilization_stop_exclusive'] != data['first_grasp_contact_lift_tick'] + 100
            or audit['recomputed_acceptance']['carry_qualified_control_tick'] != data['length']):
        raise ValueError('Actual qualified carry boundary differs')
    return data


class BalancedNewCarryPrefix(CarryPrefix):
    def __init__(self, manifest, normalizer):
        data = admit(manifest)
        super().__init__(manifest, normalizer)
        self.episodes = [{'id': data['source_case_id']}]
        self.boundary = data['approach_and_stabilization_stop_exclusive']
        self.phase_draw_counts = {'approach_and_grasp': 0, 'carry_remainder': 0}

    def sample(self, index):
        if random.random() < .5:
            phase, frame = 'approach_and_grasp', random.randrange(self.boundary)
        else:
            phase, frame = 'carry_remainder', random.randrange(self.boundary, len(self.states))
        self.phase_draw_counts[phase] += 1
        return super().sample(frame)


class EqualCarryFamilies:
    def __init__(self, original_prefix, fresh_prefix):
        self.sources = [original_prefix, fresh_prefix]
        self.indices = range(sum(len(d.indices) for d in self.sources))
        self.episodes = [*original_prefix.episodes, *fresh_prefix.episodes]
        self.family_draw_counts = {'native201': 0, 'fresh_native2026101702': 0}

    def sample(self, index):
        family = random.randrange(2)
        self.family_draw_counts[('native201', 'fresh_native2026101702')[family]] += 1
        return self.sources[family].sample(index)

    @property
    def phase_draw_counts(self):
        return {name: sum(d.phase_draw_counts[name] for d in self.sources)
                for name in ('approach_and_grasp', 'carry_remainder')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'audit', 'authorization', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.case, args.audit, args.authorization, args.output)
    print(json.dumps({'status': value['status'], 'frames': value['length'], 'task_success_teacher': False}), flush=True)


if __name__ == '__main__':
    main()
