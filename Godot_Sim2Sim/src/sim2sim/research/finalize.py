"""Freeze a reviewable candidate bundle without replacing default policies."""
import argparse, hashlib, json, os, time
import shlex
from pathlib import Path
import numpy as np
import onnx
import torch
from sim2sim.paths import sim2sim_root
from sim2sim.policy import OnnxPolicy
from .models import NativeAnchor, Policy, parity
from .tasks import TASKS, SESSION


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(checkpoint, exported, n=10000):
    state = torch.load(checkpoint, weights_only=False, map_location='cpu')
    cfg = state['config']
    gate = tuple(map(float, cfg['time_gate'].split(','))) if cfg.get('time_gate') else None
    actor = Policy(cfg['source'], cfg['variant'], cfg['std'], cfg['bound'],
                   template=cfg.get('template') or cfg['source'], time_gate=gate,
                   command_gate=cfg.get('command_gate',''),mask_task_state=cfg.get('mask_task_state',False),
                   mask_motion_state=cfg.get('mask_motion_state',False),action_basis=cfg.get('action_basis',''))
    actor.load_state_dict(state['policy'])
    if actor.anchor.sha256 != state['factory_sha256']:
        raise ValueError('Checkpoint source has changed')
    report = parity(actor, exported, n=n)
    report.update(samples_per_distribution=n, checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=digest(checkpoint), iteration=state['iteration'])
    return report


def verify_mirror(source, reflected, n=10000):
    from .mirror import reflect_obs, reflect_action
    rng = np.random.default_rng(7319)
    obs = rng.standard_normal((n, 61), dtype=np.float32)
    parent = NativeAnchor(source)
    expected = reflect_action(parent(reflect_obs(obs, task='kick_right',heading_input=parent.heading_input,
                                               yaw_memory_input=parent.yaw_memory_input)))
    actual = NativeAnchor(reflected)(obs)
    error = float(np.max(np.abs(actual - expected)))
    return dict(samples=n, max_abs=error, threshold=1e-5, passed=error < 1e-5,
                source=str(Path(source).resolve()), source_sha256=digest(source))


def verify_gain(source, exported, gain, n=10000):
    obs = np.random.default_rng(7420).standard_normal((n, 61), dtype=np.float32)
    expected = NativeAnchor(source)(obs) * np.float32(gain)
    error = float(np.max(np.abs(expected - NativeAnchor(exported)(obs))))
    return dict(samples=n, gain=gain, max_abs=error, threshold=1e-5, passed=error < 1e-5)


def verify_distilled_walker(choice, exported, n=10000):
    from .conditioning import parity as adapter_parity
    state = torch.load(choice['distill_checkpoint'], weights_only=False, map_location='cpu')
    actor = Policy(state['config']['parent'], 'plain', template=choice['distill_template'])
    actor.load_state_dict(state['policy'])
    parent = parity(actor, choice['distill_export'], n=n)
    adapter = adapter_parity(choice['distill_export'], exported, np.eye(3), n=n,
                             forward_yaw_hinge=(.3, -3.))
    return dict(parent=parent, adapter=adapter, samples_per_distribution=n,
                checkpoint_sha256=digest(choice['distill_checkpoint']),
                passed=parent['passed'] and adapter['passed'])


def freeze(selection, out):
    """Selection is fixed before any final holdout is opened."""
    torch.set_num_threads(2)
    if set(selection) != set(TASKS): raise ValueError('Exactly nine skills are required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    models = out / 'models'; models.mkdir()
    record = dict(frozen_unix=time.time(), status='experimental_candidates',
                  replaces_default_policies=False, skills={},
                  normal_robot='microduck_ball_stand_fix', roller_robot='microduck_roller')
    baseline = json.loads((SESSION / 'session.json').read_text())['baseline_weights']
    record['original_files_unchanged'] = all(digest(v['source']) == v['sha256'] for v in baseline.values())
    root = sim2sim_root()
    physical_files = ['godot/physics_server.gd', 'godot/generated/microduck_ball_stand_fix/robot.tscn',
                      'godot/generated/microduck_ball_stand_fix/robot_spec.json',
                      'godot/generated/microduck_roller/robot.tscn', 'src/mjcf2godot/convert.py']
    record['physical_files'] = {name: digest(root / name) for name in physical_files}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    paths = {}
    for name, task in TASKS.items():
        choice = selection[name]; source = Path(choice['source']).resolve()
        target = models / task.previous; target.write_bytes(source.read_bytes())
        model = onnx.load(target); onnx.checker.check_model(model)
        meta = {p.key: p.value for p in model.metadata_props}
        sidecar = dict(action_scale=1., sim2sim=dict(use_stand_policy=name not in ('walking', 'roller')),
                       research=dict(skill=name, source=str(source), sha256=digest(target),
                                     status='experimental_candidate', role=choice['role']))
        target.with_suffix('.manifest.json').write_text(json.dumps(sidecar, indent=2))
        runtime = OnnxPolicy(target); runtime.check_dims(14)
        if choice.get('distill_checkpoint'):
            check = verify_distilled_walker(choice, target)
        elif choice.get('output_gain_parent'):
            parent_check = verify_checkpoint(choice['checkpoint'], choice['output_gain_parent'])
            gain_check = verify_gain(choice['output_gain_parent'], target, choice['output_gain'])
            check = dict(parent=parent_check, output_gain=gain_check,
                         passed=parent_check['passed'] and gain_check['passed'])
        elif choice.get('checkpoint'):
            check = verify_checkpoint(choice['checkpoint'], target)
        elif choice.get('mirror_source'):
            check = verify_mirror(choice['mirror_source'], target)
        else:
            check = dict(method='byte_identical_retained_actor', passed=digest(source) == digest(target))
        if not check['passed']: raise ValueError('Export verification failed: ' + name)
        item = dict(choice, source=str(source), exported=str(target.resolve()), sha256=digest(target),
                    bytes=target.stat().st_size, metadata=meta, verification=check)
        record['skills'][name] = item
        paths[name] = str(target.resolve())
        (out / 'bundle.json').write_text(json.dumps(record, indent=2))
    (out / 'bank.json').write_text(json.dumps(paths, indent=2))
    # All names are fixed above. The launcher only selects this isolated bank.
    script = ('#!/usr/bin/env bash\nset -euo pipefail\n'
              'bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
              f'cd -- {shlex.quote(str(root))}\n'
              'export MICRODUCK_POLICIES="$bundle_dir/models"\n'
              'exec .venv/bin/python -m sim2sim.play --robot robots/microduck_ball_stand_fix.json "$@"\n')
    (out / 'play.sh').write_text(script); os.chmod(out / 'play.sh', 0o755)
    record['verification_completed_unix'] = time.time()
    (out / 'bundle.json').write_text(json.dumps(record, indent=2))
    document_sidecars(out)
    return json.loads((out / 'bundle.json').read_text())


def document_sidecars(out):
    """Complete descriptive schema-2 fields without changing runtime behavior.

    This packaging step may follow selection freeze. Compare the actual loaded
    contract before and after, and retain both manifest hashes in the audit.
    """
    from dataclasses import asdict
    out = Path(out)
    bundle_path = out / 'bundle.json'
    bundle = json.loads(bundle_path.read_text())
    checks = []
    for name, item in bundle['skills'].items():
        path = out / 'models' / Path(item['exported']).name
        side = path.with_suffix('.manifest.json')
        before_hash = digest(side)
        actor = OnnxPolicy(path)
        def contract(p):
            return dict(obs=p.obs_dim, act=p.act_dim, scale=p.action_scale,
                        time=p.time_input_s, heading=p.heading_input,
                        yaw_memory=p.yaw_memory_input,
                        stand=p.has_standing_partner, limits=asdict(p.twist_limits))
        before = contract(actor)
        probe = np.zeros(61, np.float32)
        expected = actor.infer(probe)
        data = json.loads(side.read_text())
        commands = {
            'standing': dict(encoding='constant', value=[0] * 13),
            'walking': dict(encoding='twist', meaning='body vx, body vy, yaw rate; remaining command entries zero'),
            'sitstand': dict(encoding='sit_flag', meaning='command[0]=1 sit; 0 stand'),
            'ground_pick': dict(encoding='phase', period_s=4, meaning='command[0:2]=cos/sin(2*pi*t/4)'),
            'kick_left': dict(encoding='constant', value=[0] * 13),
            'kick_right': dict(encoding='constant', value=[0] * 13),
            'roulade': dict(encoding='one_shot_time', period_s=5,
                            meaning='obs[48]=elapsed_seconds/5; command[1:13]=0; updated time-aware runtime required'),
            'roller': dict(encoding='roller_throttle_heading_error',
                           meaning='positive push, zero coast, negative brake; command[2]=relative heading error'),
            'roller_crouch': dict(encoding='phase', period_s=5, meaning='command[0:2]=cos/sin(2*pi*t/5)'),
        }
        data.update(schema_version=2, model_api=1, obs_len=61, action_len=14,
                    robot=dict(model='microduck', hw_rev=1, servos='xl330', control_hz=50),
                    name={'standing':'stand_godot','walking':'walk_godot'}.get(name,name+'_godot'),
                    slot={'standing':'stand','walking':'walk'}.get(name, name),
                    kind='perpetual' if name in ('standing','walking','roller') else 'one_shot',
                    entry_pose='standing', description='Experimental candidate: ' + item['role'],
                    command=commands[name],
                    training=dict(source=item['source'], checkpoint=item.get('checkpoint'), role=item['role']),
                    eval=dict(bundle='../bundle.json', summary='../holdout/summary.json',relative_to='manifest_directory'))
        if actor.time_input_s:
            data['command']=dict(encoding='one_shot_time',period_s=actor.time_input_s,
                meaning=f'obs[48]=elapsed_seconds/{actor.time_input_s:g}; remaining command entries zero')
            data.setdefault('runtime_requires', []).append('one_shot_time')
        if actor.heading_input:
            data['command']['meaning']='obs[48]=elapsed_seconds/5; obs[49:51]=declared relative-heading sine/cosine; remaining command entries zero'
            data['command']['heading']='lateral_axis_sin_cos, relative to orientation at skill entry'
            data.setdefault('runtime_requires', []).append('lateral_axis_sin_cos')
        if actor.yaw_memory_input:
            data['command']['memory']='obs[55]: bounded gyro/gravity yaw-drift integral; updated stateful runtime required'
            data.setdefault('runtime_requires', []).extend(['gyro_vertical_integral_v1','reset_memory_on_episode_or_policy_switch'])
        data['sim2sim']['twist_limits'] = before['limits']
        data['sim2sim']['control'] = dict(dt=.005, decimation=4)
        side.write_text(json.dumps(data, indent=2) + '\n')
        loaded = OnnxPolicy(path)
        assert contract(loaded) == before
        assert np.array_equal(loaded.infer(probe), expected)
        assert digest(path) == item['sha256']
        checks.append(dict(skill=name, before_manifest_sha256=before_hash,
                           manifest_sha256=digest(side), runtime_contract=before,
                           model_unchanged=True, contract_unchanged=True, inference_unchanged=True))
        item['manifest_sha256'] = digest(side)
    audit = dict(completed_unix=time.time(), purpose='descriptive schema-2 packaging only', checks=checks)
    (out / 'manifest_packaging_check.json').write_text(json.dumps(audit, indent=2))
    bundle_path.write_text(json.dumps(bundle, indent=2))
    return audit


def main():
    p = argparse.ArgumentParser(); p.add_argument('selection', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); r = freeze(json.loads(a.selection.read_text()), a.out)
    print(json.dumps(dict(out=str(a.out), skills=len(r['skills']), status=r['status'])))


if __name__ == '__main__': main()
