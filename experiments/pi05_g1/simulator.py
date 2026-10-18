"""G1 source PhysX loop for paired pi05 evaluation, with truth-only audit.

Run through sim_launcher. Config supplies frozen assets, source_harness_dir,
geometry_definition, robot_geometry, case, output_dir, mode, policy_command,
norm_stats and replay_result. Model IPC contains RGB, measured self state and text only.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .sim_launcher import ARENA, BUILD, LAB, probe
from .placement import acceptance_identity, evaluate as evaluate_sampled_placement


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_config(value):
    """Bind config callables by import identity, never their ASLR address."""
    if isinstance(value, dict):
        return {str(k): canonical_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_config(v) for v in value]
    if callable(value):
        return {'callable': f'{value.__module__}.{value.__qualname__}'}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return canonical_config(value.tolist())
    raise TypeError(f'Configuration cannot be reproducibly fingerprinted: {type(value)}')


def validate_case(case):
    if not isinstance(case.get('case_id'), str) or not case['case_id']:
        raise ValueError('A frozen case_id is required')
    if not isinstance(case.get('seed'), int):
        raise ValueError('An integer frozen seed is required')
    offset = case.get('apple_offset_xy_m', case.get('apple_xy_offset_m'))
    if not isinstance(offset, list) or len(offset) != 2 or any(not math.isfinite(float(x)) or abs(float(x)) > .02 for x in offset):
        raise ValueError('Case offset must be a finite XY pair within two centimetres')
    return {**case, 'apple_offset_xy_m': offset}


def shared_action_bounds(stats, source_limits, source_names):
    """Training quantiles intersect original joint limits, in policy order."""
    import numpy as np
    from .contract import UPPER_NAMES
    lo, hi = np.asarray(stats['q01'], np.float32).copy(), np.asarray(stats['q99'], np.float32).copy()
    limits = np.asarray(source_limits, np.float32)
    if lo.shape != (32,) or hi.shape != (32,) or limits.shape != (43, 2):
        raise ValueError('Action quantiles or original joint limits have wrong shape')
    ids = [list(source_names).index(n) for n in UPPER_NAMES]
    lo[:28] = np.maximum(lo[:28], limits[ids, 0])
    hi[:28] = np.minimum(hi[:28], limits[ids, 1])
    if not np.isfinite([lo, hi]).all() or np.any(lo > hi):
        raise ValueError('Shared action bounds are nonfinite or have empty intersection')
    return lo, hi


class PolicyProcess:
    """One isolated model process; synchronous calls pause simulator physics."""
    def __init__(self, command, stderr_path, timeout_seconds=180):
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('policy_command must be an explicit argv array')
        self.stderr = Path(stderr_path).open('x')
        # Isaac's interpreter exports its own Python3.12/Kit/CUDA library paths.
        # The policy uses its independent Python3.11 venv and cu13 wheels.
        child_env = dict(os.environ)
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONHOME', 'PYTHONSTARTUP'):
            child_env.pop(key, None)
        repo = Path(__file__).resolve().parents[2]
        child_env['PYTHONPATH'] = os.pathsep.join((str(repo), str(repo / '.scratch/pi05_g1/openpi/src'),
                                                 str(repo / '.scratch/pi05_g1/openpi/packages/openpi-client/src')))
        child_env['JAX_PLATFORMS'] = 'cpu'
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1, env=child_env)
        self.timeout = timeout_seconds

    def infer(self, rgb, state, case, tick, prompt):
        import numpy as np
        request = {'schema': 'pi05_g1_observation_v1', 'episode_id': case['case_id'],
                   'control_tick': tick, 'rgb_shape': list(rgb.shape),
                   'rgb_uint8_b64': base64.b64encode(rgb.tobytes()).decode(),
                   'state': np.asarray(state).tolist(), 'prompt': prompt, 'seed': case['seed'] + tick}
        started = time.monotonic()
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError('Model inference deadline exceeded while simulation paused')
        line = self.process.stdout.readline(1_048_577)
        if not line or len(line) > 1_048_576:
            raise ValueError('Model disconnected or exceeded reply bound')
        reply = json.loads(line)
        if reply.get('error'):
            raise RuntimeError(reply['error'])
        if reply.get('episode_id') != case['case_id'] or reply.get('control_tick') != tick:
            raise ValueError('Model reply changed observation identity')
        actions = np.asarray(reply['actions'], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 32 or not 1 <= len(actions) <= 50 or not np.isfinite(actions).all():
            raise ValueError('Model must return one to fifty finite 32-dimensional actions')
        return actions, {'inference_seconds': reply['inference_seconds'],
                         'simulation_paused_wall_seconds': time.monotonic() - started,
                         'observation_control_tick': tick}

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


def refresh_camera(raw, robot, camera, objects, measured):
    """Original render-only reset repair; asserts zero physics/state changes."""
    import carb
    import omni.kit.app
    import numpy as np
    def state():
        return [measured(robot.data.joint_pos).copy(), measured(robot.data.joint_vel).copy(),
                measured(robot.data.root_link_pose_w).copy(),
                *[measured(raw.scene[n].data.root_state_w).copy() for n in objects]]
    before = state()
    counters = (int(raw._sim_step_counter), int(raw.sim._physics_step_count))
    key = '/app/player/playSimulations'
    previous = carb.settings.get_settings().get(key)
    try:
        raw.sim.physics_manager.forward()
        raw.sim.set_setting(key, False)
        omni.kit.app.get_app().update()
    finally:
        raw.sim.set_setting(key, previous)
    camera.reset()
    _ = camera.data.output['rgb']
    if counters != (int(raw._sim_step_counter), int(raw.sim._physics_step_count)) or any(
            not np.array_equal(a, b) for a, b in zip(before, state())):
        raise ValueError('Camera refresh changed physical state or advanced physics')


def run(config):
    import numpy as np
    from .contract import encode_state, decode_to_source
    case = validate_case(config['case'])
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    captures = output / 'policy_observations'
    captures.mkdir()
    result = {'schema': 'pi05_g1_case_result_v1', 'case_id': case['case_id'],
              'policy': config['mode'], 'status': 'running', 'success': None,
              'grasped': False, 'stable_placement': False, 'dropped': False,
              'first_grasp_control_tick': None,
              'fell': False, 'timeout': False, 'source_interface_passed': False,
              'replay_passed': None, 'simulation_pauses_for_inference': True,
              'real_time_1x_verified': False, 'control_hz': 50, 'physics_hz': 200,
              'video_path': str(output / 'head_50fps.mp4'), 'case': case,
              'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    acceptance = acceptance_identity()
    result.update(acceptance_version=acceptance['acceptance_version'], acceptance_sha256=acceptance['acceptance_sha256'])
    gate = None
    if config['mode'] != 'replay':
        gate_path = Path(config['replay_result'])
        gate = json.loads(gate_path.read_text())
        if gate.get('policy') != 'replay' or gate.get('status') != 'completed' or gate.get('replay_passed') is not True or gate.get('source_interface_passed') is not True:
            raise ValueError('Successful physical original-demonstration replay is required before policy evaluation')
        if any(gate.get(key) != acceptance[key] for key in ('acceptance_version', 'acceptance_sha256')):
            raise ValueError('Replay gate uses a different acceptance measurement version or module identity')
        result.update(replay_passed=True, replay_gate_sha256=digest(gate_path))
    sys.path.insert(0, str(Path(config['source_harness_dir']).resolve()))
    from unitree_g1_t1_source_task import USD_SHA, URDF_SHA, AGILE_SHA, BACKGROUND_SHA, APPLE_SHA, PLATE_SHA
    assets = config['assets']
    for name, sha in [('usd', USD_SHA), ('urdf', URDF_SHA), ('agile', AGILE_SHA),
                      ('background', BACKGROUND_SHA), ('apple', APPLE_SHA), ('plate', PLATE_SHA)]:
        if digest(assets[name]) != sha:
            raise ValueError(f'Frozen source artifact changed: {name}')
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    from isaaclab_arena_environments import galileo_g1_static_pick_and_place_environment as scene_module
    parser = get_isaaclab_arena_cli_parser()
    scene_module.GalileoG1StaticPickAndPlaceEnvironment.add_cli_args(parser)
    launch = parser.parse_args(['--headless', '--enable_cameras', '--no-solve-relations', '--embodiment', 'g1_wbc_agile_joint'])
    launch.seed = case['seed']
    # Only initial apple spawn changes between frozen cases. No runtime pose writes.
    origin = scene_module.PICK_UP_OBJECT_SPAWN_XY
    scene_module.PICK_UP_OBJECT_SPAWN_XY = tuple(a + b for a, b in zip(origin, case['apple_offset_xy_m']))
    scene_module.APPLE_SPAWN_XY_RANGE_M = 0.0
    sys.argv = [sys.argv[0]]
    model = video = env = None
    rows = []
    def finalize_evidence(error=None):
        nonlocal model, video
        if error is not None:
            result.update(status='failed', error=repr(error), success=None)
        # Source SimulationAppContext deliberately uses os._exit on error, and
        # Kit close may also terminate Python. Commit evidence BEFORE __exit__.
        if model is not None:
            try:
                model.close()
            except Exception as cleanup_error:
                result.setdefault('cleanup_errors', []).append(repr(cleanup_error))
            model = None
        if video is not None:
            try:
                video.close()
            except Exception as cleanup_error:
                result.setdefault('cleanup_errors', []).append(repr(cleanup_error))
                result.update(status='failed', success=None)
            video = None
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str))
    class EvidenceContext(SimulationAppContext):
        def __exit__(self, exc_type, exc_val, exc_tb):
            finalize_evidence(exc_val)
            return super().__exit__(exc_type, exc_val, exc_tb)
    try:
        with EvidenceContext(launch):
            import torch
            import warp as wp
            import imageio.v2 as imageio
            from PIL import Image
            from isaaclab_arena.assets.background_library import GalileoLocomanipBackground
            from isaaclab_arena.assets.object_library import Apple01ObjaverseRobolab, ClayPlatesHot3DRobolab
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as actions_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as robot_module
            from unitree_g1_t1_source_contacts import make_apple_contact_view, sample_apple_contacts
            from unitree_g1_t1_source_placement import evaluate as audit_placement, DEFINITION_SHA, ROBOT_SHA
            from unitree_g1_source_expert import load_sequence
            from importlib.metadata import version
            result['runtime_packages'] = {n: version(n) for n in ('torch', 'numpy', 'warp-lang', 'onnxruntime', 'isaaclab', 'isaaclab_arena', 'gymnasium')}
            torch._C._jit_override_can_fuse_on_gpu(False)
            def tensor(v):
                return wp.to_torch(v) if isinstance(v, wp.array) else v
            def measured(v):
                return tensor(v)[0].detach().cpu().numpy()
            factory = actions_module.get_wbc_policy
            def local_factory(name, model, cfg, count):
                if cfg.wbc_version != 'agile':
                    raise ValueError('Frozen controller must remain AGILE')
                cfg.wbc_model_path = str(assets['agile'])
                return factory(name, model, cfg, count)
            actions_module.get_wbc_policy = local_factory
            def robot_path(url, **kwargs):
                if url.endswith('/g1_29dof_with_hand.urdf'):
                    return str(assets['urdf'])
                if url.endswith('/Arena/wbc_policy/robot_model/g1/'):
                    return str(Path(assets['urdf']).parent)
                raise ValueError(f'Unexpected source robot asset: {url}')
            robot_module.retrieve_file_path = robot_path
            GalileoLocomanipBackground.usd_path = str(assets['background'])
            Apple01ObjaverseRobolab.usd_path = str(assets['apple'])
            ClayPlatesHot3DRobolab.usd_path = str(assets['plate'])
            description = scene_module.GalileoG1StaticPickAndPlaceEnvironment().get_env(launch)
            description.embodiment.scene_config.robot.spawn.usd_path = str(assets['usd'])
            env = ArenaEnvBuilder(description, launch).make_registered()
            raw = env.unwrapped
            if raw.cfg.sim.dt != .005 or raw.cfg.decimation != 4 or raw.cfg.episode_length_s != 6.0:
                raise ValueError('Frozen source physical timing changed')
            robot, camera = raw.scene['robot'], raw.scene['robot_head_cam']
            action_term = raw.action_manager.get_term('g1_action')
            names = list(robot.joint_names)
            env.reset(seed=case['seed'])
            if config['mode'] != 'replay':
                stats_path = Path(config['norm_stats'])
                stats = json.loads(stats_path.read_text())
                stats = stats.get('norm_stats', stats)
                lower, upper = shared_action_bounds(stats['actions'], measured(robot.data.soft_joint_pos_limits), names)
                result['action_bounds'] = {'lower': lower.tolist(), 'upper': upper.tolist(), 'train_stats_sha256': digest(stats_path)}
                result['clipped_channels'] = 0
                result['commanded_channels'] = 0
            original_success = {'value': False, 'first_control_tick': None, 'calls': 0}
            source_terms = {}
            # Suppress autoreset before its state is lost. Retain source functions
            # and record their actual results. Episode ends explicitly below.
            for term_name in ('success', 'time_out', 'object_dropped'):
                if term_name not in raw.termination_manager.active_terms:
                    continue
                term_cfg = copy.deepcopy(raw.termination_manager.get_term_cfg(term_name))
                original = term_cfg.func
                def observe(environment, _fn=original, _name=term_name, **params):
                    v = _fn(environment, **params)
                    source_terms[_name] = bool(v.any())
                    if _name == 'success':
                        original_success['calls'] += 1
                        original_success['value'] = bool(v.any())
                        if bool(v.any()) and original_success['first_control_tick'] is None:
                            original_success['first_control_tick'] = int(raw._sim_step_counter) // 4
                    return torch.zeros_like(v)
                term_cfg.func = observe
                raw.termination_manager.set_term_cfg(term_name, term_cfg)
            result['source_terminal_predicates_unchanged'] = True
            result['source_terminal_autoresets_suppressed'] = ['success', 'time_out', 'object_dropped']
            if 'success' not in raw.termination_manager.active_terms or 'time_out' not in raw.termination_manager.active_terms:
                raise ValueError('Original task termination terms unavailable')
            contacts, metadata = make_apple_contact_view(raw, launch.object, launch.destination)
            refresh_camera(raw, robot, camera, [launch.object, launch.destination], measured)
            interface = {'runtime_build': BUILD, 'arena_commit': ARENA, 'lab_commit': LAB,
                         'source_physics_hz': 200, 'source_control_hz': 50,
                         'original_episode_seconds': 6, 'audit_tail_seconds': 2,
                         'source_joint_names': names, 'camera': canonical_config(camera.cfg.to_dict()),
                         'source_action_shape': list(env.action_space.shape),
                         'assets_sha256': {n: digest(p) for n, p in assets.items()},
                         'contract_sha256': digest(Path(__file__).with_name('contract.py')),
                         'simulator_sha256': digest(__file__),
                         'source_harness_sha256': digest(Path(config['source_harness_dir']) / 'unitree_g1_t1_source_task.py'),
                         'geometry_sha256': digest(config['geometry_definition']),
                         'robot_geometry_sha256': digest(config['robot_geometry']),
                         'legacy_geometry_audit_sha256': digest(Path(config['source_harness_dir']) / 'unitree_g1_t1_source_placement.py'),
                         'acceptance_version': acceptance['acceptance_version'],
                         'acceptance_sha256': acceptance['acceptance_sha256']}
            result['frozen_interface_sha256'] = hashlib.sha256(json.dumps(interface, sort_keys=True, default=str).encode()).hexdigest()
            result['frozen_interface'] = interface
            if gate is not None and gate.get('frozen_interface_sha256') != result['frozen_interface_sha256']:
                raise ValueError('Replay gate belongs to a different physical/camera/action interface')
            if digest(config['geometry_definition']) != DEFINITION_SHA or digest(config['robot_geometry']) != ROBOT_SHA:
                raise ValueError('Frozen independent acceptance geometry changed')
            expert = load_sequence(config['expert_sequence'], config['expert_sha256']) if config['mode'] == 'replay' else None
            if expert:
                if max(abs(float(measured(robot.data.joint_pos)[names.index(n)]) - v) for n, v in zip(expert['joint_names'], expert['initial_joint_positions'])) > 1e-3:
                    raise ValueError('Replay initial joint pose mismatch')
            else:
                model = PolicyProcess(config['policy_command'], output / 'model.stderr.log', config.get('inference_timeout_seconds', 180))
            video = imageio.get_writer(str(output / 'head_50fps.mp4'), fps=50, codec='libx264', macro_block_size=None)
            reset_counter = int(raw._sim_step_counter)
            chunk = []
            chunk_at = 0
            last_action = None
            last_source_action = None
            candidate = None
            result['inference_calls'] = []
            initial_apple_z = None
            hand_contact_indices = [i + 1 for i, n in enumerate(metadata['robot_body_names']) if 'hand' in n.lower() or 'finger' in n.lower()]
            if not hand_contact_indices:
                raise ValueError('Original hand body contact identities unavailable')
            result['grasp_measurement_rule'] = 'apple rises >3cm from reset while at least two named hand bodies have nonzero measured contact; contact/lift proxy, not force-closure proof'
            with (output / 'trajectory.jsonl').open('x') as trace:
                for tick in range(401):
                    q = measured(robot.data.joint_pos)
                    root = measured(robot.data.root_link_pose_w)
                    gravity = measured(robot.data.projected_gravity_b)
                    rgb = measured(camera.data.output['rgb'])[:, :, :3].copy()
                    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or not np.isfinite(q).all():
                        raise ValueError('Original RGB/self-state contract failed')
                    video.append_data(rgb)
                    truth = {n: {'pose': measured(raw.scene[n].data.root_pose_w).tolist(), 'velocity': measured(raw.scene[n].data.root_vel_w).tolist()} for n in (launch.object, launch.destination)}
                    row = {'control_tick': tick, 'physics_steps_since_reset': int(raw._sim_step_counter) - reset_counter,
                           'root_link_pose_w': root.tolist(), 'joint_positions': q.tolist(), 'upright': float(-gravity[2]),
                           'acceptance_truth_only': truth, 'acceptance_contacts_only': sample_apple_contacts(contacts, .005),
                           'acceptance_robot_body_link_poses_w': measured(robot.data.body_link_pose_w).tolist(),
                           'original_success_before_audit_suppression': dict(original_success), 'original_terms': dict(source_terms),
                           'audit_tail': tick > 300, 'applied_action32': last_action,
                           'applied_source_action50': last_source_action,
                           'processed_joint_targets': action_term.processed_actions[0].detach().cpu().tolist()}
                    if row['physics_steps_since_reset'] != tick * 4:
                        raise ValueError('Source loop reset or physical integration mismatch')
                    if tick == 0:
                        initial = {'q': row['joint_positions'], 'root': row['root_link_pose_w'], 'objects': truth}
                        result['initial_physical_state_sha256'] = hashlib.sha256(json.dumps(initial, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    rows.append(row)
                    trace.write(json.dumps(row, allow_nan=False) + '\n')
                    trace.flush()
                    apple_z = truth[launch.object]['pose'][2]
                    if initial_apple_z is None:
                        initial_apple_z = apple_z
                    matrix = np.asarray(row['acceptance_contacts_only']['apple_filter_force_matrix_world_n'])
                    contacting_hand_bodies = int(np.count_nonzero(np.linalg.norm(matrix[hand_contact_indices], axis=1) > 1e-6))
                    lifted_in_hand = bool(apple_z - initial_apple_z > .03 and contacting_hand_bodies >= 2)
                    result['grasped'] |= lifted_in_hand
                    if lifted_in_hand and result['first_grasp_control_tick'] is None:
                        result['first_grasp_control_tick'] = tick
                    result['dropped'] |= bool(source_terms.get('object_dropped', False))
                    result['fell'] |= bool(root[2] + .795 < .35 or row['upright'] < .5)
                    if original_success['value'] and candidate is None and tick <= 300:
                        candidate = tick
                    if result['fell'] or result['dropped'] or tick == 400 or (tick == 300 and candidate is None):
                        break
                    if expert:
                        frame = expert['frames'][min(tick, len(expert['frames']) - 1)]
                        action32 = np.asarray(frame['left_arm'] + frame['right_arm'] + frame['left_hand'] + frame['right_hand'] + [frame['base_height_m']] + frame['navigate_mps_rps'], dtype=np.float32)
                    elif tick >= 300:
                        action32 = np.asarray(last_action, dtype=np.float32)
                    else:
                        if chunk_at == len(chunk):
                            before = int(raw._sim_step_counter)
                            state32 = encode_state(q, names)
                            Image.fromarray(rgb).save(captures / f'head_{tick:04}.png')
                            observation = {'episode_id': case['case_id'], 'control_tick': tick,
                                           'state': state32.tolist(), 'prompt': config['prompt'],
                                           'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                           'rgb_path': f'head_{tick:04}.png'}
                            (captures / f'observation_{tick:04}.json').write_text(json.dumps(observation, allow_nan=False))
                            chunk, timing = model.infer(rgb, state32, case, tick, config['prompt'])
                            np.save(captures / f'actions_{tick:04}.npy', chunk, allow_pickle=False)
                            if int(raw._sim_step_counter) != before:
                                raise ValueError('Inference advanced physical simulation')
                            result['inference_calls'].append(timing)
                            chunk_at = 0
                        action32 = chunk[chunk_at]
                        chunk_at += 1
                    if not expert:
                        result['clipped_channels'] += int(np.count_nonzero((action32 < lower) | (action32 > upper)))
                        result['commanded_channels'] += 32
                        action32 = np.clip(action32, lower, upper)
                    action = np.asarray(decode_to_source(action32, q, names), dtype=np.float32)
                    if action.shape != (50,) or not np.isfinite(action).all():
                        raise ValueError('Source decoder must return finite 50-dimensional action')
                    last_action = action32.tolist()
                    last_source_action = action.tolist()
                    _, _, terminated, truncated, info = env.step(torch.as_tensor(action[None], device=raw.device))
                    if bool(terminated.any()) or bool(truncated.any()):
                        raise ValueError('Unexpected automatic source reset; terminal state unavailable')
                    if tick % 50 == 0:
                        print(f'PI05_G1 case={case["case_id"]} tick={tick} policy={config["mode"]}', flush=True)
            receipt = {'placement_contact_view': metadata, 'success_auto_reset_suppressed_for_audit_only': True,
                       'original_drop_and_timeout_terms_unchanged': True, 'source_physics_hz': 200, 'source_control_hz': 50,
                       'completed_control_ticks': rows[-1]['control_tick'], 'measured_physics_steps_since_reset': rows[-1]['physics_steps_since_reset'],
                       'policy_calls': len(result['inference_calls'])}
            audit = audit_placement(json.loads(Path(config['geometry_definition']).read_text()), json.loads(Path(config['robot_geometry']).read_text()), receipt, rows)
            (output / 'legacy_speed_audit.json').write_text(json.dumps(audit, indent=2))
            placement = evaluate_sampled_placement(audit, rows,
                first_grasp_tick=result['first_grasp_control_tick'], source_success_candidate_tick=candidate,
                fell=result['fell'], dropped=result['dropped'])
            placement['legacy_speed_audit_sha256'] = digest(output / 'legacy_speed_audit.json')
            placement['physical_trace_sha256'] = digest(output / 'trajectory.jsonl')
            (output / 'placement_audit.json').write_text(json.dumps(placement, indent=2, allow_nan=False))
            result.update(status='completed', source_interface_passed=True, stable_placement=bool(placement['stable_placement_passed']),
                          source_success_candidate_tick=candidate, max_stable_seconds=2.0 if placement['stable_placement_passed'] else 0.0,
                          legacy_speed_audit_passed=bool(audit['source_diagnostic_placement_passed']),
                          legacy_max_stable_seconds=audit['max_stable_seconds'],
                          qualified_stability_window=placement['qualified_window'],
                          completed_control_ticks=rows[-1]['control_tick'], physics_integrations=rows[-1]['physics_steps_since_reset'],
                          independent_audit_tail_seconds=max(0, rows[-1]['control_tick'] - 300) / 50,
                          audit_tail_action='hold_last_absolute_action_with_original_WBC', runtime_build=BUILD)
            grasp_before_placement = (candidate is not None and result['first_grasp_control_tick'] is not None
                                      and result['first_grasp_control_tick'] < candidate)
            result['success'] = bool(grasp_before_placement and result['stable_placement'] and not result['fell'] and not result['dropped'])
            if config['mode'] == 'replay':
                result['replay_passed'] = result['success']
            result['timeout'] = not result['success'] and not result['fell'] and not result['dropped']
            result['action_clip_fraction'] = (result.get('clipped_channels', 0) / result.get('commanded_channels', 1))
            env.close()
            env = None
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), success=None)
        raise
    finally:
        finalize_evidence()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    runtime = probe(Path(sys.executable))
    if not runtime['ready']:
        raise SystemExit(json.dumps(runtime, indent=2))
    config = json.loads(args.config.read_text())
    if config['mode'] not in ('baseline', 'finetuned', 'replay'):
        parser.error('mode must be baseline, finetuned, or replay')
    try:
        run(config)
    except BaseException as error:
        output = Path(config['output_dir'])
        result_path = output / 'result.json'
        if output.is_dir() and not result_path.exists():
            result_path.write_text(json.dumps({'schema': 'pi05_g1_case_result_v1',
                'case_id': config.get('case', {}).get('case_id'), 'policy': config.get('mode'),
                'status': 'failed', 'success': None, 'error': repr(error)}, indent=2))
        raise


if __name__ == '__main__':
    main()
