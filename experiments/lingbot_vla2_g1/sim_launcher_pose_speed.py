"""Version gate and explicit launcher for the frozen G1 source simulator.

The probe never imports Isaac Sim, starts a GPU context, or installs software.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/home/ethan/models/unitree_g1/envs/arena_spark_cu13/bin/python')
BUILD = '6.0.0-rc.22+release.33481.407f3ea1.gl'
ARENA = '8b4a3a47fc53de23e8205089d71109a2e2348acd'
LAB = 'e57379c634b42db5a0fe9f754341be6e2a7c7c43'
IMAGE = 'nvcr.io/nvidia/isaac-sim:6.0.0-dev2'
ARM64_MANIFEST = 'sha256:9bc527c60c1cdde532b05859b955e866e6115ac5bf2fe22eaa24daa6ea8783e6'
OWNED_IMAGE = 'sai-pi05-g1-source:6.0-frozen'


def docker_command(config=None, *, gpu=False, probe_only=False, image=OWNED_IMAGE,
                   cidfile=None, launch_id=None, budget_token=None):
    """Read-only source/model mounts and one writable experiment evidence tree."""
    evidence = ROOT / '.scratch/lingbot_vla2_g1'
    command = ['docker', 'run', '--rm', '--init', '--network=host', '--ipc=host',
               '--memory', '90g', '--memory-reservation', '80g', '--memory-swap', '90g',
               '--label', 'sai.task=lingbot_vla2_g1',
               '--env', 'ACCEPT_EULA=Y', '--env', 'PRIVACY_CONSENT=Y',
               '--env', f'PYTHONPATH={ROOT}',
               '--env', 'NVIDIA_DRIVER_CAPABILITIES=all']
    if cidfile is not None:
        command += ['--cidfile', str(cidfile)]
    if launch_id is not None:
        command += ['--label', f'sai.pi05_g1.launch={launch_id}']
        logs = evidence / 'sim_launchers' / launch_id / 'kit_logs'
        logs.mkdir(parents=True, exist_ok=True)
        command += ['--volume', f'{logs}:/isaac-sim/kit/logs:rw']
    if budget_token is not None:
        command += ['--label', f'sai.pi05_g1.budget={budget_token}']
    if gpu:
        command += (['--device', 'nvidia.com/gpu=all'] if Path('/var/run/cdi/nvidia.yaml').exists()
                    else ['--gpus', 'all'])
    mounts = [Path('/home/ethan/models'), ROOT,
              Path('/home/ethan/Projects/TempWorktree/Sai_Lab')]
    for host_python in [Path('/home/ethan/.local/share/mise/installs/python'),
                        Path('/home/ethan/.local/share/uv/python')]:
        if host_python.exists():
            mounts.append(host_python)
    for source in mounts:
        command += ['--volume', f'{source}:{source}:ro']
    command += ['--volume', f'{evidence}:{evidence}:rw', '--workdir', str(ROOT),
                '--entrypoint', '/isaac-sim/python.sh']
    # Exact base inspection found no prewarmed /root/.cache/{ov,nvidia,warp}.
    # Its 7.9 GiB /isaac-sim/extscache remains in the immutable image, uncovered.
    for name in ('ov', 'nvidia', 'warp'):
        cache = ROOT / '.scratch/pi05_g1/sim_cache' / name
        cache.mkdir(parents=True, exist_ok=True)
        command += ['--volume', f'{cache}:/root/.cache/{name}:rw']
    # Kit's actual cache token in the frozen app is image-relative, not HOME.
    kit_cache = ROOT / '.scratch/pi05_g1/sim_cache' / 'kit'
    kit_cache.mkdir(parents=True, exist_ok=True)
    command += ['--volume', f'{kit_cache}:/isaac-sim/kit/cache:rw']
    # First real GPU startup confirmed CUDA's driver cache under .nv, not XDG.
    compute_cache = ROOT / '.scratch/pi05_g1/sim_cache' / 'cuda_compute'
    compute_cache.mkdir(parents=True, exist_ok=True)
    command += ['--volume', f'{compute_cache}:/root/.nv/ComputeCache:rw']
    command += [image, '-m', 'experiments.pi05_g1_mobile.sim_launcher_pose_speed', '--python',
                '/isaac-sim/kit/python/bin/python3']
    if probe_only:
        command += ['--probe']
    elif config is not None:
        command += ['--config', str(Path(config).resolve())]
    else:
        raise ValueError('An explicit config or probe_only is required')
    return command


def run_owned_docker(config=None, *, probe_only=False, image=OWNED_IMAGE):
    """Stop only this launch's labelled container when its owner is cancelled."""
    launch_id = uuid.uuid4().hex
    output = ROOT / '.scratch/lingbot_vla2_g1/sim_launchers' / launch_id
    output.mkdir(parents=True, exist_ok=False)
    cidfile = output / 'container.cid'
    ownership_dir = os.environ.get('PI05_G1_BUDGET_OWNERSHIP_DIR')
    budget_token = os.environ.get('PI05_G1_BUDGET_TOKEN')
    if bool(ownership_dir) != bool(budget_token):
        raise ValueError('Docker budget ownership directory and token must be supplied together')
    registration_path = registration = None
    def update_registration():
        if registration_path is not None:
            temporary = registration_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(registration, indent=2))
            temporary.replace(registration_path)
    if ownership_dir:
        directory = Path(ownership_dir)
        if not directory.is_absolute():
            raise ValueError('Budget Docker ownership directory must be absolute')
        directory.mkdir(parents=True, exist_ok=True)
        registration_path = directory / f'launch-{launch_id}.json'
        registration = {'schema': 'pi05_g1_docker_ownership_v1', 'budget_token': budget_token,
                        'launch_id': launch_id, 'cidfile': str(cidfile),
                        'budget_label': 'sai.pi05_g1.budget', 'launch_label': 'sai.pi05_g1.launch',
                        'status': 'registered', 'container_id': None, 'removed': None,
                        'cleanup_receipt': str(output / 'receipt.json')}
        update_registration()
    command = docker_command(config, gpu=not probe_only, probe_only=probe_only,
                             image=image, cidfile=cidfile, launch_id=launch_id, budget_token=budget_token)
    receipt = {'launch_id': launch_id, 'command': command, 'probe_only': probe_only,
               'gpu_requested': not probe_only, 'started_unix_s': time.time(),
               'cleanup_commands': [], 'returncode': None}
    process = None
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    prior_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        process = subprocess.Popen(command)
        receipt['returncode'] = process.wait()
        if receipt['returncode']:
            raise subprocess.CalledProcessError(receipt['returncode'], command)
    finally:
        # Do not interrupt bounded cleanup with another TERM from the owner.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if cidfile.exists():
                cid = cidfile.read_text().strip()
                if len(cid) == 64 and all(c in '0123456789abcdef' for c in cid):
                    receipt['container_id'] = cid
                    inspected = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, timeout=3)
                    if inspected.returncode == 0:
                        container = json.loads(inspected.stdout)[0]
                        labels = container.get('Config', {}).get('Labels', {}) or {}
                        owned = labels.get('sai.pi05_g1.launch') == launch_id and container.get('Id') == cid
                        receipt['container_ownership_verified'] = owned
                        if owned and container.get('State', {}).get('Running'):
                            stop = ['docker', 'stop', '--time', '5', cid]
                            receipt['cleanup_commands'].append(stop)
                            try:
                                stopped = subprocess.run(stop, capture_output=True, text=True, timeout=8)
                                receipt['stop_returncode'] = stopped.returncode
                            except subprocess.TimeoutExpired:
                                receipt['stop_timed_out'] = True
                            remaining = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', cid], capture_output=True, text=True, timeout=2)
                            if remaining.returncode == 0 and remaining.stdout.strip() == 'true':
                                kill = ['docker', 'kill', cid]
                                receipt['cleanup_commands'].append(kill)
                                receipt['kill_returncode'] = subprocess.run(kill, capture_output=True, text=True, timeout=2).returncode
                    else:
                        receipt['container_already_removed'] = True
                else:
                    receipt['cleanup_error'] = 'Invalid owned container identity; no container was signalled'
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        except Exception as error:
            receipt['cleanup_error'] = repr(error)
        finally:
            if registration is not None:
                registration['status'] = 'finished'
                registration['container_id'] = receipt.get('container_id')
                try:
                    cid = registration['container_id']
                    if cid:
                        remaining = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, timeout=2)
                        if remaining.returncode == 0:
                            registration['removed'] = False
                        elif 'no such' in remaining.stderr.lower():
                            registration['removed'] = True
                        else:
                            registration['removed'] = None
                            registration['cleanup_pending'] = remaining.stderr.strip()
                    else:
                        registration['cleanup_pending'] = 'No CID available; budget must inspect owned token labels'
                except Exception as error:
                    registration['cleanup_pending'] = repr(error)
                update_registration()
            receipt['finished_unix_s'] = time.time()
            (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
            signal.signal(signal.SIGTERM, prior_handler)


def probe(python: Path = PYTHON) -> dict:
    code = """import importlib.metadata as m, importlib.util as u, json, pathlib
def dist(n):
 try: return m.version(n)
 except m.PackageNotFoundError: return None
paths={n:str(u.find_spec(n).origin) if u.find_spec(n) else None for n in ['isaacsim','isaaclab','isaaclab_arena']}
build=pathlib.Path('/isaac-sim/VERSION')
print(json.dumps({'versions':{n:dist(n) for n in ['isaacsim','isaaclab','isaaclab_arena']},'module_paths':paths,'runtime_build':build.read_text().strip() if build.exists() else None}))
"""
    result = {'schema': 'pi05_g1_sim_runtime_probe_v1', 'python': str(python),
              'gpu_started': False, 'expected_runtime_build': BUILD,
              'restore_image': IMAGE, 'arm64_manifest': ARM64_MANIFEST,
              'ready': False, 'blockers': []}
    if not python.exists():
        result['blockers'].append('Runtime Python is absent')
        return result
    completed = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=30)
    if completed.returncode:
        result['blockers'].append(completed.stderr.strip())
        return result
    result.update(json.loads(completed.stdout))
    if result['runtime_build'] != BUILD:
        result['blockers'].append(f"Frozen IsaacSim build required: {BUILD}; found {result['runtime_build'] or result['versions']['isaacsim']}")
    for name, expected in [('isaaclab_arena', ARENA), ('isaaclab', LAB)]:
        module = result['module_paths'][name]
        checkout = None
        if module:
            checkout = next((p for p in Path(module).parents if (p / '.git').exists()), None)
        if checkout is None:
            result['blockers'].append(f'{name}: source checkout identity unavailable')
            continue
        git = ['git', '-c', f'safe.directory={checkout}', '-C', str(checkout)]
        revision_run = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        status_run = subprocess.run(git + ['status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, timeout=10)
        revision, dirty = revision_run.stdout.strip(), status_run.stdout.strip()
        if revision_run.returncode or status_run.returncode:
            result['blockers'].append(f'{name}: git identity check failed: {revision_run.stderr} {status_run.stderr}')
            continue
        result.setdefault('source_revisions', {})[name] = {'commit': revision, 'tracked_changes': bool(dirty)}
        if revision != expected or dirty:
            result['blockers'].append(f'{name}: requires clean {expected}, found {revision}, dirty={bool(dirty)}')
    result['ready'] = not result['blockers']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--docker', action='store_true', help='Use the owned frozen source image')
    parser.add_argument('--image', default=OWNED_IMAGE)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.docker:
        if not args.probe and not args.config:
            parser.error('--config is required to execute; --probe is read only')
        run_owned_docker(args.config, probe_only=args.probe, image=args.image)
        return
    receipt = probe(args.python)
    print(json.dumps(receipt, indent=2), flush=True)
    if args.probe:
        return
    if not args.config:
        parser.error('--config is required to execute; --probe is read only')
    if not receipt['ready']:
        raise SystemExit('Blocked: restore the frozen runtime before executing a task. No GPU was started.')
    subprocess.run([str(args.python), '-m', 'experiments.pi05_g1_mobile.acceptance_pose_speed', '--config', str(args.config.resolve())], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
