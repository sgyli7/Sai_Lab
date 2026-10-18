"""Lock the official RoboTwin baseline identity without running it.

The entry reads public metadata and the pinned upstream tree. It does not
download weights, install packages, or start a GPU, simulator, trainer, or server.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / '.scratch' / 'lingbot_vla2_g1'
UPSTREAM = SCRATCH / 'upstream'
OUTPUT_DIR = SCRATCH / 'official_baseline_preparation'
VENV_PYTHON = SCRATCH / 'venv' / 'bin' / 'python'
BASE_MANIFEST = SCRATCH / 'remote_model_manifest.json'
EXPERIMENT_README = ROOT / 'experiments' / 'lingbot_vla2_g1' / 'README.md'

SCHEMA = 'lingbot_vla2_official_robotwin_baseline_preparation_v1'
UPSTREAM_COMMIT = 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6'
ROBOTWIN_REVISION = '13c3c47ff4312dd62484bcd51be034af55c062d1'
GENERIC_BASE_REPO = 'robbyant/lingbot-vla-v2-6b'
GENERIC_BASE_REVISION = '11c703bf6a5c1f45b3b69168482da11fdbba53d7'
ROBOTWIN_REPO = 'robbyant/lingbot-vla-v2-6b-robotwin'
HF_CKPT = 'checkpoints/global_step_50000/hf_ckpt'
QWEN_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
BUDGET_BYTES = 20 * 1024 * 1024
MAX_SMALL_FILE = 1024 * 1024
HISTORICAL_OWN_CAP_BYTES = 25 * 1024 ** 3
STORAGE_COMMITMENT = SCRATCH / 'storage_continuation_R4_v1.json'
PERMANENT_BACKUP = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
EXPECTED_SHARDS = tuple(f'model-{index:05d}-of-00006.safetensors' for index in range(1, 7))
BASE_SHARD_SHA256 = {
    'model-00001-of-00006.safetensors': '4afb52b06a13df8b738a156ae5c8196d3bfe6b3ca931cecbd701e44cb9674e45',
    'model-00002-of-00006.safetensors': 'ec131afa26a340db94c0dba8ec00e990be5f3d842ce6532070f0c8e26a067501',
    'model-00003-of-00006.safetensors': '7dccb068ca66c11fa514476d64661eeead56e2e80e1c0572c9ea82aa0d9ecf27',
    'model-00004-of-00006.safetensors': '1c2cb78066b69ae11255db851df95071e2cd69a3a4bc5020b3fd3b13b17819fb',
    'model-00005-of-00006.safetensors': '8fe36bf1f4f617869954bdfa1ad12e16abd8b0235a5fa99c88d571d4cddf4a17',
    'model-00006-of-00006.safetensors': '3cf613d592dad64b1e2a1b1bb34a6f556a7fa05eacdb9972d73e0ea4882555a0',
}
FORBIDDEN_SUFFIXES = (
    '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.zip', '.whl', '.gguf', '.onnx',
)
HF_SMALL_FILES = (
    'README.md',
    'lingbotvla_cli.yaml',
    'checkpoints/global_step_50000/hf_ckpt/config.json',
    'checkpoints/global_step_50000/hf_ckpt/model.safetensors.index.json',
)
ROBOTWIN_RAW_FILES = (
    'README.md',
    'script/_install.sh',
    'script/requirements.txt',
    'script/_download_assets.sh',
    'assets/_download.py',
    'task_config/_camera_config.yml',
    'task_config/demo_clean.yml',
    'task_config/demo_randomized.yml',
)
PYPI_URLS = {
    'sapien': 'https://pypi.org/pypi/sapien/3.0.0b1/json',
    'mplib': 'https://pypi.org/pypi/mplib/0.2.1/json',
    'open3d': 'https://pypi.org/pypi/open3d/0.18.0/json',
    'warp-lang': 'https://pypi.org/pypi/warp-lang/1.12.0/json',
    'torch': 'https://pypi.org/pypi/torch/2.4.1/json',
}
ASSET_ZIPS = ('background_texture.zip', 'embodiments.zip', 'objects.zip')
CAMERA_ORIGINS = (
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
)
HEADERS = {'User-Agent': 'lingbot-official-baseline-prep', 'Accept': 'application/json'}


def urlopen(request, timeout=60):
    return urllib.request.urlopen(request, timeout=timeout)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def base_shards(manifest: dict) -> dict:
    if manifest.get('id') != GENERIC_BASE_REPO and manifest.get('modelId') != GENERIC_BASE_REPO:
        raise ValueError('local manifest is not the generic base repo')
    if manifest.get('sha') != GENERIC_BASE_REVISION:
        raise ValueError('local generic base revision differs from the pinned base')
    found = {}
    for item in manifest['siblings']:
        name = item['rfilename']
        if name in EXPECTED_SHARDS:
            found[name] = item['lfs']['sha256']
    if found != BASE_SHARD_SHA256:
        raise ValueError('local generic base shard hashes differ')
    return found


def admit_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get('repo_id') != ROBOTWIN_REPO:
        raise ValueError('generic base weights are not the RoboTwin post-training checkpoint')
    revision = checkpoint.get('revision') or ''
    if revision == GENERIC_BASE_REVISION or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('generic base weights revision cannot lock the RoboTwin checkpoint')
    if checkpoint.get('hf_ckpt') != HF_CKPT:
        raise ValueError('official model_path is checkpoints/global_step_50000/hf_ckpt')
    if checkpoint.get('weights_downloaded') is not False:
        raise ValueError('checkpoint weights must stay downloaded=false in this entry')
    shards = checkpoint.get('shards') or []
    names = [item.get('name') for item in shards]
    if tuple(names) != EXPECTED_SHARDS:
        raise ValueError('hf_ckpt shard set is not the official six safetensors files')
    for item in shards:
        digest = item.get('sha256') or ''
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('shard sha256 missing')
        if digest == BASE_SHARD_SHA256[item['name']]:
            raise ValueError('generic base weights hash matches a RoboTwin shard')
        if not isinstance(item.get('size'), int) or item['size'] <= 0:
            raise ValueError('shard size missing')
    index_names = checkpoint.get('index_shards')
    if index_names is not None and tuple(index_names) != EXPECTED_SHARDS:
        raise ValueError('safetensors index does not list the hf_ckpt shards')


def admit_precision(use_bf16, use_fp32) -> None:
    if use_bf16 is not False or use_fp32 is not True:
        raise ValueError('release validation requires FP32 inference, not BF16')


def admit_robotwin_revision(revision: str) -> None:
    if revision != ROBOTWIN_REVISION:
        raise ValueError(f'official RoboTwin revision is {ROBOTWIN_REVISION}')


def robotwin_revision_from_readme(text: str) -> str:
    found = re.findall(r'^git checkout ([0-9a-f]{40})$', text, re.M)
    if found != [ROBOTWIN_REVISION]:
        raise ValueError(f'official RoboTwin revision pin is {ROBOTWIN_REVISION}, found {found}')
    return ROBOTWIN_REVISION


def policy_argument_defaults(source: str) -> dict:
    found = {}
    for flag in ('--use_bf16', '--use_fp32'):
        match = re.search(rf'"{flag}"[\s\S]{{0,180}}?default=(True|False)', source)
        if match is None:
            raise ValueError(f'missing {flag} default in deploy policy')
        found[flag[2:]] = match.group(1) == 'True'
    return found


def launcher_fp32_defaults(source: str) -> None:
    if re.search(r'^use_bf16=False$', source, re.M) is None:
        raise ValueError('launcher does not default to FP32 (use_bf16=False)')
    if re.search(r'^use_fp32=True$', source, re.M) is None:
        raise ValueError('launcher does not default to FP32 (use_fp32=True)')


def assert_model_path(path: str) -> None:
    normalized = path.rstrip('/')
    if not normalized.endswith('checkpoints/global_step_50000/hf_ckpt'):
        raise ValueError('model_path must be checkpoints/global_step_50000/hf_ckpt')
    if 'lingbot-vla-v2-6b' in Path(normalized).parts:
        raise ValueError('generic base weights path')


def admit_command(command: dict) -> None:
    if command.get('execution') != 'NOT_EXECUTED':
        raise ValueError('FP32 command must be marked NOT_EXECUTED')
    argv = command.get('argv') or []
    if '--use_bf16' not in argv or '--use_fp32' not in argv:
        raise ValueError('FP32 command must set use_bf16 and use_fp32 explicitly')
    flags = {argv[index]: argv[index + 1] for index in range(len(argv) - 1) if argv[index].startswith('--')}
    if flags.get('--use_bf16') != 'False' or flags.get('--use_fp32') != 'True':
        raise ValueError('BF16 is not the published RoboTwin release setting')
    assert_model_path(flags.get('--model_path', ''))
    if flags.get('--task_config') not in {'demo_clean', 'demo_randomized'}:
        raise ValueError('task_config must be demo_clean or demo_randomized')


def training_yaml_for_model(model_path: Path) -> Path:
    return Path(model_path).parent.parent.parent / 'lingbotvla_cli.yaml'


def documented_flash_attn_wheel(readme: str) -> str:
    match = re.search(r'flash_attn-2\.8\.3\+cu12torch2\.8cxx11abiTRUE-cp312-cp312-linux_x86_64\.whl', readme)
    if match is None:
        raise ValueError('official README no longer documents the x86_64 flash-attn wheel')
    return match.group(0)


def current_storage_report(checkpoint_bytes, asset_zip_bytes, *,
                           commitment_path=STORAGE_COMMITMENT,
                           scratch=SCRATCH, backup=PERMANENT_BACKUP) -> dict:
    """Use the current combined commitment; README's old cap is historical."""
    document = json.loads(Path(commitment_path).read_text())
    cap = document['combined_cap_bytes']
    growth = document['prospective_remaining_growth_commitment_bytes']
    if type(cap) is not int or type(growth) is not int or not 0 < growth <= cap:
        raise ValueError('Current combined storage commitment is invalid')
    sizes = {str(root): sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
             for root in (scratch, backup)}
    combined = sum(sizes.values())
    return {
        'current_combined_commitment_path': str(commitment_path),
        'current_combined_commitment_sha256': sha256_file(Path(commitment_path)),
        'combined_cap_bytes': cap, 'owned_and_backup_current_bytes': sizes,
        'combined_current_bytes': combined, 'existing_future_growth_commitment_bytes': growth,
        'checkpoint_fits_before_future_growth': combined + checkpoint_bytes <= cap,
        'checkpoint_fits_with_existing_future_commitment': combined + growth + checkpoint_bytes <= cap,
        'baseline_minimum_additional_bytes': checkpoint_bytes + asset_zip_bytes,
        'baseline_compressed_files_fit_with_existing_future_commitment':
            combined + growth + checkpoint_bytes + asset_zip_bytes <= cap,
        'asset_unpacked_bytes': None, 'asset_unpacked_size_measured': False,
        'historical_README_own_cap_bytes': HISTORICAL_OWN_CAP_BYTES,
        'historical_README_own_cap_is_current_authority': False,
        'scope': 'Prospective local storage estimate; not a GPU-memory limit or an installation feasibility verdict.',
    }


def classify_release_files(filenames) -> dict:
    linux_arm, linux_x86, other, sdists = [], [], [], []
    for name in filenames:
        lower = name.lower()
        if lower.endswith('.whl'):
            if 'manylinux' in lower and 'aarch64' in lower:
                linux_arm.append(name)
            elif 'manylinux' in lower and 'x86_64' in lower:
                linux_x86.append(name)
            else:
                other.append(name)
        elif lower.endswith(('.tar.gz', '.zip')):
            sdists.append(name)
        else:
            other.append(name)
    return {
        'linux_aarch64_wheels': linux_arm,
        'linux_x86_64_wheels': linux_x86,
        'other_files': other,
        'sdists': sdists,
        'direct_linux_aarch64_wheel': bool(linux_arm),
        'direct_linux_x86_64_wheel': bool(linux_x86),
        'linux_aarch64_cpython_tags': sorted({f'cp{match.group(1)}' for name in linux_arm if (match := re.search(r'cp(\d+)', name))}),
    }


def aarch64_pip_obstacle(report: dict) -> str | None:
    if report['direct_linux_aarch64_wheel']:
        return None
    if report['sdists']:
        return 'no_linux_aarch64_wheel_sdist_untested'
    return 'no_linux_aarch64_wheel_and_no_sdist'


def index_shard_names(index: dict) -> list:
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('safetensors index has no weight_map')
    return sorted(set(weight_map.values()))


def camera_selection(demo: dict, cameras: dict) -> dict:
    chosen = demo['camera']
    selected = {}
    for key in ('head_camera_type', 'wrist_camera_type'):
        name = chosen[key]
        spec = cameras[name]
        selected[key] = {'type': name, 'fovy': spec['fovy'], 'h': spec['h'], 'w': spec['w']}
    return selected


def assert_url_allowed(url: str) -> None:
    if url.startswith(f'https://huggingface.co/{ROBOTWIN_REPO}/resolve/'):
        filename = url.split('/resolve/', 1)[1].split('/', 1)[1]
        if filename not in HF_SMALL_FILES or any(filename.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise RuntimeError(f'refusing download {url}')
        return
    allowed = {
        f'https://huggingface.co/api/models/{ROBOTWIN_REPO}?blobs=true',
        'https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0?blobs=true',
        f'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/commits/{ROBOTWIN_REVISION}',
        f'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/contents/INSTALLATION.md?ref={ROBOTWIN_REVISION}',
        'https://raw.githubusercontent.com/NVlabs/curobo/v0.7.8/setup.py',
        'https://robotwin-platform.github.io/doc/usage/robotwin-install.html',
        *PYPI_URLS.values(),
        *(f'https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/{ROBOTWIN_REVISION}/{name}' for name in ROBOTWIN_RAW_FILES),
    }
    if url not in allowed:
        raise RuntimeError(f'refusing download {url}')


class DownloadBudget:
    def __init__(self, limit=BUDGET_BYTES):
        self.limit = limit
        self.used = 0
        self.entries = []

    def add(self, name: str, data: bytes) -> None:
        if any(name.lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise RuntimeError(f'refusing download {name}')
        if len(data) > MAX_SMALL_FILE or self.used + len(data) > self.limit:
            raise RuntimeError(f'download budget exceeded by {name}')
        self.used += len(data)
        self.entries.append({'name': name, 'bytes': len(data), 'sha256': sha256_bytes(data)})


def fetch(url: str, name: str, budget: DownloadBudget, directory: Path) -> bytes:
    assert_url_allowed(url)
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urlopen(request, timeout=60) as response:
            declared = response.headers.get('Content-Length')
            if declared is not None and int(declared) > MAX_SMALL_FILE:
                raise RuntimeError(f'{name} declares {declared} bytes')
            data = response.read(MAX_SMALL_FILE + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and name == 'INSTALLATION.md':
            budget.entries.append({'name': name, 'bytes': 0, 'http_status': 404})
            return b''
        raise
    if len(data) > MAX_SMALL_FILE:
        raise RuntimeError(f'{name} exceeds the small-file cap')
    budget.add(name, data)
    destination = directory / name.replace('/', '_')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.incomplete')
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return data


def directory_bytes(root: Path, skip_names=frozenset({'venv'})) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in skip_names and not os.path.islink(os.path.join(dirpath, name))]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
    return total


def installed_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_requirement_line(line: str) -> dict:
    name = re.match(r'([A-Za-z0-9_.-]+)', line).group(1)
    version = installed_version(name)
    pinned = re.search(r'==\s*([^\s#]+)', line)
    if version is None:
        status = 'not_installed'
    elif pinned and not version.split('+')[0] == pinned.group(1):
        status = 'version_mismatch'
    elif pinned:
        status = 'pin_matches'
    else:
        status = 'present_unpinned'
    return {'requirement': line, 'distribution': name, 'installed': version, 'status': status}


def local_requirement_report(text: str) -> list:
    rows = []
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if line:
            rows.append(probe_requirement_line(line))
    return rows


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    distribution = {'yaml': 'PyYAML'}.get(name, name)
    return {
        'module': name,
        'importable': spec is not None,
        'version': installed_version(distribution) if spec is not None else None,
    }


def html_text(html: str) -> str:
    text = re.sub(r'<script[\s\S]*?</script>', ' ', html)
    text = re.sub(r'<style[\s\S]*?</style>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    for source, target in (('&nbsp;', ' '), ('&lt;', '<'), ('&gt;', '>'), ('&amp;', '&'), ('&quot;', '"')):
        text = text.replace(source, target)
    return re.sub(r'\s+', ' ', text).strip()


def collapsed(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


def require_text(text: str, needle: str) -> str:
    if needle not in collapsed(text):
        raise ValueError(f'official text is missing {needle!r}')
    return needle


def git_identity(path: Path) -> dict:
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
    if commit != UPSTREAM_COMMIT or dirty.strip():
        raise ValueError(f'upstream checkout is not clean {UPSTREAM_COMMIT}')
    remote = subprocess.check_output(['git', '-C', str(path), 'remote', 'get-url', 'origin'], text=True).strip()
    return {'path': str(path), 'commit': commit, 'dirty': False, 'origin': remote}


def file_record(path: Path) -> dict:
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def local_config_hashes(upstream: Path) -> dict:
    robot = upstream / 'configs' / 'robot_configs' / 'robotwin.yaml'
    norm = upstream / 'assets' / 'norm_stats' / 'robotwin.json'
    robot_text = robot.read_text()
    for origin in CAMERA_ORIGINS:
        require_text(robot_text, origin)
    norm_text = norm.read_text()
    for key in ('action.arm.position', 'action.effector.position'):
        require_text(norm_text, key)
    parsed = yaml.safe_load(robot_text)
    images = {item_key: item_value['origin_keys'] for item in parsed['images'] for item_key, item_value in item.items()}
    return {
        'robot_config': {**file_record(robot), 'image_origin_keys': images, 'norm_stats': parsed['norm_stats']},
        'action_norm_stats': file_record(norm),
    }


def fp32_command(task_config: str) -> dict:
    model_path = '/NOT_PRESENT/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt'
    argv = [
        'bash', 'experiment/robotwin/start_robotwin_infer_and_eval.sh',
        '--model_path', model_path,
        '--eval_workdir', '/NOT_PRESENT/RoboTwin',
        '--output_base', '/NOT_PRESENT/robotwin_eval_output',
        '--task_config', task_config,
        '--use_bf16', 'False',
        '--use_fp32', 'True',
        '--use_compile', 'True',
        '--use_length', '50',
        '--robo_name', 'robotwin',
        '--num_tasks', '50',
        '--num_gpus', '1',
        '--num_per_gpu', '1',
        '--conda_sh', '/NOT_PRESENT/conda.sh',
        '--inference_env', 'lingbotvla',
        '--sim_env', 'RoboTwin',
    ]
    environment = {
        'QWEN3VL_PATH': str(SCRATCH / 'qwen_processor'),
        'EVAL_WORKDIR': '/NOT_PRESENT/RoboTwin',
    }
    shell = f"cd {shlex.quote(str(UPSTREAM))} && " + ' '.join(f'{key}={shlex.quote(value)}' for key, value in environment.items()) + ' ' + shlex.join(argv)
    command = {
        'execution': 'NOT_EXECUTED',
        'cwd': str(UPSTREAM),
        'environment': environment,
        'argv': argv,
        'shell': shell,
        'paths_exist': False,
        'published_success_rate_reproduced': False,
    }
    admit_command(command)
    return command


def server_argv() -> dict:
    argv = [
        'python', '-m', 'deploy.lingbot_vla_v2_policy',
        '--model_path', '/NOT_PRESENT/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt',
        '--use_length', '50',
        '--use_bf16', 'False',
        '--use_fp32', 'True',
        '--use_compile', 'True',
        '--port', '9330',
    ]
    command = {'execution': 'NOT_EXECUTED', 'argv': argv, 'published_success_rate_reproduced': False}
    flags = {argv[index]: argv[index + 1] for index in range(len(argv) - 1) if argv[index].startswith('--')}
    admit_precision(flags['--use_bf16'] == 'True', flags['--use_fp32'] == 'True')
    assert_model_path(flags['--model_path'])
    return command


def seal(document: dict) -> dict:
    if document.get('success_rate_reproduced') is not False:
        raise ValueError('official success rate was not reproduced')
    if document.get('official_baseline_execution') != 'NOT_EXECUTED':
        raise ValueError('official baseline execution must be NOT_EXECUTED')
    if document.get('runnable_on_this_host') is not False:
        raise ValueError('this entry cannot claim the official baseline runs')
    if any(document.get(flag) for flag in ('gpu_started', 'simulator_started', 'training_started', 'service_started', 'weights_downloaded', 'packages_installed')):
        raise ValueError('this entry must not start work or download weights')
    for command in document.get('next_commands') or []:
        if command.get('status') != 'NOT_EXECUTED':
            raise ValueError('next command must be NOT_EXECUTED')
    download_bytes = document.get('download_bytes', 0)
    if download_bytes > BUDGET_BYTES:
        raise ValueError('download budget exceeded')
    for entry in document.get('downloads') or []:
        if entry.get('bytes', 0) > MAX_SMALL_FILE:
            raise ValueError('small-file cap exceeded')
        if any(str(entry.get('name', '')).lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            raise ValueError('forbidden download recorded')
    if document.get('preparation_probe') == 'NOT_EXECUTED':
        if 'checkpoint' in document or 'content_hashes' in document:
            raise ValueError('dry run cannot carry a locked checkpoint or content hashes')
        return document
    if document.get('preparation_probe') != 'completed':
        raise ValueError('preparation probe must be completed or NOT_EXECUTED')
    admit_checkpoint(document['checkpoint'])
    admit_precision(document['precision']['use_bf16'], document['precision']['use_fp32'])
    admit_robotwin_revision(document['robotwin']['revision'])
    admit_command(document['fp32_command'])
    admit_command(document['fp32_command_randomized'])
    return document


def build_dry_run(output_dir: Path) -> dict:
    return seal({
        'schema': SCHEMA,
        'preparation_probe': 'NOT_EXECUTED',
        'official_baseline_execution': 'NOT_EXECUTED',
        'success_rate_reproduced': False,
        'published_success_rates_are_local_measurements': False,
        'runnable_on_this_host': False,
        'gpu_started': False,
        'simulator_started': False,
        'training_started': False,
        'service_started': False,
        'weights_downloaded': False,
        'packages_installed': False,
        'download_bytes': 0,
        'downloads': [],
        'next_commands': [],
        'output_dir': str(output_dir),
        'note': 'Dry run writes this marker only. It does not lock a revision or claim a RoboTwin result.',
    })


def checkpoint_from_api(payload: dict, index: dict) -> dict:
    if payload.get('id') != ROBOTWIN_REPO and payload.get('modelId') != ROBOTWIN_REPO:
        raise ValueError('Hugging Face payload is not the RoboTwin checkpoint repo')
    revision = payload.get('sha') or ''
    siblings = {item['rfilename']: item for item in payload['siblings']}
    prefix = HF_CKPT + '/'
    shards = []
    for name in EXPECTED_SHARDS:
        item = siblings[prefix + name]
        lfs = item.get('lfs') or {}
        shards.append({'name': name, 'rfilename': prefix + name, 'size': item['size'], 'sha256': lfs['sha256']})
    members = []
    for name, item in sorted(siblings.items()):
        if name == 'lingbotvla_cli.yaml' or name.startswith(prefix):
            record = {'rfilename': name, 'size': item.get('size'), 'lfs_sha256': (item.get('lfs') or {}).get('sha256')}
            record['loaded_by_deploy_policy'] = name.startswith(prefix) and name.endswith('.safetensors') or name == 'lingbotvla_cli.yaml'
            members.append(record)
    checkpoint = {
        'repo_id': ROBOTWIN_REPO,
        'revision': revision,
        'hf_ckpt': HF_CKPT,
        'resolve_root': f'https://huggingface.co/{ROBOTWIN_REPO}/resolve/{revision}',
        'weights_downloaded': False,
        'shards': shards,
        'index_shards': index_shard_names(index),
        'weight_bytes': sum(item['size'] for item in shards),
        'repo_used_storage_bytes': payload.get('usedStorage'),
        'sibling_count': len(siblings),
        'members': members,
        'rejected_generic_base': {'repo_id': GENERIC_BASE_REPO, 'revision': GENERIC_BASE_REVISION},
    }
    admit_checkpoint(checkpoint)
    return checkpoint


def pypi_report(payload: dict) -> dict:
    filenames = [item['filename'] for item in payload['urls']]
    report = classify_release_files(filenames)
    report['version'] = payload['info']['version']
    report['obstacle'] = aarch64_pip_obstacle(report)
    return report


def asset_records(payload: dict) -> dict:
    siblings = {item['rfilename']: item for item in payload['siblings']}
    records = []
    for name in ASSET_ZIPS:
        item = siblings[name]
        records.append({'rfilename': name, 'size': item['size'], 'sha256': item['lfs']['sha256'], 'downloaded': False})
    return {
        'repo_id': 'TianxingChen/RoboTwin2.0',
        'observed_revision': payload.get('sha'),
        'revision_pinned_by_lingbot': False,
        'used_storage_bytes': payload.get('usedStorage'),
        'files': records,
        'bytes': sum(item['size'] for item in records),
        'unpacked_bytes': None,
        'unpacked_measured': False,
    }


def next_commands(revision: str, robotwin_revision: str) -> list:
    download = (
        'huggingface-cli download robbyant/lingbot-vla-v2-6b-robotwin '
        f'--revision {revision} --local-dir /NOT_PRESENT/lingbot-vla-v2-6b-robotwin'
    )
    clone = (
        'git clone https://github.com/RoboTwin-Platform/RoboTwin.git /NOT_PRESENT/RoboTwin && '
        f'git -C /NOT_PRESENT/RoboTwin checkout {robotwin_revision}'
    )
    assets = 'cd /NOT_PRESENT/RoboTwin && bash script/_download_assets.sh'
    return [
        {'name': 'download_robotwin_checkpoint', 'status': 'NOT_EXECUTED', 'shell': download, 'writes_weights': True},
        {'name': 'checkout_robotwin', 'status': 'NOT_EXECUTED', 'shell': clone},
        {'name': 'download_robotwin_assets', 'status': 'NOT_EXECUTED', 'shell': assets, 'writes_asset_zips': True},
        {'name': 'fp32_clean_eval', 'status': 'NOT_EXECUTED', 'shell': fp32_command('demo_clean')['shell']},
        {'name': 'fp32_randomized_eval', 'status': 'NOT_EXECUTED', 'shell': fp32_command('demo_randomized')['shell']},
    ]


def build_preparation(output_dir: Path) -> dict:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sources = output_dir / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    budget = DownloadBudget()
    identity = git_identity(UPSTREAM)
    base_shards(json.loads(BASE_MANIFEST.read_text()))
    experiment_readme = EXPERIMENT_README.read_text()
    upstream_readme = (UPSTREAM / 'README.md').read_text()
    robotwin_readme = (UPSTREAM / 'experiment' / 'robotwin' / 'README.md').read_text()
    policy_source = (UPSTREAM / 'deploy' / 'lingbot_vla_v2_policy.py').read_text()
    launcher_source = (UPSTREAM / 'experiment' / 'robotwin' / 'start_robotwin_infer_and_eval.sh').read_text()
    robotwin_revision_from_readme(robotwin_readme)
    launcher_fp32_defaults(launcher_source)
    policy_defaults = policy_argument_defaults(policy_source)
    flash_wheel = documented_flash_attn_wheel(upstream_readme)
    require_text(upstream_readme, 'Python 3.12')
    require_text(upstream_readme, 'PyTorch 2.8.0')
    require_text(robotwin_readme, 'Release validation of the published checkpoint uses **FP32 inference**')
    require_text(robotwin_readme, 'one FP32 policy server plus its simulator uses roughly 32 GB')
    require_text(robotwin_readme, 'which does not reproduce the published FP32 benchmark')
    config_hashes = local_config_hashes(UPSTREAM)

    api = json.loads(fetch(f'https://huggingface.co/api/models/{ROBOTWIN_REPO}?blobs=true', 'hf_model_api.json', budget, sources))
    revision = api['sha']
    small = {}
    for name in HF_SMALL_FILES:
        small[name] = fetch(f'https://huggingface.co/{ROBOTWIN_REPO}/resolve/{revision}/{name}', name, budget, sources)
    index = json.loads(small['checkpoints/global_step_50000/hf_ckpt/model.safetensors.index.json'])
    checkpoint = checkpoint_from_api(api, index)
    training_yaml = yaml.safe_load(small['lingbotvla_cli.yaml'])
    hf_config = json.loads(small['checkpoints/global_step_50000/hf_ckpt/config.json'])

    commit_payload = json.loads(fetch(
        f'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/commits/{ROBOTWIN_REVISION}',
        'robotwin_commit.json', budget, sources,
    ))
    if commit_payload.get('sha') != ROBOTWIN_REVISION:
        raise ValueError('GitHub RoboTwin commit does not match the pinned revision')
    fetch(
        f'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/contents/INSTALLATION.md?ref={ROBOTWIN_REVISION}',
        'INSTALLATION.md', budget, sources,
    )
    raw = {}
    for name in ROBOTWIN_RAW_FILES:
        raw[name] = fetch(
            f'https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/{ROBOTWIN_REVISION}/{name}',
            'robotwin_' + name,
            budget, sources,
        )
    curobo_setup = fetch('https://raw.githubusercontent.com/NVlabs/curobo/v0.7.8/setup.py', 'curobo_v0.7.8_setup.py', budget, sources)
    install_html = fetch('https://robotwin-platform.github.io/doc/usage/robotwin-install.html', 'robotwin_install.html', budget, sources)
    install_text = html_text(install_html.decode())
    for needle in (
        'Python 3.10',
        '12.1 (Recommended)',
        'Most constraints stem from what the SAPIEN package is capable of supporting.',
        'Linux / NVIDIA GPU',
    ):
        require_text(install_text, needle)
    require_text(install_text, 'scripts/_install.sh')
    download_py = raw['assets/_download.py'].decode()
    for needle in ('allow_patterns', 'background_texture.zip', 'embodiments.zip', 'objects.zip'):
        require_text(download_py, needle)
    pypi = {}
    for package, url in PYPI_URLS.items():
        pypi[package] = pypi_report(json.loads(fetch(url, f'pypi_{package}.json', budget, sources)))
    dataset = asset_records(json.loads(fetch(
        'https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0?blobs=true',
        'robotwin_dataset_api.json', budget, sources,
    )))

    cameras = yaml.safe_load(raw['task_config/_camera_config.yml'])
    clean = yaml.safe_load(raw['task_config/demo_clean.yml'])
    randomized = yaml.safe_load(raw['task_config/demo_randomized.yml'])
    content_hashes = {
        'lingbot_robot_config': config_hashes['robot_config'],
        'lingbot_action_norm_stats': config_hashes['action_norm_stats'],
        'robotwin_camera_config': {'sha256': sha256_bytes(raw['task_config/_camera_config.yml']), 'bytes': len(raw['task_config/_camera_config.yml'])},
        'robotwin_demo_clean': {'sha256': sha256_bytes(raw['task_config/demo_clean.yml']), 'bytes': len(raw['task_config/demo_clean.yml']), 'cameras': camera_selection(clean, cameras)},
        'robotwin_demo_randomized': {'sha256': sha256_bytes(raw['task_config/demo_randomized.yml']), 'bytes': len(raw['task_config/demo_randomized.yml']), 'cameras': camera_selection(randomized, cameras)},
    }
    qwen = SCRATCH / 'qwen_processor'
    qwen_files = sorted(path.name for path in qwen.iterdir() if path.is_file()) if qwen.is_dir() else []
    scratch_bytes = directory_bytes(SCRATCH)
    machine = platform.machine()
    sim_requirements = raw['script/requirements.txt'].decode()
    inference_requirements = (UPSTREAM / 'requirements.txt').read_text()
    gaps = []
    if machine == 'aarch64':
        for package in ('sapien', 'mplib', 'open3d', 'warp-lang', 'torch'):
            obstacle = pypi[package]['obstacle']
            if obstacle:
                gaps.append({
                    'component': package,
                    'version': pypi[package]['version'],
                    'obstacle': obstacle,
                    'linux_aarch64_wheels': len(pypi[package]['linux_aarch64_wheels']),
                    'sdists': len(pypi[package]['sdists']),
                    'blocks_direct_pip_wheel': True,
                    'source_build': 'NOT_EXECUTED',
                })
    if b'CUDAExtension' not in curobo_setup:
        raise ValueError('pinned curobo setup.py no longer builds CUDA extensions')
    gaps.append({
        'component': 'curobo',
        'version': 'v0.7.8',
        'evidence': 'NVlabs/curobo setup.py CUDAExtension and script/_install.sh clone --branch v0.7.8',
        'source_build': 'NOT_EXECUTED',
        'blocks_direct_binary_install': True,
    })
    interpreter_ok = Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    document = {
        'schema': SCHEMA,
        'preparation_probe': 'completed',
        'official_baseline_execution': 'NOT_EXECUTED',
        'success_rate_reproduced': False,
        'published_success_rates_are_local_measurements': False,
        'published_success_rates': {
            'source': 'upstream README RoboTwin 2.0 table and the robotwin checkpoint model card',
            'clean': '93.52%',
            'randomized': '92.80%',
            'local_status': 'NOT_EXECUTED',
        },
        'runnable_on_this_host': False,
        'gpu_started': False,
        'simulator_started': False,
        'training_started': False,
        'service_started': False,
        'weights_downloaded': False,
        'packages_installed': False,
        'upstream': identity,
        'checkpoint': checkpoint,
        'hf_config_json': hf_config,
        'training_yaml_loaded_by_policy': {
            'relative_to_hf_ckpt': '../../../lingbotvla_cli.yaml',
            'sha256': sha256_bytes(small['lingbotvla_cli.yaml']),
            'norm_stats_file': training_yaml['data'].get('norm_stats_file'),
            'tokenizer_path': training_yaml['model'].get('tokenizer_path'),
            'attention_implementation': training_yaml['model'].get('attn_implementation'),
            'deploy_overrides_attention_to': 'eager',
            'train_enable_fp32': training_yaml['train'].get('enable_fp32'),
            'train_enable_mixed_precision': training_yaml['train'].get('enable_mixed_precision'),
            'inference_precision_comes_from': 'launcher --use_bf16 False --use_fp32 True',
        },
        'robotwin': {
            'repo': 'https://github.com/RoboTwin-Platform/RoboTwin.git',
            'revision': ROBOTWIN_REVISION,
            'github_commit_sha': commit_payload['sha'],
            'installation_md_at_pin': 'absent_http_404',
            'install_script': 'script/_install.sh',
            'live_doc_install_script': 'scripts/_install.sh',
            'live_doc_differs_from_pin_path': True,
            'checkout_present': False,
        },
        'precision': {
            'use_bf16': False,
            'use_fp32': True,
            'policy_module_default_use_bf16': policy_defaults['use_bf16'],
            'policy_module_default_use_fp32': policy_defaults['use_fp32'],
            'policy_module_defaults_accepted_as_release': False,
            'launcher_defaults': {'use_bf16': False, 'use_fp32': True},
        },
        'fp32_command': fp32_command('demo_clean'),
        'fp32_command_randomized': fp32_command('demo_randomized'),
        'fp32_server_argv': server_argv(),
        'documented_example_num_gpus': 8,
        'single_host_command_num_gpus': 1,
        'official_fp32_memory_note': 'one FP32 policy server plus its simulator uses roughly 32 GB',
        'content_hashes': content_hashes,
        'dependencies': {
            'lingbot_inference': {
                'python': '3.12',
                'torch': '2.8.0',
                'requirements': local_requirement_report(inference_requirements),
                'flash_attn_documented_wheel': flash_wheel,
                'flash_attn_wheel_platform': 'linux_x86_64',
            },
            'robotwin_sim': {
                'documented_python': '3.10',
                'documented_cuda': '12.1',
                'requirements': local_requirement_report(sim_requirements),
                'install_script_extras': ['pytorch3d git+https://github.com/facebookresearch/pytorch3d.git@stable', 'curobo v0.7.8', 'warp-lang==1.12.0'],
                'pypi': {
                    name: {key: value[key] for key in (
                        'version', 'obstacle', 'direct_linux_aarch64_wheel', 'linux_aarch64_wheels',
                        'linux_x86_64_wheels', 'sdists', 'linux_aarch64_cpython_tags',
                    )}
                    for name, value in pypi.items()
                },
            },
        },
        'host': {
            'machine': machine,
            'platform': platform.platform(),
            'python': sys.version,
            'executable': sys.executable,
            'isolated_lingbot_venv': interpreter_ok,
            'modules': [module_status(name) for name in ('torch', 'transformers', 'numpy', 'safetensors', 'yaml', 'websockets', 'flash_attn', 'sapien', 'mplib', 'curobo', 'open3d', 'warp', 'pytorch3d')],
            'path_tools': {name: shutil.which(name) for name in ('conda', 'nvcc', 'ffmpeg', 'vulkaninfo')},
            'qwen_processor_dir': str(qwen),
            'qwen_processor_revision': QWEN_REVISION,
            'qwen_processor_files': qwen_files,
            'qwen_weight_shards_present': any(name.endswith('.safetensors') for name in qwen_files),
            'qwen_processor_execution': 'NOT_EXECUTED',
        },
        'arm64_gaps': gaps,
        'storage_estimate': {
            'checkpoint_safetensors_bytes': checkpoint['weight_bytes'],
            'checkpoint_repo_used_storage_bytes': checkpoint['repo_used_storage_bytes'],
            'robotwin_asset_zips': dataset,
            'qwen_full_weights_required_by_loader': False,
            'existing_scratch_non_venv_bytes': scratch_bytes,
            **current_storage_report(checkpoint['weight_bytes'], dataset['bytes']),
            'host_filesystem_avail_bytes': shutil.disk_usage(ROOT).free,
        },
        'downloads': budget.entries,
        'download_bytes': budget.used,
        'download_budget_bytes': BUDGET_BYTES,
        'next_commands': next_commands(revision, ROBOTWIN_REVISION),
    }
    # The policy module's own defaults are BF16. Recording them must not admit them.
    try:
        admit_precision(policy_defaults['use_bf16'], policy_defaults['use_fp32'])
    except ValueError:
        document['precision']['policy_module_defaults_rejected'] = True
    else:
        raise ValueError('policy defaults were accepted as the FP32 release setting')
    return seal(document)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Prepare the official LingBot-VLA 2.0 RoboTwin baseline without executing it.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--refresh-storage', action='store_true',
                        help='Recheck the current local storage commitment using existing locked metadata only.')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    destination = args.output / 'preparation.json'
    if args.refresh_storage:
        if args.dry_run:
            parser.error('--refresh-storage and --dry-run are mutually exclusive')
        document = seal(json.loads(destination.read_text()))
        storage = document['storage_estimate']
        storage.pop('documented_own_storage_cap_bytes', None)
        storage.pop('checkpoint_fits_in_remaining_own_cap', None)
        storage.update(current_storage_report(document['checkpoint']['weight_bytes'],
                                              storage['robotwin_asset_zips']['bytes']))
        document['storage_refresh_network_download_bytes'] = 0
    else:
        document = build_dry_run(args.output) if args.dry_run else build_preparation(args.output)
    write_json(destination, document)
    print(json.dumps({
        'preparation_probe': document['preparation_probe'],
        'official_baseline_execution': document['official_baseline_execution'],
        'success_rate_reproduced': document['success_rate_reproduced'],
        'path': str(destination),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
