"""Run frozen cases serially; infrastructure errors stop instead of scoring 0."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--replay-result', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--policy', choices=['baseline', 'finetuned'], required=True)
    ap.add_argument('--norm-stats', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-cases', type=int)
    args = ap.parse_args()
    weights = args.checkpoint / 'model.safetensors'
    identity = sha256(weights)
    args.output.mkdir(parents=True, exist_ok=True)
    configs = args.output / 'configs'
    configs.mkdir(exist_ok=True)
    cases = json.loads(args.cases.read_text())['cases']
    template = json.loads(args.template.read_text())
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    for case in cases:
        output = (args.output / case['case_id']).resolve()
        config = {**template, 'mode': args.policy, 'case': case, 'output_dir': str(output),
                  'norm_stats': str(args.norm_stats.resolve()), 'replay_result': str(args.replay_result.resolve()),
                  'checkpoint_sha256': identity, 'evaluation_cases_sha256': sha256(args.cases),
                  'policy_command': ['env', 'OPENPI_DATA_HOME=' + str(ROOT / '.scratch/pi05_g1/venv/openpi-cache'),
                                     str(ROOT / '.scratch/pi05_g1/venv/bin/python'),
                                     '-m', 'experiments.pi05_g1.model_runtime', '--server',
                                     '--checkpoint', str(args.checkpoint.resolve()),
                                     '--norm-stats', str(args.norm_stats.resolve())]}
        config.pop('expert_sequence', None)
        config.pop('expert_sha256', None)
        config_path = configs / (case['case_id'] + '.json')
        result_path = output / 'result.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            previous_config = json.loads((output / 'config.json').read_text())
            if previous.get('status') != 'completed' or previous_config != config:
                raise RuntimeError(f'Existing case is incomplete or changed: {output}; audit before any retry')
            print('CASE_REUSE', args.policy, case['case_id'], previous.get('success'), flush=True)
            continue
        if output.exists():
            raise RuntimeError(f'Partial evidence exists without result: {output}; audit before any retry')
        config_path.write_text(json.dumps(config, indent=2))
        command = [sys.executable, '-m', 'experiments.pi05_g1.sim_launcher', '--docker', '--config', str(config_path.resolve())]
        started = time.time()
        print('CASE_START', args.policy, case['case_id'], flush=True)
        with (configs / (case['case_id'] + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'case_id': case['case_id'], 'policy': args.policy, 'command': command,
                  'started_unix': started, 'elapsed_seconds': time.time() - started,
                  'exit_code': result.returncode, 'checkpoint_sha256': identity}
        (configs / (case['case_id'] + '.launch.json')).write_text(json.dumps(record, indent=2))
        if result.returncode or not result_path.is_file():
            raise RuntimeError(f'Case infrastructure failed, not a completed task trial: {case["case_id"]}; see {config_path.with_suffix(".log")}')
        outcome = json.loads(result_path.read_text())
        if outcome.get('status') != 'completed' or not outcome.get('source_interface_passed') or not outcome.get('replay_passed'):
            raise RuntimeError(f'Case infrastructure or physical interface failed: {case["case_id"]}')
        print('CASE_COMPLETE', args.policy, case['case_id'], json.dumps({k: outcome.get(k) for k in ['success', 'grasped', 'stable_placement', 'fell', 'dropped', 'timeout']}), flush=True)


if __name__ == '__main__':
    main()
