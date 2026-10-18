"""Execute a predeclared serial job plan under the caller's one GPU supervisor.

This runner never chooses parameters/checkpoints from final-case outcomes. Each
case keeps its own launch log and receipt. Infrastructure failures stop the plan;
completed physical task failures remain valid observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def persist(path, value):
    temp = path.with_suffix(path.suffix + '.incomplete')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def run(plan_path, receipt_path):
    if receipt_path.exists():
        raise FileExistsError('Preserve/audit an existing batch receipt before rerunning')
    plan = json.loads(plan_path.read_text())
    names = [job['name'] for job in plan['jobs']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Nonempty uniquely named jobs required')
    frozen = dict(plan.get('frozen_files', {}))
    frozen[str(plan_path.resolve())] = digest(plan_path)
    for job in plan['jobs']:
        frozen[job['config_path']] = digest(job['config_path'])
    receipt = {'schema': 'pi05_g1_round02_batch_receipt_v1', 'status': 'prepared',
               'plan': str(plan_path.resolve()), 'plan_sha256': digest(plan_path),
               'frozen_files': frozen, 'jobs': [], 'started_unix': time.time()}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    persist(receipt_path, receipt)
    try:
        for job in plan['jobs']:
            for path, expected in frozen.items():
                if digest(path) != expected:
                    raise RuntimeError(f'Frozen batch input changed: {path}')
            config_path = Path(job['config_path'])
            config = json.loads(config_path.read_text())
            normalized = dict(config)
            if 'action_limits_file' in config:
                normalized['action_limits'] = config['action_limits_file']
            case_output = Path(config['output_dir'])
            result_path = case_output / 'result.json'
            record = {'name': job['name'], 'case_id': config['case']['case_id'],
                      'policy': config.get('policy_id', config['mode']),
                      'config_path': str(config_path), 'config_file_sha256': digest(config_path),
                      'output': str(case_output), 'started_unix': time.time()}
            receipt['status'] = 'running'
            receipt['jobs'].append(record)
            persist(receipt_path, receipt)
            if case_output.exists():
                if not result_path.is_file() or json.loads((case_output / 'config.json').read_text()) != normalized:
                    raise RuntimeError(f'Partial or changed existing case: {case_output}')
                result = json.loads(result_path.read_text())
                if result.get('status') != 'completed':
                    raise RuntimeError(f'Incomplete existing case: {case_output}')
                record.update(reused=True, launch_elapsed_seconds=0, exit_code=0)
            else:
                log_path = config_path.with_suffix('.launch.log')
                launch_path = config_path.with_suffix('.launch.json')
                if log_path.exists() or launch_path.exists():
                    raise FileExistsError(f'Prior launch evidence exists: {log_path}')
                command = [sys.executable, '-m', 'experiments.pi05_g1_round02.sim_launcher',
                           '--docker', '--config', str(config_path.resolve())]
                print('CASE_START', record['policy'], record['case_id'], flush=True)
                with log_path.open('x') as log:
                    child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                launch = {'command': command, 'exit_code': child.returncode,
                          'started_unix': record['started_unix'],
                          'elapsed_seconds': time.time() - record['started_unix']}
                persist(launch_path, launch)
                record.update(exit_code=child.returncode, launch_elapsed_seconds=launch['elapsed_seconds'])
                if child.returncode or not result_path.is_file():
                    raise RuntimeError(f'Case infrastructure failed: {record["name"]}; see {log_path}')
                result = json.loads(result_path.read_text())
            if result.get('status') != 'completed' or result.get('source_interface_passed') is not True:
                raise RuntimeError(f'Case physical interface incomplete: {record["name"]}')
            if config['mode'] != 'replay' and result.get('replay_passed') is not True:
                raise RuntimeError(f'Case replay gate not passed: {record["name"]}')
            record.update(status='completed', result_sha256=digest(result_path),
                          success=result['success'], grasped=result['grasped'],
                          stable_placement=result['stable_placement'],
                          fell=result['fell'], dropped=result['dropped'], timeout=result['timeout'],
                          initial_physical_state_sha256=result['initial_physical_state_sha256'],
                          frozen_interface_sha256=result['frozen_interface_sha256'])
            persist(receipt_path, receipt)
            print('CASE_COMPLETE', record['policy'], record['case_id'], json.dumps({key: record[key] for key in ('success','grasped','stable_placement','fell','dropped','timeout')}), flush=True)
            if job.get('require_task_success') and result['success'] is not True:
                raise RuntimeError(f'Required replay/teacher task qualification failed: {record["name"]}')
        receipt.update(status='completed', ended_unix=time.time())
        persist(receipt_path, receipt)
        print('BATCH_COMPLETE', len(receipt['jobs']), flush=True)
    except BaseException as error:
        receipt.update(status='infrastructure_failed', error=repr(error), ended_unix=time.time())
        persist(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.receipt)


if __name__ == '__main__':
    main()
