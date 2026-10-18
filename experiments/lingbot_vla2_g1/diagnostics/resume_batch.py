"""One finite owned turn: sampled-action diagnosis, then fresh TRAIN teachers.

Successful process completion never means a successful robot task. The original
physical results require an independent CPU audit after GPU cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.absolute().is_relative_to(OWN):
        raise FileExistsError('Fresh owned batch output required')
    plan = json.loads(args.manifest.read_text())
    if plan.get('schema') != 'lingbot_resumed_diagnostic_and_train_collection_v1' or len(plan['jobs']) != 4:
        raise ValueError('Exactly the reviewed diagnosis and three physical jobs are required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': 'lingbot_resumed_finite_batch_result_v1', 'status': 'running',
              'manifest_sha256': sha(args.manifest), 'jobs': [],
              'LingBot_task_success_claim': False, 'formal_benchmark_trial': False,
              'independent_physical_audits_pending': True}
    try:
        for index, job in enumerate(plan['jobs']):
            command = job['argv']
            allowed = ('experiments.lingbot_vla2_g1.diagnostics.action_probe',
                       'experiments.lingbot_vla2_g1.run_case_pose_speed')
            if command[:2] != [str(OWN / 'venv/bin/python'), '-m'] or command[2] not in allowed:
                raise ValueError('Unreviewed workload entry point')
            output = Path(job['result_file'])
            if output.exists() or not output.is_relative_to(OWN):
                raise FileExistsError('Preserve every workload result')
            print(json.dumps({'event': 'resumed_subjob_started', 'index': index,
                              'purpose': job['purpose']}), flush=True)
            log = args.output / f'job_{index:02}_stdout.log'
            with log.open('x') as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT,
                                         timeout=job['timeout_seconds'], check=False,
                                         env=dict(os.environ, HF_HUB_OFFLINE='1',
                                                  TRANSFORMERS_OFFLINE='1'))
            result = json.loads(output.read_text()) if output.is_file() else {}
            record = {'index': index, 'purpose': job['purpose'], 'returncode': process.returncode,
                      'result_file': str(output), 'status': result.get('status'),
                      'actual_physical_success': result.get('success'),
                      'log': str(log), 'log_sha256': sha(log)}
            report['jobs'].append(record)
            if process.returncode or result.get('status') != 'completed':
                raise RuntimeError('Finite subjob failed; preserve evidence: ' + str(index))
            record['result_sha256'] = sha(output)
            print(json.dumps({'event': 'resumed_subjob_completed', **record}), flush=True)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        with (args.output / 'result.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
