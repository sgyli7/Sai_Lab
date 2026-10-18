"""Finite twenty-case second round in an independent CPU service, FIFO per case."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .benchmark_pose_speed import load_plan
from .data import OWN, ROOT, sha

PYTHON = str(OWN / 'venv/bin/python')


def run(plan_path, output, unit):
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned driver evidence required')
    if (not unit.startswith('lingbot-eval-round02-') or not unit.endswith('.service')
            or not os.environ.get('INVOCATION_ID')
            or unit not in Path('/proc/self/cgroup').read_text()):
        raise ValueError('Run inside its independent named CPU service')
    plan = load_plan(plan_path)
    if plan.get('round') != 2 or plan.get('same_cohort_reused_after_baseline') is not True:
        raise ValueError('Explicit second-round repeated cohort required')
    output.mkdir(parents=True)
    state_path = output / 'state.json'
    state = {'schema': 'lingbot_round02_finite_driver_v1', 'status': 'prepared',
             'plan': str(plan_path), 'plan_sha256': sha(plan_path),
             'source_sha256': sha(__file__), 'maximum_cases': 20,
             'maximum_segments': 7, 'maximum_cases_per_segment': 3,
             'completed_cases': 0, 'segments': [], 'unit': unit,
             'pid': os.getpid(), 'started_unix': time.time()}
    def save():
        temporary = state_path.with_suffix('.incomplete')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
        temporary.replace(state_path)
    child = None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        state['signal'] = {'number': signum, 'unix': time.time()}
        save()
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    previous = {s: signal.signal(s, stop) for s in [signal.SIGTERM, signal.SIGINT]}
    receipts = []
    save()
    try:
        for index in range(1, 8):
            if stopping:
                raise InterruptedError('No new reservation after stop')
            if sha(plan_path) != state['plan_sha256'] or sha(__file__) != state['source_sha256']:
                raise ValueError('Immutable finite driver changed')
            receipt = output / f'segment{index:02d}' / 'receipt.json'
            command = [PYTHON, '-m', 'experiments.lingbot_vla2_g1.evaluate_round02',
                       '--plan', str(plan_path), '--receipt', str(receipt), '--max-jobs', '3']
            for prior in receipts:
                command += ['--resume-receipt', str(prior)]
            segment = {'index': index, 'receipt': str(receipt), 'command': command,
                       'status': 'running'}
            state['segments'].append(segment)
            state['status'] = 'running'
            save()
            with (output / f'segment{index:02d}.log').open('x', buffering=1) as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                state['child_pid'] = child.pid
                save()
                for line in child.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
                code = child.wait()
                child = None
            if code:
                raise RuntimeError('Finite evaluation segment failed: ' + str(code))
            document = json.loads(receipt.read_text())
            if document.get('status') != 'completed':
                raise ValueError('Segment producer did not complete')
            receipts.append(receipt)
            segment.update(status='completed', receipt_sha256=sha(receipt))
            state.update(completed_cases=document['completed_total_cases'], child_pid=None)
            save()
            if state['completed_cases'] == 20:
                break
        if state['completed_cases'] != 20:
            raise ValueError('Finite twenty-case coverage incomplete')
        summary = OWN / 'benchmark_round02_pose_speed_summary.json'
        command = [PYTHON, '-m', 'experiments.lingbot_vla2_g1.benchmark_pose_speed_v2',
                   '--plan', str(plan_path), '--output', str(summary)]
        for receipt in receipts:
            command += ['--receipt', str(receipt)]
        child = subprocess.Popen(command, cwd=ROOT)
        if child.wait():
            raise RuntimeError('CPU report failed')
        child = None
        result = json.loads(summary.read_text())
        if result.get('status') != 'completed' or result.get('formal_verdict_available') is not True:
            raise ValueError('All twenty physical cases must qualify')
        state.update(status='completed', summary=str(summary), summary_sha256=sha(summary),
                     child_pid=None, ended_unix=time.time())
        save()
        print(json.dumps({'event': 'round02_benchmark_completed',
                          'successes': result['success_count'],
                          'qualified_carry': result['long_carry_qualified_count']}), flush=True)
    except BaseException as error:
        state.update(status='stopped' if stopping else 'failed', error=repr(error),
                     ended_unix=time.time())
        save()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    a = parser.parse_args()
    run(a.plan.absolute(), a.output.absolute(), a.unit)
