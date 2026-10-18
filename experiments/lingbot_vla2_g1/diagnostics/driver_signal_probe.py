"""CPU-only signal exposure probe using the recorded finite driver unchanged.

Only the GPU subprocess boundary is replaced by a sleeping CPU child. An
explicitly created disposable process group receives the synthetic signal.
This can reproduce signal propagation; it cannot identify a historical sender.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).absolute().parents[3]
OWN = ROOT / '.scratch/lingbot_vla2_g1'
DRIVER = OWN / 'preparation_scripts/evaluate_pose_speed_driver.py'
PLAN = OWN / 'evaluation_plan_pose_speed/plan.json'


def wait_running(path):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if path.exists():
            value = json.loads(path.read_text())
            if value.get('status') == 'running':
                return value
        time.sleep(.05)
    raise RuntimeError('Recorded driver did not reach its CPU subprocess seam')


def probe(mode, output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN / 'diagnostics'):
        raise ValueError('Fresh diagnostic-only output required')
    output.mkdir(parents=True)
    # All actual driver gates and signal handlers remain in use. No policy,
    # queue, CUDA or simulator subprocess can run through this boundary.
    wrapper = '''import importlib.util, pathlib, subprocess, sys
spec=importlib.util.spec_from_file_location('recorded_driver', sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.subprocess.Popen
def cpu_child(command, **kwargs):
    return original([sys.executable,'-u','-c','import time;time.sleep(30)'],**kwargs)
module.subprocess.Popen=cpu_child
module.run(pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]))
'''
    command = [sys.executable, '-c', wrapper, str(DRIVER), str(PLAN), str(output / 'driver')]
    unit = None
    driver = None
    disposable = None
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        with (output / 'driver.log').open('x') as stream:
            if mode == 'attached':
                driver = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                disposable = driver
            else:
                unit = 'lingbot-cpu-signal-probe-' + uuid.uuid4().hex
                subprocess.run(['systemd-run', '--user', '--quiet', '--collect', '--unit=' + unit,
                                '--property=Type=exec', '--property=Environment=CUDA_VISIBLE_DEVICES=',
                                '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
                                '--property=WorkingDirectory=' + str(ROOT),
                                '--property=StandardOutput=append:' + str(output / 'driver.log'),
                                '--property=StandardError=append:' + str(output / 'driver.log'),
                                *command], check=True, cwd=ROOT)
                disposable = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                              env=environment, start_new_session=True)
            initial = wait_running(output / 'driver/state.json')
            if os.getpgid(disposable.pid) != disposable.pid or os.getsid(disposable.pid) != disposable.pid:
                raise ValueError('Refuse to signal a process group not created for this probe')
            driver_pid = driver.pid if driver else int(subprocess.check_output(
                ['systemctl', '--user', 'show', unit, '--property=MainPID', '--value'], text=True).strip())
            identities = {'driver_pid': driver_pid, 'driver_pgid': os.getpgid(driver_pid),
                          'synthetic_terminal_pgid': disposable.pid, 'unit': unit}
            os.killpg(disposable.pid, signal.SIGTERM)
            time.sleep(.5)
            if driver is not None:
                driver.wait(timeout=5)
            state = json.loads((output / 'driver/state.json').read_text())
            survived = state.get('status') == 'running'
            receipt = {'schema': 'lingbot_cpu_driver_signal_probe_v1', 'mode': mode,
                       'passed': survived, 'observed_state_after_external_group_signal': state['status'],
                       'driver_source_sha256': hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
                       'identities': identities, 'signal': 'SIGTERM', 'gpu_workloads_replaced_by_cpu_sleep': True,
                       'historical_signal_sender_identified': False,
                       'scope': 'Controlled process-group exposure only; not a historical cause attribution'}
            (output / 'result.json').write_text(json.dumps(receipt, indent=2))
            return receipt
    finally:
        if unit:
            subprocess.run(['systemctl', '--user', 'stop', unit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if disposable is not None and disposable.poll() is None:
            # Exact disposable group, created above with start_new_session.
            os.killpg(disposable.pid, signal.SIGTERM)
            disposable.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['attached', 'isolated'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.mode, args.output)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 2)
