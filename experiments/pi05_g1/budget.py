"""Durable GPU accounting and external Linux supervision of owned children.

A transient user service bounds CPU/unified allocations charged to its cgroup.
CUDA allocations may escape that accounting on GB10, so host MemAvailable is
also watched. Docker daemon-created containers require their own limits; the
CLI service does not contain those containers. No unrelated PID is signalled.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

LIMITS = {'preflight': 3600, 'training': 14400, 'evaluation': 10800}
TOTAL = 28800
GIB = 1024**3


def utc():
    return datetime.now(timezone.utc).isoformat()


def host_available_bytes(path=Path('/proc/meminfo')):
    for line in Path(path).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable; refusing unsupervised launch')


def memory_limits(high_gib, max_gib, reserve_gib, available):
    if (not all(math.isfinite(value) for value in (high_gib, max_gib, reserve_gib))
            or not 0 < high_gib < max_gib or reserve_gib <= 0):
        raise ValueError('Require finite 0 < memory-high-gib < memory-max-gib and positive host reserve')
    page_size = int(os.sysconf('SC_PAGE_SIZE'))
    if page_size <= 0:
        raise RuntimeError('Cannot determine the Linux memory-controller page size')
    # Linux floors cgroup memory controls to pages; match its actual values before
    # launch so the enforcement check stays exact without weakening the limits.
    raw_maximum = min(int(max_gib * GIB), available - math.ceil(reserve_gib * GIB))
    maximum = raw_maximum // page_size * page_size
    if maximum < 16 * 1024**2:
        raise RuntimeError('Less than 16 MiB safely available above the host reserve')
    raw_high = min(int(high_gib * GIB), maximum * 9 // 10)
    high = raw_high // page_size * page_size
    if not 0 < high < maximum or maximum > raw_maximum:
        raise RuntimeError('Invalid page-aligned owned memory limits')
    return high, maximum


def service_command(unit, command, high, maximum, work_seconds, grace):
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])
    args = ['systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect',
            '--service-type=exec', '--unit=' + unit, '--working-directory=' + os.getcwd(),
            '--property=MemoryAccounting=yes', f'--property=MemoryHigh={high}',
            f'--property=MemoryMax={maximum}', '--property=MemorySwapMax=0',
            '--property=OOMPolicy=kill',
            '--property=KillMode=control-group', f'--property=TimeoutStopSec={grace}s',
            f'--property=RuntimeMaxSec={work_seconds}s']
    # Name-only flags inherit the supervisor environment without printing secret values.
    args += ['--setenv=' + key for key in os.environ if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key)]
    return args + ['--', executable, *command[1:]]


def unit_properties(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=ActiveState,SubState,ControlGroup,Result,ExecMainStatus,MemoryPeak,MemoryCurrent'],
        capture_output=True, text=True, timeout=2)
    if result.returncode and ('not found' in result.stderr or 'not be found' in result.stderr):
        return {'ActiveState': 'inactive', 'Result': 'collected'}
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'systemd unit query failed')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def signal_owned_unit(unit, signum):
    if not re.fullmatch(r'pi05-g1-owned-[0-9a-f]{32}\.service', unit):
        raise ValueError('Refusing to signal a foreign unit')
    subprocess.run(['systemctl', '--user', 'kill', f'--signal={signum}', '--kill-whom=all', unit],
                   capture_output=True, text=True, timeout=2)


def cgroup_snapshot(properties):
    path = properties.get('ControlGroup')
    result = {}
    if not path:
        return result
    root = Path('/sys/fs/cgroup') / path.lstrip('/')
    for name in ('memory.current', 'memory.peak', 'memory.high', 'memory.max', 'memory.swap.max', 'memory.oom.group'):
        file = root / name
        try:
            result[name] = int(file.read_text().strip())
        except FileNotFoundError:
            pass  # A collected owned unit can disappear between monitoring reads.
    file = root / 'memory.events'
    try:
        result['memory.events'] = {key: int(value) for key, value in
                                  (line.split() for line in file.read_text().splitlines())}
    except FileNotFoundError:
        pass
    return result


def explicitly_missing_container(stderr):
    """Docker varies capitalization; only explicit missing-object errors prove absence."""
    return re.search(r'\bno such (?:object|container)(?:\s*:|\s*$)', stderr, re.IGNORECASE) is not None


def cleanup_owned_docker(directory, token, deadline):
    """Verify launcher receipts and remove only containers bearing this run token."""
    files = sorted(Path(directory).glob('launch-*.json'))
    report = {'registered_launches': len(files), 'containers': [], 'complete': True}
    if not files:
        return report
    rows = [json.loads(file.read_text()) for file in files]
    if any(row.get('schema') != 'pi05_g1_docker_ownership_v1' or row.get('budget_token') != token for row in rows):
        raise ValueError('Invalid Docker ownership receipt/token')
    def call(argv, timeout=3):
        left = deadline - time.monotonic()
        if left <= .05:
            raise TimeoutError('Owned Docker cleanup reserve exhausted')
        return subprocess.run(['docker', *argv], capture_output=True, text=True, timeout=min(timeout, left))
    listed = call(['ps', '--all', '--quiet', '--no-trunc', '--filter', 'label=sai.pi05_g1.budget=' + token])
    if listed.returncode:
        raise RuntimeError('Cannot verify owned Docker containers: ' + listed.stderr.strip())
    ids = set(listed.stdout.split())
    for row in rows:
        cid = row.get('container_id')
        cidfile = Path(row['cidfile'])
        if not cid and cidfile.is_file():
            cid = cidfile.read_text().strip()
        if cid:
            if not re.fullmatch(r'[0-9a-f]{12,64}', cid):
                raise ValueError('Invalid owned container ID')
            ids.add(cid)
        elif row.get('status') != 'finished':
            report['complete'] = False
            report.setdefault('unresolved_launches', []).append(row['launch_id'])
    for cid in sorted(ids):
        receipt = {'container_id': cid}
        inspected = call(['inspect', '--format', '{{json .Config.Labels}}', cid])
        if inspected.returncode:
            if explicitly_missing_container(inspected.stderr):
                receipt['removed'] = True
                report['containers'].append(receipt)
                continue
            raise RuntimeError('Cannot inspect owned container: ' + inspected.stderr.strip())
        labels = json.loads(inspected.stdout)
        if labels.get('sai.pi05_g1.budget') != token:
            raise ValueError('Refusing to change foreign Docker container')
        # The service already had twenty seconds for graceful launcher cleanup.
        # Finish remaining owned containers with bounded kill/removal, never global pruning.
        receipt['kill_exit_code'] = call(['kill', cid]).returncode
        receipt['remove_exit_code'] = call(['rm', '--force', cid]).returncode
        check = call(['inspect', cid])
        receipt['removed'] = bool(check.returncode and explicitly_missing_container(check.stderr))
        report['complete'] &= receipt['removed']
        report['containers'].append(receipt)
    check = call(['ps', '--all', '--quiet', '--filter', 'label=sai.pi05_g1.budget=' + token])
    if check.returncode or check.stdout.strip():
        report['complete'] = False
        report['remaining_owned_container_ids'] = check.stdout.split()
    return report


def blocks_new_launch(row):
    return row.get('status') in ('running', 'cleanup_pending') or row.get('accounting_open') is True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--stage', choices=LIMITS, required=True)
    ap.add_argument('--max-seconds', type=float)
    ap.add_argument('--memory-high-gib', type=float, default=80.)
    ap.add_argument('--memory-max-gib', type=float, default=90.)
    ap.add_argument('--host-reserve-gib', type=float, default=12.)
    ap.add_argument('--poll-seconds', type=float, default=.5)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.poll_seconds <= 1:
        ap.error('A command and poll-seconds in (0, 1] are required')
    high, maximum = memory_limits(args.memory_high_gib, args.memory_max_gib,
                                  args.host_reserve_gib, host_available_bytes())
    if not shutil.which('systemd-run') or not shutil.which('systemctl'):
        raise RuntimeError('systemd user cgroup supervision required; no unbounded fallback')
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {
            'schema': 'pi05_g1_gpu_budget_v1', 'total_limit_seconds': TOTAL,
            'stage_limits_seconds': LIMITS, 'runs': []}
        if any(blocks_new_launch(r) for r in ledger['runs']):
            raise RuntimeError('An unfinished owned run needs audited reconciliation before another launch')
        used_total = sum(r['elapsed_seconds'] for r in ledger['runs'])
        used_stage = sum(r['elapsed_seconds'] for r in ledger['runs'] if r['stage'] == args.stage)
        remaining = min(TOTAL - used_total, LIMITS[args.stage] - used_stage)
        if args.max_seconds is not None:
            remaining = min(remaining, args.max_seconds)
        if remaining < 1:
            raise RuntimeError('GPU budget exhausted')
        row = {'stage': args.stage, 'started_utc': utc(), 'started_unix': time.time(),
               'command': command, 'status': 'running', 'allowed_seconds': remaining,
               'elapsed_seconds': 0, 'supervisor_pid': os.getpid()}
        grace = min(20., remaining * .2)
        docker_reserve = min(15., remaining * .2)
        work_seconds = max(.1, remaining - grace - docker_reserve - min(5., remaining * .1))
        token = uuid.uuid4().hex
        unit = 'pi05-g1-owned-' + token + '.service'
        ownership_dir = (args.ledger.parent / 'owned_runs' / token).resolve()
        ownership_dir.mkdir(parents=True, exist_ok=False)
        os.environ['PI05_G1_BUDGET_TOKEN'] = token
        os.environ['PI05_G1_BUDGET_OWNERSHIP_DIR'] = str(ownership_dir)
        row.update(child_work_limit_seconds=work_seconds, cleanup_reserve_seconds=remaining-work_seconds,
                   owned_unit=unit, ownership_receipt_dir=str(ownership_dir), budget_token=token,
                   cleanup_grace_seconds=grace, docker_cleanup_reserve_seconds=docker_reserve, memory_high_bytes=high, memory_max_bytes=maximum,
                   host_memavailable_stop_bytes=int(args.host_reserve_gib * GIB), memory_swap_max_bytes=0,
                   memory_scope='owned systemd service descendants only; Docker daemon containers need independent limits',
                   cuda_accounting='cgroup coverage not assumed; external host MemAvailable safeguard enabled')
        ledger['runs'].append(row)
        def save():
            temp = args.ledger.with_suffix('.tmp')
            temp.write_text(json.dumps(ledger, indent=2, allow_nan=False))
            temp.replace(args.ledger)
        save()
        started = time.monotonic()
        child = None
        interrupted = False
        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        reason, termination_at, killed, code = None, None, False, 1
        properties = {}
        try:
            child = subprocess.Popen(service_command(unit, command, high, maximum, work_seconds, grace),
                                     start_new_session=True)
            row['launcher_pid'] = child.pid
            save()
            while True:
                elapsed = time.monotonic() - started
                properties = unit_properties(unit)
                row['last_unit_properties'] = properties
                snapshot = cgroup_snapshot(properties)
                if snapshot:
                    row['last_cgroup_snapshot'] = snapshot
                    if snapshot.get('memory.max') != maximum or snapshot.get('memory.high') != high or snapshot.get('memory.swap.max') != 0 or snapshot.get('memory.oom.group') != 1:
                        raise RuntimeError('Owned cgroup memory limits/OOM-group were not enforced')
                row['peak_cgroup_memory_bytes'] = max(row.get('peak_cgroup_memory_bytes', 0), snapshot.get('memory.peak', 0))
                available = host_available_bytes()
                row['minimum_host_memavailable_bytes'] = min(row.get('minimum_host_memavailable_bytes', available), available)
                if snapshot.get('memory.current', 0) >= high or available < int(args.host_reserve_gib * 1.5 * GIB):
                    if not row.get('memory_warning_utc'):
                        row['memory_warning_utc'] = utc()
                        print('BUDGET_MEMORY_WARNING', json.dumps({'owned_unit': unit, 'cgroup': snapshot, 'host_available_bytes': available}), flush=True)
                if reason is None:
                    if interrupted:
                        reason = 'interrupted'
                    elif available < int(args.host_reserve_gib * GIB):
                        reason = 'host_memory_guard'
                    elif elapsed >= work_seconds:
                        reason = 'budget_expired'
                    if reason:
                        termination_at = time.monotonic()
                        signal_owned_unit(unit, 'SIGTERM')
                        if reason == 'host_memory_guard':
                            # Host protection cannot wait twenty seconds for application cleanup.
                            signal_owned_unit(unit, 'SIGKILL')
                            killed = True
                            row['docker_memory_emergency_cleanup'] = cleanup_owned_docker(ownership_dir, token, started + remaining)
                if reason and not killed and time.monotonic() - termination_at >= grace:
                    signal_owned_unit(unit, 'SIGKILL')
                    killed = True
                state = properties.get('ActiveState')
                if child.poll() is not None and state in ('inactive', 'failed'):
                    code = child.returncode
                    if reason:
                        code = 124 if reason == 'budget_expired' else 137
                    row['status'] = reason or ('completed' if code == 0 else
                                     'memory_limit' if properties.get('Result') == 'oom-kill' else 'failed')
                    cleanup = cleanup_owned_docker(ownership_dir, token, started + remaining)
                    row['docker_cleanup'] = cleanup
                    if not cleanup['complete']:
                        row.update(status='cleanup_pending', accounting_open=True)
                        code = 124
                    break
                if elapsed >= remaining:
                    if not killed:
                        signal_owned_unit(unit, 'SIGKILL')
                    row.update(status='cleanup_pending', accounting_open=True,
                               cleanup_note='Owned unit may contain uninterruptible tasks. SIGKILL is not proof of exit; elapsed accounting remains open and new launches are blocked.')
                    code = 124
                    break
                row['elapsed_seconds'] = elapsed
                save()
                time.sleep(min(args.poll_seconds, max(.01, remaining-elapsed)))
        except BaseException as exc:
            row.update(status='supervisor_error', error=repr(exc))
            code = 1
            if child is not None:
                try:
                    signal_owned_unit(unit, 'SIGKILL')
                    properties = unit_properties(unit)
                except Exception as cleanup_error:
                    row['cleanup_error'] = repr(cleanup_error)
                if properties.get('ActiveState') not in ('inactive', 'failed'):
                    row.update(status='cleanup_pending', accounting_open=True)
                try:
                    cleanup = cleanup_owned_docker(ownership_dir, token, started + remaining)
                    row['docker_cleanup'] = cleanup
                    if not cleanup['complete']:
                        row.update(status='cleanup_pending', accounting_open=True)
                except Exception as docker_error:
                    row.update(status='cleanup_pending', accounting_open=True, docker_cleanup_error=repr(docker_error))
        finally:
            row['elapsed_seconds'] = time.monotonic() - started
            row['budget_overrun_seconds'] = max(0., row['elapsed_seconds'] - remaining)
            row['supervisor_ended_utc'] = utc()
            if not row.get('accounting_open'):
                row['ended_utc'] = row['supervisor_ended_utc']
            if child is not None and child.poll() is None:
                # This is only the owned systemd-run launcher, never a foreign process.
                os.killpg(child.pid, signal.SIGKILL)
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    row['launcher_cleanup_pending'] = True
            row['exit_code'] = code
            save()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        print('GPU_BUDGET', json.dumps(row), flush=True)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
