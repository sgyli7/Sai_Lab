"""Account the union of confirmed work intervals, including supervised jobs.

No elapsed wall-clock gap is charged on resume. Agent heartbeats confirm only
the interval since their previous heartbeat (within max_gap); a supervised job
keeps confirming its own intervals even while the agent is disconnected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_start_ticks(pid):
    try:return int(Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ',1)[1].split()[19])
    except (FileNotFoundError,ProcessLookupError):return None


def _merge(intervals, start, end):
    result = []
    for left, right in sorted([*intervals, [start, end]]):
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


class ActiveBudget:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'active_budget.json'

    @contextmanager
    def _edit(self):
        with (self.directory / 'active_budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else None
            yield data
            if data is not None:
                temporary = self.path.with_suffix(f'.{os.getpid()}.tmp')
                with temporary.open('w') as stream:
                    json.dump(data, stream, indent=2, allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)

    def initialize(self, seconds=28800., initial_seconds=0., credit_note=''):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Budget must be positive and finite')
        if not 0 <= initial_seconds <= seconds or not math.isfinite(initial_seconds):
            raise ValueError('Invalid initial verified work credit')
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'active_budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError(self.path)
            data = dict(schema_version=1, limit_seconds=float(seconds),
                        initial_seconds=float(initial_seconds), credit_note=credit_note,
                        created_unix=time.time(), intervals={}, actors={}, gaps=[], events=[])
            temporary = self.path.with_suffix('.initial.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            os.replace(temporary, self.path)
        return self.status()

    def heartbeat(self, actor, *, max_gap=120., stop=False, now=None, boot=None):
        now = time.monotonic() if now is None else float(now)
        boot = _boot_id() if boot is None else boot
        with self._edit() as data:
            if data is None:
                raise FileNotFoundError(self.path)
            prior = data['actors'].get(actor)
            if prior and prior['boot'] == boot:
                gap = now - prior['monotonic']
                if 0 <= gap <= max_gap:
                    data['intervals'][boot] = _merge(data['intervals'].get(boot, []),
                                                      prior['monotonic'], now)
                else:
                    data['gaps'].append(dict(actor=actor, boot=boot,
                                             start=prior['monotonic'], end=now,
                                             reason='unconfirmed_gap'))
            if stop:
                data['actors'].pop(actor, None)
            else:
                data['actors'][actor] = dict(boot=boot, monotonic=now, unix=time.time())
        return self.status()

    def event(self, kind, **fields):
        with self._edit() as data:
            if data is None:
                raise FileNotFoundError(self.path)
            data['events'].append(dict(kind=kind, unix=time.time(), **fields))

    def status(self):
        data = json.loads(self.path.read_text())
        used = data['initial_seconds'] + sum(
            end - start for intervals in data['intervals'].values() for start, end in intervals)
        def covered(gap):
            return any(start<=gap['start'] and end>=gap['end']
                       for start,end in data['intervals'].get(gap['boot'],[]))
        return dict(limit_seconds=data['limit_seconds'], used_seconds=used,
                    remaining_seconds=max(0., data['limit_seconds'] - used),
                    actors=list(data['actors']),
                    unconfirmed_gaps=sum(not gap.get('resolution') and not covered(gap) for gap in data['gaps']))


def remaining(directory=None, *, reserve=0.):
    if directory is None:
        from .tasks import SESSION
        directory = SESSION
    directory = Path(directory)
    ledger = ActiveBudget(directory)
    if ledger.path.exists():
        return ledger.status()['remaining_seconds'] - reserve
    session = json.loads((directory / 'session.json').read_text())
    return float(session['deadline_unix']) - time.time() - reserve


def require_supervision(directory):
    directory = Path(directory).resolve()
    if (directory / 'active_budget.json').exists():
        supervised = os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR')
        if not supervised or Path(supervised).resolve() != directory:
            raise RuntimeError('Active-time experiments must run through research.budget run')


def legacy_deadline(directory):
    directory = Path(directory)
    if (directory / 'active_budget.json').exists():
        raise RuntimeError('This legacy search uses wall-clock deadlines; use supervised research.train for an active-time session')
    return float(json.loads((directory / 'session.json').read_text())['deadline_unix'])


def live_group_members(pgid):
    """Only inspect the process group created for this job, excluding zombies."""
    members=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(') ',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':members.append(int(path.parent.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return members


def cleanup_group(pgid, leader_start):
    if pgid==os.getpgrp():raise RuntimeError('Refusing to clean the supervisor process group')
    current=process_start_ticks(pgid)
    if current is not None and current!=leader_start:
        raise RuntimeError('Job leader PID was reused; refusing to signal another job')
    before=live_group_members(pgid)
    for sig,seconds in [(signal.SIGTERM,3.),(signal.SIGKILL,2.)]:
        if not live_group_members(pgid):break
        try:os.killpg(pgid,sig)
        except ProcessLookupError:break
        until=time.monotonic()+seconds
        while live_group_members(pgid) and time.monotonic()<until:time.sleep(.1)
    return dict(found=before,remaining=live_group_members(pgid))


def start_supervised(directory,command,*,timeout,reserve=0.,label='job',cpus=4,nice=10):
    """Run the supervisor independently of the launching session and its pipes."""
    _validate_resources(timeout,reserve,cpus,nice)
    directory=Path(directory).resolve();budget=ActiveBudget(directory)
    if budget.status()['remaining_seconds']<=reserve:raise RuntimeError('Insufficient active budget')
    logs=directory/'supervisors';logs.mkdir(exist_ok=True)
    log=logs/(label+'-'+uuid.uuid4().hex[:10]+'.log')
    launch=[sys.executable,str(Path(__file__).resolve()),'--directory',str(directory),'run',
            '--timeout',str(timeout),'--reserve',str(reserve),'--label',label,
            '--cpus',str(cpus),'--nice',str(nice),'--',*command]
    with log.open('x') as stream:
        process=subprocess.Popen(launch,start_new_session=True,stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT)
    record=dict(supervisor_pid=process.pid,process_start_ticks=process_start_ticks(process.pid),
                log=str(log),command=command,timeout=timeout,reserve=reserve,cpus=cpus,nice=nice)
    budget.event('detached_supervisor_started',**record)
    return process,record


def _validate_resources(timeout,reserve,cpus,nice):
    if not math.isfinite(timeout) or timeout<=0 or not math.isfinite(reserve) or reserve<0:
        raise ValueError('Positive finite timeout and nonnegative finite reserve required')
    if cpus<1 or not 0<=nice<=19:
        raise ValueError('Positive CPU count and background nice level 0..19 required')


def run_supervised(directory, command, *, timeout, reserve=0., label='job', cpus=4, nice=10):
    _validate_resources(timeout,reserve,cpus,nice)
    budget = ActiveBudget(directory)
    if budget.status()['remaining_seconds'] <= reserve:
        raise RuntimeError('Insufficient active budget; no job started')
    actor = f'{label}:{uuid.uuid4().hex[:10]}'
    start = time.monotonic()
    environment = os.environ.copy()
    environment['SIM2SIM_ACTIVE_BUDGET_DIR'] = str(Path(directory).resolve())
    environment['SIM2SIM_RESEARCH_DIR'] = str(Path(directory).resolve())
    allowed=sorted(os.sched_getaffinity(0))[-cpus:]
    launch=['taskset','--cpu-list',','.join(map(str,allowed)),'nice','-n',str(nice),*command]
    process = subprocess.Popen(launch, start_new_session=True, env=environment)
    leader_start=process_start_ticks(process.pid)
    budget.event('job_started', actor=actor, pid=process.pid, command=command,
                 timeout=timeout, reserve=reserve,boot=_boot_id(),
                 process_start_ticks=leader_start,cpus=allowed,nice=nice,supervisor_pid=os.getpid())
    interrupted = None

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = signum

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    reason = 'completed'
    try:
        while True:
            status = budget.heartbeat(actor, max_gap=10.)
            if process.poll() is not None:
                break
            if interrupted or time.monotonic() - start >= timeout or status['remaining_seconds'] <= reserve:
                reason = 'interrupted' if interrupted else 'budget_or_timeout'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(1.)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        try:cleanup=cleanup_group(process.pid,leader_start)
        except Exception as error:cleanup=dict(found=[],remaining=[],error=str(error))
        budget.heartbeat(actor, max_gap=30., stop=True)
        budget.event('job_finished', actor=actor, returncode=process.returncode, reason=reason,
                     process_cleanup=cleanup)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if cleanup['remaining']:raise RuntimeError('Job left live processes: '+str(cleanup['remaining']))
    if cleanup.get('error'):raise RuntimeError('Could not verify process cleanup: '+cleanup['error'])
    return process.returncode if reason == 'completed' else 124


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    sub = p.add_subparsers(dest='action', required=True)
    init = sub.add_parser('init')
    init.add_argument('--seconds', type=float, default=28800.)
    init.add_argument('--initial-seconds', type=float, default=0.)
    init.add_argument('--credit-note', default='')
    pulse = sub.add_parser('heartbeat')
    pulse.add_argument('--actor', default='codex')
    pulse.add_argument('--stop', action='store_true')
    pulse.add_argument('--max-gap', type=float, default=120.,
                       help='Increase only for an explicitly confirmed uninterrupted work interval, never blindly on resume')
    sub.add_parser('status')
    for action in ['run','start']:
        run = sub.add_parser(action)
        run.add_argument('--timeout', type=float, required=True)
        run.add_argument('--reserve', type=float, default=0.)
        run.add_argument('--label', default='job')
        run.add_argument('--cpus',type=int,default=4,help='Maximum CPU affinity size; preserve desktop resources')
        run.add_argument('--nice',type=int,default=10,help='Background scheduling priority 0..19')
        run.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    budget = ActiveBudget(args.directory)
    if args.action == 'init':
        result = budget.initialize(args.seconds, args.initial_seconds, args.credit_note)
    elif args.action == 'heartbeat':
        result = budget.heartbeat(args.actor, stop=args.stop, max_gap=args.max_gap)
    elif args.action == 'status':
        result = budget.status()
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            p.error('run requires a command')
        if args.action=='start':
            process,result=start_supervised(args.directory,command,timeout=args.timeout,
                reserve=args.reserve,label=args.label,cpus=args.cpus,nice=args.nice)
        else:
            raise SystemExit(run_supervised(args.directory, command, timeout=args.timeout,
                                            reserve=args.reserve, label=args.label,cpus=args.cpus,nice=args.nice))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
