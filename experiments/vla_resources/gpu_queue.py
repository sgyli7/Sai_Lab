"""CPU-only FIFO admission outside each peer's owned GPU supervisor."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / '.scratch/vla_gpu_coordination'
TERMINAL = {'completed', 'failed', 'cancelled'}


def utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def transaction(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'queue.json'
        if path.is_symlink():
            raise ValueError('Queue cannot be a symlink')
        queue = json.loads(path.read_text()) if path.exists() else {
            'schema': 'vla_gb10_fifo_queue_v1', 'next_sequence': 1, 'tickets': []}
        if queue.get('schema') != 'vla_gb10_fifo_queue_v1':
            raise ValueError('Unknown queue schema')
        original = json.dumps(queue, sort_keys=True)
        yield queue
        if path.exists() and json.dumps(queue, sort_keys=True) == original:
            return
        temp = root / f'.queue.{os.getpid()}.tmp'
        with temp.open('w') as output:
            json.dump(queue, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)


def owner_check(root, owner):
    doc = json.loads((Path(root) / 'lingbot_request.json').read_text())
    if owner not in (doc['thread_id'], doc['peer_thread_id']):
        raise ValueError('Owner must be one of the two coordinating chats')


def ticket_by_id(queue, ticket_id):
    return next(row for row in queue['tickets'] if row['ticket'] == ticket_id)


def barrier_active(root):
    try:
        doc = json.loads((Path(root) / 'legacy_barrier.json').read_text())
        return doc.get('active') is not False
    except (OSError, ValueError):
        return True


def request(root, owner, stage, max_seconds):
    owner_check(root, owner)
    if not stage or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('A stage and positive finite duration are required')
    with transaction(root) as queue:
        sequence = queue['next_sequence']
        row = {'ticket': f't{sequence:06d}', 'sequence': sequence, 'owner': owner,
               'stage': stage, 'max_seconds': max_seconds, 'status': 'waiting',
               'created_utc': utc()}
        queue['tickets'].append(row)
        queue['next_sequence'] += 1
    return row


def cancel(root, ticket_id, owner):
    owner_check(root, owner)
    with transaction(root) as queue:
        row = ticket_by_id(queue, ticket_id)
        if row['owner'] != owner or row['status'] != 'waiting':
            raise ValueError('Only the owner may cancel a waiting ticket')
        row.update(status='cancelled', closed_utc=utc())
    return row


def status(root):
    with transaction(root) as queue:
        return {'legacy_barrier_active': barrier_active(root), **queue}


def ledger_runs(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = json.loads(path.read_text())['runs']
    if not isinstance(rows, list):
        raise ValueError('Invalid ledger')
    return rows


def cleanup_confirmed(row):
    return (row.get('last_unit_properties', {}).get('ActiveState') in ('inactive', 'failed')
            and row.get('accounting_open') is not True
            and row.get('docker_cleanup', {}).get('complete') is True
            and row.get('launcher_cleanup_pending') is not True
            and row.get('status') not in ('running', 'cleanup_pending'))


def check_command(command, row, ledger):
    prefix = command[:command.index('--')] if '--' in command else command
    def option(name):
        values = []
        for index, value in enumerate(prefix):
            if value == name:
                values.append(prefix[index + 1])
            elif value.startswith(name + '='):
                values.append(value.split('=', 1)[1])
        if len(values) != 1:
            raise ValueError('Exactly one supervisor ' + name + ' is required')
        return values[0]
    duration = float(option('--max-seconds'))
    if not math.isfinite(duration) or not 0 < duration <= row['max_seconds']:
        raise ValueError('Supervisor duration exceeds ticket')
    if option('--stage') != row['stage']:
        raise ValueError('Supervisor stage differs from ticket')
    if Path(option('--ledger')).resolve() != Path(ledger).resolve():
        raise ValueError('Supervisor ledger differs from audited ledger')


def run(root, ticket_id, ledger, command, wait_seconds=None):
    root = Path(root).resolve()
    if not command:
        raise ValueError('Missing supervisor command')
    if wait_seconds is not None and (not math.isfinite(wait_seconds) or wait_seconds < 0):
        raise ValueError('Wait timeout must be nonnegative and finite')
    child = None
    received_signal = None
    def forward(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    gpu = None
    marked = False
    start = time.monotonic()
    before = None
    try:
        while not marked:
            if received_signal is not None:
                return 128 + received_signal
            with transaction(root) as queue:
                row = ticket_by_id(queue, ticket_id)
                if row['status'] != 'waiting':
                    raise ValueError('Ticket is not waiting: ' + row['status'])
                check_command(command, row, ledger)
                pending = [r for r in queue['tickets'] if r['status'] not in TERMINAL]
                ready = (pending[0]['ticket'] == ticket_id and not barrier_active(root)
                         and not any(r['status'] in ('running', 'cleanup_unknown') for r in pending))
                if ready:
                    candidate = (root / 'gb10-exclusive.lock').open('a')
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                    else:
                        gpu = candidate
                        before = len(ledger_runs(ledger))
                        row.update(status='running', wrapper_pid=os.getpid(), started_utc=utc(),
                                   ledger=str(Path(ledger).resolve()))
                        marked = True
            if not marked:
                if wait_seconds is not None and time.monotonic() - start >= wait_seconds:
                    return 75
                time.sleep(.1)
        try:
            child = subprocess.Popen(command)
        except OSError as error:
            with transaction(root) as queue:
                ticket_by_id(queue, ticket_id).update(status='waiting', launch_error=str(error))
            marked = False
            raise
        if received_signal is not None:
            forward(received_signal, None)
        code = child.wait()
        try:
            rows = ledger_runs(ledger)
            cleaned = len(rows) == before + 1 and cleanup_confirmed(rows[-1])
        except (OSError, ValueError, KeyError):
            cleaned = False
        with transaction(root) as queue:
            row = ticket_by_id(queue, ticket_id)
            row.update(status=('completed' if code == 0 else 'failed') if cleaned else 'cleanup_unknown',
                       closed_utc=utc(), exit_code=code, cleanup_confirmed=cleaned,
                       ledger_runs_before=before)
        marked = False
        return (code if code >= 0 else 128 - code) if cleaned else 70
    finally:
        # Unexpected failure leaves the running ticket intact and blocks admission.
        if gpu is not None:
            fcntl.flock(gpu, fcntl.LOCK_UN)
            gpu.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coord-root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='mode', required=True)
    req = sub.add_parser('request')
    req.add_argument('--owner', required=True)
    req.add_argument('--stage', required=True)
    req.add_argument('--max-seconds', type=float, required=True)
    sub.add_parser('status')
    can = sub.add_parser('cancel')
    can.add_argument('--ticket', required=True)
    can.add_argument('--owner', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--ticket', required=True)
    runner.add_argument('--ledger', type=Path, required=True)
    runner.add_argument('--wait-seconds', type=float)
    runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'request':
        result = request(args.coord_root, args.owner, args.stage, args.max_seconds)
    elif args.mode == 'cancel':
        result = cancel(args.coord_root, args.ticket, args.owner)
    elif args.mode == 'status':
        result = status(args.coord_root)
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        raise SystemExit(run(args.coord_root, args.ticket, args.ledger, command, args.wait_seconds))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
