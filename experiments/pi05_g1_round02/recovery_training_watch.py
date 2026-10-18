"""Read-only foreground progress watch; imports no training or GPU packages.

Usage (the operator starts this only after choosing the explicit run):
  python -m experiments.pi05_g1_round02.recovery_training_watch \
    --progress /abs/round02/training_run02/progress.jsonl \
    --ledger /abs/round02/gpu_budget.json

Emits immediately and at intervals >=60 seconds. A final record requires the
latest ledger run whose exact --output matches the progress file's parent to
be terminal AND accounting/cleanup closed. --once reads a single snapshot.
Only progress JSONL, ledger JSON and /proc/meminfo are read; nothing is written.
"""

import argparse
import datetime
import json
from pathlib import Path
import time


TERMINAL = frozenset(('completed', 'failed', 'memory_limit', 'host_memory_guard',
                      'budget_expired', 'supervisor_error', 'interrupted'))


def matching_run(ledger, output, command_cwd):
    """Exact argv binding avoids accidentally observing another training run."""
    selected = None
    for row in ledger.get('runs', []):
        command = row.get('command', [])
        if not isinstance(command, list):
            continue
        for index, token in enumerate(command):
            value = None
            if token == '--output' and index + 1 < len(command):
                value = command[index + 1]
            elif isinstance(token, str) and token.startswith('--output='):
                value = token.split('=', 1)[1]
            if not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = command_cwd / candidate
            if candidate.resolve() == output:
                selected = row
    return selected


def is_closed(row):
    if not row or row.get('status') not in TERMINAL or not row.get('ended_utc'):
        return False
    if row.get('accounting_open') or row.get('launcher_cleanup_pending'):
        return False
    cleanup = row.get('docker_cleanup')
    return cleanup is None or cleanup.get('complete') is True


class ProgressReader:
    """Incrementally consume complete JSONL lines, retaining only summaries."""

    def __init__(self, path):
        self.path = Path(path)
        self.identity = None
        self.offset = 0
        self.state = {}
        self.malformed_lines = 0

    def update(self):
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.identity, self.offset, self.state = identity, 0, {}
            self.malformed_lines = 0
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break  # A writer may still be finishing this record.
                self.offset = stream.tell()
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Progress line is not an object')
                except (ValueError, UnicodeError):
                    self.malformed_lines += 1
                    continue
                self.consume(row)
        return dict(self.state)

    def consume(self, row):
        self.state['latest_event'] = row.get('event')
        if 'inherited_round_updates' in row:
            self.state['inherited_round_updates'] = row['inherited_round_updates']
        if 'step' in row:
            self.state['local_step'] = row['step']
            self.state['cumulative_step'] = row.get('cumulative_round_updates',
                self.state.get('inherited_round_updates', 0) + row['step'])
        if row.get('event') == 'optimizer_step':
            self.state['loss'] = row.get('flow_loss')
            self.state['loss_local_step'] = row.get('step')
        if row.get('event') == 'validation' and row.get('complete') is True:
            self.state['last_complete_validation'] = {
                'local_step': row.get('step'), 'metrics': row.get('metrics')}
        if row.get('event') == 'training_stopped' and row.get('final_validation') is not None:
            self.state['last_complete_validation'] = {
                'local_step': row.get('step'), 'metrics': row['final_validation']}
        cleanup = row.get('cpu_staging_cleanup')
        if isinstance(cleanup, list) and cleanup:
            self.state['cpu_staging_trim'] = [
                {'phase': item.get('phase'), 'before_rss_bytes': item.get('before', {}).get('VmRSS_bytes'),
                 'after_rss_bytes': item.get('after', {}).get('VmRSS_bytes'),
                 'trim_available': item.get('trim_available'), 'trim_return': item.get('trim_return')}
                for item in cleanup if isinstance(item, dict)]
            self.state['cpu_staging_trim_local_step'] = row.get('step')


def mem_available(path=Path('/proc/meminfo')):
    for line in Path(path).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def snapshot(reader, ledger_path, command_cwd, meminfo=Path('/proc/meminfo')):
    output = reader.path.resolve().parent
    result = {'schema': 'pi05_g1_recovery_training_watch_v1', 'final': False,
              'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'progress': str(reader.path.resolve()), 'training_output': str(output),
              'local_step': None, 'cumulative_step': None, 'loss': None,
              'last_complete_validation': None, 'cpu_staging_trim': None,
              'host_memavailable_bytes': None, 'ledger_status': None, 'diagnostics': []}
    try:
        result.update(reader.update())
        if reader.malformed_lines:
            result['diagnostics'].append({'malformed_complete_progress_lines': reader.malformed_lines})
    except (OSError, ValueError) as error:
        result['diagnostics'].append({'progress_read_error': str(error)})
    try:
        ledger = json.loads(Path(ledger_path).read_text())
        row = matching_run(ledger, output, Path(command_cwd).resolve())
        if row:
            result['ledger_status'] = row.get('status')
            result['ledger_run_started_utc'] = row.get('started_utc')
            result['ledger_run_ended_utc'] = row.get('ended_utc')
            result['ledger_exit_code'] = row.get('exit_code')
            result['final'] = is_closed(row)
        else:
            result['diagnostics'].append('No ledger run has this exact training --output')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        result['diagnostics'].append({'ledger_read_error': str(error)})
    try:
        result['host_memavailable_bytes'] = mem_available(meminfo)
    except (OSError, ValueError) as error:
        result['diagnostics'].append({'meminfo_read_error': str(error)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--command-cwd', type=Path, default=Path.cwd(),
                        help='Base for relative --output paths in ledger argv')
    parser.add_argument('--interval-seconds', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if not 60 <= args.interval_seconds < float('inf'):
        parser.error('--interval-seconds must be finite and >=60')
    reader = ProgressReader(args.progress)
    while True:
        value = snapshot(reader, args.ledger, args.command_cwd)
        print(json.dumps(value, separators=(',', ':'), allow_nan=False), flush=True)
        if value['final'] or args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
