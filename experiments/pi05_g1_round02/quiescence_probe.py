"""Read-only final GPU shutdown evidence. Never starts/stops/signals workloads.

After every owned GPU job ends:
  python -m experiments.pi05_g1_round02.quiescence_probe \
    --ledger /abs/gpu_budget.json --output /abs/NEW_quiescence.json

Creates only the new report and adjacent command-evidence directory. Exit 0
means this point-in-time probe passed, not a guarantee against future launches.
The supervisor uses systemd-run --user --service-type=exec (not a scope).
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess


SCHEMA = 'pi05_g1_round02_gpu_quiescence_v1'
TERMINAL = {'completed', 'failed', 'memory_limit', 'host_memory_guard',
            'budget_expired', 'supervisor_error', 'interrupted'}
UNIT = re.compile(r'pi05-g1-owned-[0-9a-f]{32}\.service')
TOKEN = re.compile(r'[0-9a-f]{32}')
CID = re.compile(r'[0-9a-f]{12,64}')
DISPLAY_BACKGROUNDS = frozenset(('gnome-remote-desktop', 'gnome-remote-desktop-daemon'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(ledger_path, output):
    ledger_path, output = Path(ledger_path).resolve(), Path(output).resolve()
    evidence = output.with_name(output.stem + '.evidence')
    if output.exists() or evidence.exists():
        raise FileExistsError('Report and evidence directory must both be new')
    original = ledger_path.read_bytes()
    ledger = json.loads(original)
    evidence.mkdir(parents=True)
    result = {'schema': SCHEMA, 'ledger': str(ledger_path),
              'ledger_sha256': hashlib.sha256(original).hexdigest(),
              'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'passed': False, 'supervision_scope': 'systemd user transient service',
              'active_owned_units': [], 'active_owned_container_ids': [],
              'accounting_open_tokens': [], 'total_gpu_seconds': 0.0,
              'compute_processes': [], 'allowed_display_background': [],
              'unexpected_compute_processes': [], 'existing_stopped_container_ids': [],
              'unit_checks': [], 'container_checks': [], 'errors': [], 'evidence_files': {},
              'scope_limit': 'Point-in-time ledger/owned-service/container/compute query; no signals, no physics or GPU workload launch. No guarantee against subsequent launches.'}
    counter = 0

    def bind(path):
        result['evidence_files'][str(path.resolve())] = sha(path)

    ledger_copy = evidence / 'ledger_snapshot.json'
    ledger_copy.write_bytes(original)
    bind(ledger_copy)

    def call(argv):
        nonlocal counter
        counter += 1
        prefix = evidence / f'{counter:03d}_{Path(argv[0]).name}'
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
            error = None
        except (OSError, subprocess.TimeoutExpired) as exc:
            stdout, stderr, code, error = '', '', None, repr(exc)
            if isinstance(exc, subprocess.TimeoutExpired):
                stdout, stderr = exc.stdout or '', exc.stderr or ''
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8', errors='replace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode('utf-8', errors='replace')
        for suffix, text in (('.stdout.txt', stdout), ('.stderr.txt', stderr)):
            path = Path(str(prefix) + suffix)
            path.write_text(text)
            bind(path)
        receipt = Path(str(prefix) + '.command.json')
        receipt.write_text(json.dumps({'argv': argv, 'returncode': code, 'error': error}, indent=2) + '\n')
        bind(receipt)
        if error:
            result['errors'].append({'command': argv, 'error': error})
        return code, stdout, stderr

    runs = ledger.get('runs', [])
    if not isinstance(runs, list) or not runs:
        result['errors'].append('Ledger must contain nonempty runs')
        runs = []
    if ledger.get('accounting_open') or ledger.get('launcher_cleanup_pending'):
        result['accounting_open_tokens'].append('ledger-global')
    units, owned = set(), {}
    for index, row in enumerate(runs):
        token = row.get('budget_token')
        label = token or f'run-index-{index}'
        elapsed = row.get('elapsed_seconds')
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            result['errors'].append(f'Invalid elapsed_seconds: {label}')
        else:
            result['total_gpu_seconds'] += elapsed
        if (row.get('status') not in TERMINAL or not row.get('ended_utc') or
                row.get('accounting_open') or row.get('launcher_cleanup_pending') or
                (row.get('docker_cleanup') is not None and row['docker_cleanup'].get('complete') is not True)):
            result['accounting_open_tokens'].append(label)
        unit = row.get('owned_unit')
        if not isinstance(unit, str) or not UNIT.fullmatch(unit):
            result['errors'].append(f'Missing or invalid owned_unit: {label}')
        else:
            units.add(unit)
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            result['errors'].append(f'Missing or invalid budget_token: {label}')
            continue
        ids = owned.setdefault(token, set())
        for section in ('docker_cleanup', 'docker_memory_emergency_cleanup'):
            for item in (row.get(section) or {}).get('containers', []):
                if item.get('container_id'):
                    ids.add(item['container_id'])
        directory = row.get('ownership_receipt_dir')
        if directory:
            for path in sorted(Path(directory).glob('launch-*.json')):
                try:
                    receipt = json.loads(path.read_text())
                    bind(path)
                    if receipt.get('schema') != 'pi05_g1_docker_ownership_v1' or receipt.get('budget_token') != token:
                        raise ValueError('Ownership receipt schema/token mismatch')
                    cid = receipt.get('container_id')
                    if not cid and receipt.get('cidfile'):
                        cidfile = Path(receipt['cidfile'])
                        if cidfile.is_file():
                            cid = cidfile.read_text().strip()
                            bind(cidfile)
                    if cid:
                        ids.add(cid)
                    elif receipt.get('status') != 'finished':
                        raise ValueError('Unresolved container launch without CID')
                except (OSError, ValueError, TypeError) as exc:
                    result['errors'].append({'ownership_receipt': str(path), 'error': repr(exc)})

    for unit in sorted(units):
        code, stdout, stderr = call(['systemctl', '--user', 'show', unit,
                                    '--property=ActiveState,SubState,Result,LoadState'])
        props = dict(line.split('=', 1) for line in stdout.splitlines() if '=' in line)
        missing = code not in (0, None) and re.search(r'\bunit\b[^\n]*\b(?:not found|not be found)\b', stderr, re.I)
        inactive = code == 0 and props.get('ActiveState') in ('inactive', 'failed')
        result['unit_checks'].append({'unit': unit, 'properties': props, 'collected': bool(missing)})
        if not (missing or inactive):
            result['active_owned_units'].append(unit)
            if code != 0 or not props.get('ActiveState'):
                result['errors'].append(f'Cannot verify user unit inactive: {unit}')

    for token, ids in sorted(owned.items()):
        code, stdout, _ = call(['docker', 'ps', '--all', '--quiet', '--no-trunc',
                                '--filter', 'label=sai.pi05_g1.budget=' + token])
        if code != 0:
            result['errors'].append(f'Cannot enumerate token-owned containers: {token}')
        else:
            ids.update(stdout.split())
        for cid in sorted(ids):
            if not isinstance(cid, str) or not CID.fullmatch(cid):
                result['errors'].append(f'Invalid container ID for {token}: {cid}')
                continue
            code, stdout, stderr = call(['docker', 'inspect', cid])
            if code not in (0, None) and re.search(r'\bno such (?:object|container)(?:\s*:|\s*$)', stderr, re.I):
                result['container_checks'].append({'container_id': cid, 'removed': True, 'budget_token': token})
                continue
            try:
                if code != 0:
                    raise ValueError('docker inspect did not succeed or explicitly prove absence')
                data = json.loads(stdout)[0]
                if data['Config']['Labels'].get('sai.pi05_g1.budget') != token:
                    raise ValueError('Container ownership label mismatch')
                state = data['State']
                result['container_checks'].append({'container_id': cid, 'state': state, 'budget_token': token, 'removed': False})
                if state.get('Running') or state.get('Paused') or state.get('Restarting'):
                    result['active_owned_container_ids'].append(cid)
                elif state.get('Status') in ('exited', 'dead', 'created'):
                    result['existing_stopped_container_ids'].append(cid)
                else:
                    raise ValueError('Unrecognized container state')
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                result['errors'].append({'container_id': cid, 'error': repr(exc)})

    code, stdout, _ = call(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                            '--format=csv,noheader,nounits'])
    if code != 0:
        result['errors'].append('Cannot query NVIDIA compute processes')
    else:
        for fields in csv.reader(io.StringIO(stdout)):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 4 or not fields[1].strip().isdigit():
                result['errors'].append({'unparsed_compute_row': fields})
                continue
            gpu, pid, name, memory = (field.strip() for field in fields)
            item = {'gpu_uuid': gpu, 'pid': int(pid), 'process_name': name, 'used_gpu_memory_mib': memory}
            result['compute_processes'].append(item)
            if Path(name).name in DISPLAY_BACKGROUNDS:
                result['allowed_display_background'].append(item)
            else:
                result['unexpected_compute_processes'].append(item)
    if ledger_path.read_bytes() != original:
        result['errors'].append('Ledger changed during probe; evidence is not a closed consistent snapshot')
    result['passed'] = not any(result[key] for key in ('errors', 'active_owned_units',
        'active_owned_container_ids', 'accounting_open_tokens', 'unexpected_compute_processes'))
    result['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.ledger, args.output)
    print(json.dumps({'passed': result['passed'], 'output': str(args.output.resolve()),
                      'total_gpu_seconds': result['total_gpu_seconds'], 'errors': result['errors']}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
