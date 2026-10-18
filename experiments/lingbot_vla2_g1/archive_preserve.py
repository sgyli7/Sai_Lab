"""Archive owned history by atomic rename while keeping every old path usable.

This module is preparation only. Its CLI requires a real, complete, successful
twenty-case report and all owned GPU work closed before any historical move.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .backup_round07 import BACKUP
from .data import OWN, ROOT, sha


def inventory(folder):
    folder = Path(folder).resolve()
    result = {}
    def visit(current):
        for child in sorted(current.iterdir()):
            name = str(child.relative_to(folder))
            if child.is_symlink():
                result[name] = {'symlink': os.readlink(child)}
            elif child.is_dir():
                result[name] = {'directory': True}
                visit(child)
            elif child.is_file():
                result[name] = {'bytes': child.stat().st_size, 'sha256': sha(child)}
            else:
                raise ValueError('Unsupported historical filesystem entry: ' + str(child))
    visit(folder)
    return result


def move_with_alias(source, destination):
    source, destination = [Path(p).absolute() for p in (source, destination)]
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise ValueError('Existing historical alias points elsewhere')
        return inventory(destination)
    if not source.is_dir() or destination.exists() or destination.is_symlink():
        raise ValueError('Fresh historical destination and regular source directory required')
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError('Archive must use one filesystem; no partial data copy')
    expected = inventory(source)
    temporary = source.with_name(source.name + '.archive_alias_pending')
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError('Preserve interrupted alias evidence')
    source.rename(destination)
    try:
        temporary.symlink_to(destination, target_is_directory=True)
        os.replace(temporary, source)
        if inventory(source) != expected:
            raise ValueError('Historical bytes or nested aliases changed')
    except BaseException:
        if temporary.is_symlink():
            temporary.unlink()
        if source.is_symlink():
            source.unlink()
        if not source.exists():
            destination.rename(source)
        raise
    return expected


def require_acceptance(report):
    if (report.get('synthetic') is True or report.get('status') != 'completed'
            or report.get('expected_cases') != 20 or report.get('completed_evidence_cases') != 20
            or report.get('goal_performance_passed') is not True
            or report.get('target_success_count') != 16 or report.get('success_count', 0) < 16
            or report.get('errors') or report.get('actual_robot_success_claimed') is not False):
        raise ValueError('Actual all20 valid and at least16 full successes required before final archive')


def save(path, value):
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def archive(report_path):
    report_path = Path(report_path).absolute()
    if not report_path.is_relative_to(OWN):
        raise ValueError('Owned actual formal report required')
    require_acceptance(json.loads(report_path.read_text()))
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    owner = '01a145c6-0b86-7623-a869-086b33292199'
    if any(t['owner'] == owner and t['status'] in ('running', 'waiting') for t in queue['tickets']):
        raise ValueError('Close every owned queue ticket before archival')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    if any(not cleanup_confirmed(row) for row in ledger['runs']):
        raise ValueError('Every owned GPU process and container must be cleaned')
    reserve = 256 * 1024**2
    combined = sum(p.stat().st_size for root in (OWN, BACKUP) for p in root.rglob('*') if p.is_file())
    if shutil.disk_usage(OWN).free - reserve < 32 * 1024**3 or combined + reserve > 64 * 1024**3:
        raise RuntimeError('Preserve global32GiB floor, combined64GiB cap and archive metadata reserve')
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
    dated_parent = BACKUP.parent / date
    dated_parent.mkdir(exist_ok=True)
    dated = dated_parent / BACKUP.name
    relocation = move_with_alias(BACKUP, dated)
    history = BACKUP / 'archive_after_actual_acceptance'
    history.mkdir(exist_ok=True)
    journal = OWN / 'permanent_history_archive_receipt.json'
    if journal.exists():
        state = json.loads(journal.read_text())
        if state['formal_report_sha256'] != sha(report_path) or state['dated_backup_root'] != str(dated):
            raise ValueError('Existing archive belongs to another accepted report or date')
    else:
        excluded = {'venv', 'upstream', 'runtime_source', 'inductor_cache', 'triton_cache', 'owned_runs', 'sim_launchers'}
        folders = [p for p in sorted(OWN.iterdir()) if p.is_dir() and not p.is_symlink() and p.name not in excluded]
        folders += [p for parent in ('owned_runs', 'sim_launchers') for p in sorted((OWN / parent).iterdir())
                    if p.is_dir() and not p.is_symlink()]
        state = {'schema': 'lingbot_complete_history_permanent_archive_v1', 'status': 'in_progress',
                 'formal_report': str(report_path), 'formal_report_sha256': sha(report_path),
                 'dated_backup_root': str(dated), 'backup_old_path': str(BACKUP),
                 'backup_relocation_inventory': relocation,
                 'all_unique_models_recordings_sources_results_retained': True,
                 'plan': [str(p) for p in folders], 'moved': {}, 'root_files_copied': {}}
        save(journal, state)
    for name in state['plan']:
        source = Path(name)
        destination = history / 'working_history' / source.relative_to(OWN)
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = move_with_alias(source, destination)
        if name in state['moved'] and state['moved'][name]['inventory'] != values:
            raise ValueError('Previously archived history changed')
        state['moved'][name] = {'permanent_path': str(destination), 'inventory': values}
        save(journal, state)
    metadata = history / 'root_metadata'
    metadata.mkdir(exist_ok=True)
    for source in sorted(OWN.iterdir()):
        if not source.is_file() or source.is_symlink() or source == journal:
            continue
        destination = metadata / source.name
        expected = sha(source)
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError('Previously preserved root metadata changed: ' + str(source))
        else:
            with destination.open('xb') as writer, source.open('rb') as reader:
                shutil.copyfileobj(reader, writer, 1024**2)
            if sha(destination) != expected:
                raise ValueError('Root metadata copy differs')
        state['root_files_copied'][str(source)] = {'path': str(destination), 'sha256': expected}
    sources = history / 'current_task_sources'
    source_root = ROOT / 'experiments/lingbot_vla2_g1'
    state['current_task_sources_copied'] = {}
    for source in sorted(source_root.rglob('*.py')):
        destination = sources / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError('Previously preserved task source changed')
        else:
            with destination.open('xb') as writer, source.open('rb') as reader:
                shutil.copyfileobj(reader, writer, 1024**2)
            if sha(destination) != expected:
                raise ValueError('Task source copy differs')
        state['current_task_sources_copied'][str(source)] = {'path': str(destination), 'sha256': expected}
    state.update(status='completed', actual_robot_success_claimed=False,
                 completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(journal, state)
    save(history / 'archive_manifest.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-accepted-formal-report', type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.actual_accepted_formal_report)
    print(json.dumps({'status': result['status'], 'dated_backup_root': result['dated_backup_root'],
                      'all_unique_models_and_evidence_retained': True}))


if __name__ == '__main__':
    main()
