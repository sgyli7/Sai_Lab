"""Preserve completed checkpoints indefinitely with hardlinks, without reading weights.

The trainer atomically renames complete checkpoint directories. Payload links
survive later trainer pruning; metadata is copied so its later edits cannot
rewrite historical evidence. No source or historical output is ever deleted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

PAYLOADS = ('model.safetensors', 'training_state.pt')
GIB = 1024**3


def utc():
    return datetime.now(timezone.utc).isoformat()


def event(name, **fields):
    print('EVENT ' + json.dumps({'event': name, 'utc': utc(), **fields}), flush=True)


def read_manifest(path):
    entries = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            entries[row['source_checkpoint']] = row
    return entries


def archive_checkpoint(source, destination, *, previous=None):
    """Link immutable payload inodes and copy metadata; reject conflicting outputs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    metadata_path = source / 'metadata.json'
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if 'identity' not in metadata or 'step' not in metadata:
        raise ValueError(f'Not a completed training checkpoint: {source}')
    destination.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.stat().st_dev:
        raise RuntimeError(f'Cross-filesystem archive cannot hardlink without reading payload: {source}')
    entries = []
    known = {Path(row['backup_path']).name: row for row in (previous or {}).get('payloads', [])}
    for name in PAYLOADS:
        src, dst = source / name, destination / name
        if src.is_symlink() or dst.is_symlink():
            raise RuntimeError('Checkpoint archive refuses symlinks')
        if src.exists():
            original = src.stat()
            if dst.exists():
                target = dst.stat()
                if (target.st_dev, target.st_ino) != (original.st_dev, original.st_ino):
                    raise FileExistsError(f'Existing archive is not the same payload inode: {dst}')
            else:
                os.link(src, dst)
            target = dst.stat()
            if (target.st_dev, target.st_ino, target.st_size) != (original.st_dev, original.st_ino, original.st_size):
                raise RuntimeError(f'Payload changed during archival: {src}')
        else:
            if not previous or name not in known or not dst.is_file():
                raise FileNotFoundError(f'Unpreserved payload disappeared before archival: {src}')
            target = dst.stat()
            old = known[name]
            if (target.st_dev, target.st_ino, target.st_size) != (old['device'], old['inode'], old['size_bytes']):
                raise RuntimeError(f'Existing retained payload identity changed: {dst}')
        entries.append({'source_path': str(src), 'backup_path': str(dst), 'size_bytes': target.st_size,
                        'device': target.st_dev, 'inode': target.st_ino, 'hardlink_count': target.st_nlink})
    backup_metadata = destination / 'metadata.json'
    if previous:
        if not backup_metadata.is_file() or hashlib.sha256(backup_metadata.read_bytes()).hexdigest() != previous['metadata_sha256']:
            raise RuntimeError(f'Historical metadata changed: {backup_metadata}')
        return previous
    if backup_metadata.exists():
        if backup_metadata.read_bytes() != metadata_bytes:
            raise FileExistsError(f'Existing historical metadata differs: {backup_metadata}')
        if backup_metadata.stat().st_ino == metadata_path.stat().st_ino:
            raise RuntimeError('Metadata must be an independent copy, never a hardlink')
    else:
        with backup_metadata.open('xb') as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    return {'schema': 'pi05_g1_checkpoint_archive_v1', 'archived_utc': utc(),
            'source_checkpoint': str(source), 'backup_checkpoint': str(destination),
            'method': 'same-filesystem payload hardlinks; independent metadata copy; no payload hash/read',
            'payloads': entries, 'checkpoint_metadata': metadata,
            'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest()}


def scan(sources, archive_root, entries, manifest):
    count = 0
    for root in map(Path, sources):
        if not root.exists():
            continue
        if root.stat().st_dev != archive_root.stat().st_dev:
            raise RuntimeError(f'Source and backup root must share a filesystem: {root}')
        for source in sorted(root.iterdir()):
            if not source.is_dir() or source.name.endswith('.incomplete') or not (source / 'metadata.json').is_file():
                continue
            key = str(source.resolve())
            row = archive_checkpoint(source, archive_root / 'checkpoints' / root.name / source.name,
                                     previous=entries.get(key))
            if key not in entries:
                with manifest.open('a') as output:
                    output.write(json.dumps(row, sort_keys=True) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                entries[key] = row
                count += 1
                event('checkpoint_preserved', source=key, backup=row['backup_checkpoint'],
                      payload_bytes=sum(item['size_bytes'] for item in row['payloads']),
                      step=row['checkpoint_metadata']['step'])
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--max-hours', type=float, default=4.)
    parser.add_argument('--poll-seconds', type=float, default=5.)
    parser.add_argument('--low-disk-gib', type=float, default=60.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.max_hours <= 0 or not 0 < args.poll_seconds <= 60 or args.low_disk_gib <= 0:
        parser.error('Positive duration/disk threshold and poll-seconds in (0,60] required')
    root = args.archive_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stop_file = args.stop_file or root / 'ARCHIVE_STOP'
    manifest = root / 'archive_manifest.jsonl'
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (root / 'archive_worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = read_manifest(manifest)
        started = last_heartbeat = time.monotonic()
        deadline = started + args.max_hours * 3600
        event('archive_started', pid=os.getpid(), archive_root=str(root), stop_file=str(stop_file),
              sources=[str(p.resolve()) for p in args.sources], existing_checkpoints=len(entries),
              poll_seconds=args.poll_seconds, max_hours=args.max_hours)
        low_warned = False
        while True:
            scan(args.sources, root, entries, manifest)
            free = shutil.disk_usage(root).free
            if free < args.low_disk_gib * GIB and not low_warned:
                warning = {'utc': utc(), 'available_bytes': free, 'threshold_bytes': int(args.low_disk_gib*GIB),
                           'action': 'Notify root to manage the owned training run; archive worker never deletes history or kills GPU jobs.'}
                alert = root / ('low_disk_' + str(time.time_ns()) + '.json')
                with alert.open('x') as output:
                    json.dump(warning, output, indent=2)
                event('low_disk_alert', alert_file=str(alert), **warning)
                low_warned = True
            if args.once or stopping or stop_file.exists() or time.monotonic() >= deadline:
                event('archive_finished', checkpoints=len(entries), available_bytes=free,
                      reason='once' if args.once else 'signal' if stopping else 'stop_file' if stop_file.exists() else 'time_limit')
                break
            if time.monotonic() - last_heartbeat >= 60:
                event('archive_heartbeat', checkpoints=len(entries), available_bytes=free)
                last_heartbeat = time.monotonic()
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
