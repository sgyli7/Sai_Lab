"""One-time preservation of immutable experiment inputs; never hash/read large payloads."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import filecmp
import fcntl
import json
import os
from pathlib import Path
import shutil

PAYLOAD_SUFFIXES = {'.safetensors', '.pt', '.pth', '.parquet', '.mp4', '.bin', '.npz', '.npy'}
METADATA_SUFFIXES = {'.json', '.jsonl', '.md', '.yaml', '.yml', '.toml', '.txt'}
DEFAULT_INPUTS = ('pi05_base', 'pi05_base_pytorch', 'data/lerobot', 'prepared',
                  'eval_cases.json', 'acceptance_v2_frozen.json', 'ENVIRONMENT.md', 'environment_probe.json',
                  'data/README.md', 'data/_download_state/remote_manifest.json', 'data/_download_state/receipt.json')


def preserve_file(source, destination):
    if source.is_symlink() or destination.is_symlink():
        raise RuntimeError('Input preservation refuses symlinks')
    original = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if original.st_dev != destination.parent.stat().st_dev:
        raise RuntimeError('Inputs and backup must share a filesystem')
    metadata = source.suffix in METADATA_SUFFIXES
    hardlink = not metadata and (source.suffix in PAYLOAD_SUFFIXES or original.st_size >= 1024**2)
    if hardlink:
        if destination.exists():
            target = destination.stat()
            if (target.st_dev, target.st_ino) != (original.st_dev, original.st_ino):
                raise FileExistsError(f'Existing backup is not the same payload inode: {destination}')
        else:
            os.link(source, destination)
        method = 'hardlink_no_payload_read'
    else:
        if destination.exists():
            if destination.stat().st_ino == original.st_ino or not filecmp.cmp(source, destination, shallow=False):
                raise FileExistsError(f'Existing metadata backup is not an independent identical copy: {destination}')
        else:
            with source.open('rb') as incoming, destination.open('xb') as output:
                shutil.copyfileobj(incoming, output, length=1024**2)
                output.flush()
                os.fsync(output.fileno())
        method = 'independent_metadata_copy'
    target = destination.stat()
    after = source.stat()
    if (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input changed while preserving: {source}')
    if target.st_size != original.st_size:
        raise RuntimeError(f'Backup size differs: {destination}')
    return {'schema':'pi05_g1_input_archive_v1','archived_utc':datetime.now(timezone.utc).isoformat(),
            'source_path':str(source),'backup_path':str(destination),'size_bytes':original.st_size,
            'source_device':original.st_dev,'source_inode':original.st_ino,'source_mtime_ns':original.st_mtime_ns,
            'backup_device':target.st_dev,'backup_inode':target.st_ino,'method':method}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--archive-root',type=Path,required=True)
    args=parser.parse_args()
    source_root=args.source_root.resolve();root=args.archive_root.resolve()/'inputs'
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'inputs_manifest.jsonl'
    with (root/'inputs_archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        known={}
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                entry=json.loads(line);known[entry['source_path']]=entry
        files=[]
        for relative in DEFAULT_INPUTS:
            path=source_root/relative
            if not path.exists():
                raise FileNotFoundError(path)
            files.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
        new=0;linked_bytes=0;copied_bytes=0
        for source in files:
            target=root/source.relative_to(source_root)
            row=preserve_file(source,target)
            if row['source_path'] in known:
                old=known[row['source_path']]
                for key in ('backup_path','size_bytes','source_device','source_inode','backup_device','backup_inode','method'):
                    if row[key]!=old[key]:
                        raise RuntimeError('Existing input manifest identity differs: '+str(source))
                continue
            with manifest.open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
            new+=1
            if row['method'].startswith('hardlink'):linked_bytes+=row['size_bytes']
            else:copied_bytes+=row['size_bytes']
        print('INPUTS_PRESERVED '+json.dumps({'files':len(files),'new_manifest_entries':new,
              'hardlinked_bytes_no_payload_read':linked_bytes,'copied_metadata_bytes':copied_bytes,
              'manifest':str(manifest),'available_bytes':shutil.disk_usage(root).free}),flush=True)


if __name__=='__main__':
    main()
