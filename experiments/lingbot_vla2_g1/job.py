"""Verify an immutable CPU-prepared work item before entering its workload."""
import argparse
import json
import os
from pathlib import Path

from .data import OWN, sha


def verify(path):
    path = Path(path).absolute()
    if not path.is_relative_to(OWN):
        raise ValueError('LingBot-owned work item required')
    job = json.loads(path.read_text())
    if job.get('schema') != 'lingbot_vla2_verified_job_v1':
        raise ValueError('Unsupported work item')
    if Path(job['result_file']).exists():
        raise FileExistsError('Preserve prior work item result')
    for name, expected in job['frozen_files'].items():
        if sha(name) != expected:
            raise ValueError('Frozen work item input changed: ' + name)
    for name, identity in job.get('immutable_payloads', {}).items():
        payload = Path(name)
        if payload.stat().st_size != identity['bytes'] or sha(payload) != identity['sha256']:
            raise ValueError('Immutable payload changed: ' + name)
        with payload.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    argv = job['workload_argv']
    if len(argv) < 3 or Path(argv[0]).absolute() != OWN / 'venv/bin/python' or argv[1] != '-m' or not argv[2].startswith('experiments.lingbot_vla2_g1.'):
        raise ValueError('Explicit LingBot workload entry required')
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    job = verify(args.job)
    print(json.dumps({'event': 'frozen_work_item_verified', 'job': str(args.job.absolute()), 'job_sha256': sha(args.job), 'files': len(job['frozen_files']), 'payloads': len(job.get('immutable_payloads', {}))}), flush=True)
    if not args.verify_only:
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        os.execve(job['workload_argv'][0], job['workload_argv'], env)


if __name__ == '__main__':
    main()
