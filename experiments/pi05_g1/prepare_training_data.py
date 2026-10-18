"""Audit complete local LeRobot episodes and split before fitting normalization.

No download or GPU activity. Incomplete/invalid episodes never enter the split.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import re
import sys
import numpy as np

REVISION = '37ba80a99486a4c308477854f54b4e82e77777dc'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def grouped_split(records, seed=42):
    """Union duplicate data OR complete video content; assign whole groups."""
    ids = sorted(int(r['episode_id']) for r in records)
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for record in records:
        eid = int(record['episode_id'])
        keys = [('data', record['content_sha256'])]
        # A whole camera stream can prove duplicate demonstrations even if indices differ.
        keys += [('video', digest) for digest in record['video_sha256'].values()]
        for key in keys:
            if key in seen:
                parent[find(eid)] = find(seen[key])
            else:
                seen[key] = eid
    groups = {}
    for eid in ids:
        groups.setdefault(find(eid), []).append(eid)
    values = sorted(groups.values(), key=lambda g: g[0])
    if len(values) < 3:
        raise ValueError('Need at least three independent trajectory groups for train/validation/test')
    random.Random(seed).shuffle(values)
    # Reserve a group for each holdout; then target 80/10/10 by episode counts.
    split = {'train': [], 'validation': [], 'test': []}
    targets = {'train': .8 * len(ids), 'validation': .1 * len(ids), 'test': .1 * len(ids)}
    for name in ('validation', 'test'):
        split[name].extend(values.pop())
    for group in sorted(values, key=len, reverse=True):
        name = max(split, key=lambda k: targets[k] - len(split[k]))
        split[name].extend(group)
    return {name: sorted(items) for name, items in split.items()}


def fit_norm(episodes, train_ids):
    result = {}
    for key in ('state', 'actions'):
        array = np.concatenate([np.asarray(episodes[eid][key], dtype=np.float64) for eid in train_ids])
        if array.ndim != 2 or array.shape[1] != 32 or not np.isfinite(array).all():
            raise ValueError(f'{key} must be finite N x 32')
        result[key] = {'mean': array.mean(0).tolist(), 'std': array.std(0).tolist(),
                       'q01': np.quantile(array, .01, axis=0).tolist(),
                       'q99': np.quantile(array, .99, axis=0).tolist()}
    return result


def check_video_alignment(timestamps, frames, fps):
    """Require the same nearest 50 Hz frame, including the episode origin."""
    times = np.asarray(timestamps, dtype=np.float64)
    if frames != len(times):
        raise ValueError('video frame count mismatch')
    if not np.isfinite(fps) or abs(fps - 50.) > .01:
        raise ValueError('video must use the frozen 50 Hz control timing')
    expected = np.arange(frames, dtype=np.float64) / fps
    if not np.isfinite(times).all() or np.any(np.abs(times - expected) > .5/fps + 1e-7):
        raise ValueError('video/parquet timestamp alignment exceeds half a control frame')


def probe_video(path):
    # PyAV is shared with training; decode real frames rather than trusting metadata counts.
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        timestamps = [float(frame.pts * stream.time_base) for frame in container.decode(stream)]
    if not timestamps or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('video contains missing, nonfinite or nonmonotonic decoded timestamps')
    check_video_alignment(timestamps, len(timestamps), fps)
    return {'frames': len(timestamps), 'fps': fps, 'duration': timestamps[-1] + 1/fps,
            'first_frame_timestamp': timestamps[0], 'last_frame_timestamp': timestamps[-1]}


def load_encoder():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module('contract').encode_row


def prepare(dataset_root, output, encoder=None):
    import pyarrow.parquet as pq
    root, output = Path(dataset_root), Path(output)
    if (root / 'lerobot/meta/info.json').is_file():
        root = root / 'lerobot'
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / 'meta/info.json').read_text())
    encoder = encoder or load_encoder()
    video_keys = sorted(k for k, v in info['features'].items() if v.get('dtype') == 'video')
    if not video_keys:
        raise ValueError('No video features in dataset metadata')
    episode_meta = {}
    for file in (root / 'meta').glob('episodes*.jsonl'):
        for line in file.read_text().splitlines():
            entry = json.loads(line)
            episode_meta[int(entry['episode_index'])] = entry
    paths = sorted(root.glob('data/**/episode_*.parquet'))
    records, rejected, episodes = [], [], {}
    for path in paths:
        eid = int(re.search(r'episode_(\d+)', path.name).group(1))
        record = {'episode_id': eid, 'parquet': str(path.relative_to(root)), 'parquet_sha256': sha256(path),
                  'video_sha256': {}, 'videos': {}, 'success_labels': {}, 'errors': []}
        try:
            rows = pq.read_table(path).to_pylist()
            record['length'] = len(rows)
            if not rows:
                raise ValueError('empty episode')
            timestamps = np.asarray([r['timestamp'] for r in rows], dtype=float)
            record['duplicate_timestamps'] = int(len(timestamps) - len(np.unique(timestamps)))
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                record['errors'].append('nonfinite, duplicate or nonmonotonic timestamps')
            record['terminal_signals'] = {key: rows[-1][key] for key in ('next.reward', 'next.done') if key in rows[-1]}
            if 'observation.img_state_delta' in rows[0]:
                deltas = np.asarray([r['observation.img_state_delta'] for r in rows], dtype=float)
                if not np.isfinite(deltas).all():
                    record['errors'].append('nonfinite image/state timestamp delta')
                else:
                    record['image_state_delta'] = {'min': float(deltas.min()), 'max': float(deltas.max()),
                                                   'nonzero_count': int(np.count_nonzero(deltas))}
                    if np.any(np.abs(deltas) > .01 + 1e-7):
                        record['errors'].append('image/state timestamp delta exceeds half control frame; explicit alignment required')
            record['metadata_length'] = episode_meta.get(eid, {}).get('length')
            if record['metadata_length'] is not None and record['metadata_length'] != len(rows):
                record['errors'].append('episode metadata length mismatch')
            for key, value in episode_meta.get(eid, {}).items():
                if 'success' in key:
                    record['success_labels'][key] = value
            for key in rows[0]:
                if 'success' in key:
                    record['success_labels'][key] = sorted(set(str(r[key]) for r in rows))
            encoded = [encoder(row) for row in rows]
            arrays = {key: np.asarray([r[key] for r in encoded], dtype=np.float64) for key in ('state', 'actions')}
            for key, array in arrays.items():
                if array.shape != (len(rows), 32) or not np.isfinite(array).all():
                    record['errors'].append(f'invalid {key}: requires finite N x 32')
            h = hashlib.sha256()
            for key in ('state', 'actions'):
                h.update(arrays[key].astype('<f8').tobytes())
            h.update((timestamps - timestamps[0]).astype('<f8').tobytes())
            record['content_sha256'] = h.hexdigest()
            for key in video_keys:
                rel = info['video_path'].format(episode_chunk=eid // info.get('chunks_size', 1000),
                                               video_key=key, episode_index=eid)
                video = root / rel
                if not video.is_file():
                    record['errors'].append(f'missing video: {rel}')
                    continue
                record['video_sha256'][key] = sha256(video)
                details = probe_video(video)
                record['videos'][key] = {'path': rel, **details}
                # This source is synchronous: one encoded camera frame per parquet row.
                try:
                    check_video_alignment(timestamps, details['frames'], details['fps'])
                except ValueError as exc:
                    record['errors'].append(f'{key}: {exc}')
            if record['errors']:
                rejected.append(record)
            else:
                episodes[eid] = arrays
                records.append(record)
        except Exception as exc:
            record['errors'].append(f'{type(exc).__name__}: {exc}')
            rejected.append(record)
    declared = info.get('total_episodes')
    readme_path = next((p for p in (root / 'README.md', root.parent / 'README.md') if p.exists()), None)
    readme = readme_path.read_text() if readme_path else ''
    report = {'revision': REVISION, 'metadata_total_episodes': declared, 'metadata_splits': info.get('splits'),
              'readme_episode_claims': re.findall(r'.{0,60}(?:200|251|episodes|轨迹).{0,80}', readme),
              'metadata_episode_entries': len(episode_meta),
              'metadata_total_frames': info.get('total_frames'),
              'actual_frames': sum(r['length'] for r in records),
              'declared_index_range_gaps': sorted(set(range(declared or 0)) - {r['episode_id'] for r in records + rejected}),
              'actual_parquet_files': len(paths), 'actual_complete_episodes': len(records),
              'missing_parquet_episode_ids': sorted(set(episode_meta) - {r['episode_id'] for r in records + rejected}),
              'complete': records, 'rejected': rejected,
              'episodes_without_success_labels': [r['episode_id'] for r in records if not r['success_labels']],
              'alignment_validation': 'Decoded frame count and per-row zero-origin timestamp/FPS agreement within half a control frame; pixel-level synchronization requires replay.',
              'unreferenced_video_files': sorted(str(p.relative_to(root)) for p in root.glob('videos/**/*.mp4')
                  if str(p.relative_to(root)) not in {v['path'] for r in records + rejected for v in r['videos'].values()}),
              'unique_trajectory_content_hashes': len({r['content_sha256'] for r in records}),
              'unique_video_hashes': len({h for r in records for h in r['video_sha256'].values()}),
              'count_conflict': declared != len(records),
              'note': 'Only audited complete parquet/video pairs count; metadata and README are claims. Declared index range gaps need not indicate missing upstream assets: compare pinned remote_manifest.json.'}
    (output / 'preparation_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    split = grouped_split(records)
    manifest = {'revision': REVISION, 'seed': 42, 'dataset_root': str(root.resolve()), **split,
                'metadata_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted((root / 'meta').glob('*')) if p.is_file()},
                'readme_sha256': sha256(readme_path) if readme_path else None,
                'files': records, 'actual_ratios': {k: len(v)/len(records) for k,v in split.items()}}
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    norms = fit_norm(episodes, split['train'])
    (output / 'train_norm_stats.json').write_text(json.dumps(norms, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.dataset_root, args.output_dir)
    print(json.dumps({k: report[k] for k in ('actual_parquet_files', 'actual_complete_episodes', 'count_conflict')}))
