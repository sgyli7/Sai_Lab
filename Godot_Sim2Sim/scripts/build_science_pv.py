"""Edit native fixed-camera Sai footage at wall-clock speed; emit a 15 s homepage GIF."""
from bisect import bisect_left
from fractions import Fraction
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import av
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cut(root, cut):
    path = root / cut['source'] / 'hub.json'
    source = json.loads(path.read_text())
    frames = source['frames']
    assert source['scene'] == 'science_station' and source['active_robot'] == 'sai'
    assert all(f['robot'] == 'sai' for f in frames)
    assert len({json.dumps([f['camera_position'], f['camera_basis'], f['fov']]) for f in frames}) == 1
    base = frames[0]['milliseconds'] / 1000
    times = [f['milliseconds'] / 1000 - base for f in frames]
    assert 0 <= cut['start'] and cut['start'] + cut['duration'] <= times[-1]
    selected = [f for t, f in zip(times, frames) if cut['start'] <= t <= cut['start'] + cut['duration']]
    a, b = selected[0], selected[-1]
    samples = [s for s in source['samples'] if a['hub_seconds'] <= s['time'] <= b['hub_seconds']]
    distance = sum(math.dist(a['position'], b['position']) for a, b in zip(samples, samples[1:]))
    evidence = dict(**cut, source_json_sha256=digest(path), native_frames=len(selected),
                    camera_position=a['camera_position'], camera_basis=a['camera_basis'], fov=a['fov'],
                    simulation_seconds=b['hub_seconds']-a['hub_seconds'],
                    wall_seconds=(b['milliseconds']-a['milliseconds'])/1000,
                    chassis_distance_m=distance,
                    source_frame_sha256=[digest(f['file']) for f in selected])
    if cut.get('require_delivery', False):
        deliveries = [d for s in source['grab_sessions'] for d in s['deliveries']]
        assert deliveries and all(d['success'] and d['captured'] and d['released'] and d['supported'] for d in deliveries)
        evidence['deliveries'] = deliveries
    return frames, times, evidence


def encode(root, cuts, output):
    container = av.open(str(output), 'w')
    stream = container.add_stream('libx264', rate=30)
    stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'yuv420p'
    stream.time_base = Fraction(1, 30)
    stream.codec_context.time_base = Fraction(1, 30)
    stream.options = {'crf': '19', 'preset': 'fast', 'threads': '2'}
    evidence = []
    tick = 0
    for cut in cuts:
        frames, times, record = load_cut(root, cut)
        evidence.append(record)
        for i in range(round(cut['duration'] * 30)):
            t = cut['start'] + i/30
            index = min(bisect_left(times, t), len(times)-1)
            if index and abs(times[index-1]-t) < abs(times[index]-t): index -= 1
            with Image.open(frames[index]['file']) as image:
                assert image.size == (1920, 1080)
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = tick, Fraction(1, 30)
            tick += 1
            for packet in stream.encode(frame): container.mux(packet)
    for packet in stream.encode(): container.mux(packet)
    container.close()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edit', type=Path, default=Path('docs/science-station/pv-edit.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/science-station/media'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.edit.read_text())
    assert abs(sum(c['duration'] for c in recipe['pv']) - 15) < .001
    args.output.mkdir(parents=True, exist_ok=True)
    movie = args.output / 'sai-windpass-15s.mp4'
    evidence = encode(root, recipe['pv'], movie)
    encode(root, recipe['film'], args.output / 'sai-windpass-film.mp4')
    ffmpeg = shutil.which('ffmpeg')
    gif = args.output / 'sai-windpass-15s.gif'
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-threads', '2', '-i', str(movie),
                    '-filter_complex_threads', '1', '-filter_complex',
                    'fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4',
                    '-loop', '0', str(gif)], check=True)
    with Image.open(gif) as image:
        duration = 0
        for i in range(image.n_frames):
            image.seek(i); duration += image.info.get('duration', 0)
        assert image.size == (960, 540) and image.n_frames == 180 and abs(duration/1000-15) < .02
    assert gif.stat().st_size < 15_000_000
    manifest_path = (root / recipe['pv'][0]['source']).parent / 'source-manifest.json'
    world_source = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    report = dict(method='Native Godot/Jolt fixed-camera frames. Cuts at original wall time; no pose animation, speed-up, camera motion, overlays or interpolation. Nearest recorded frames resampled to 30 fps MP4 and 12 fps GIF.',
                  world_source=world_source,
                  interaction_source={p: digest(root / p) for p in ['src/sim2sim/workshop_grab.py', 'godot/hub/scene_grab.gd']},
                  duration_seconds=duration/1000, gif_bytes=gif.stat().st_size,
                  gif_sha256=digest(gif), mp4_sha256=digest(movie), cuts=evidence)
    (args.output / 'pv-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cuts'}, indent=2))

if __name__ == '__main__': main()
