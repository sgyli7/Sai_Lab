"""Build the README montage from existing scene, contact and policy footage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/media'
CUTS = [
    ('distant-scenery-preview.mp4', 0., 5., 'walking'),
    ('loose-props.mp4', 4., 3., 'bottle_contact'),
    ('microduck-service-bay-pv.mp4', 0., 3.5, 'forward_roll'),
    ('microduck-service-bay-pv.mp4', 16.4, 3., 'roller_crouch'),
]


def main():
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.nice(19)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin']
    filters = []
    recipe = []
    for index, (name, start, duration, action) in enumerate(CUTS):
        path = MEDIA / name
        command += ['-threads', '1', '-i', str(path)]
        filters.append(f'[{index}:v]trim=start={start}:duration={duration},'
                       f'setpts=PTS-STARTPTS,fps=12,scale=640:360:flags=lanczos,setsar=1[v{index}]')
        recipe.append(dict(source=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           start=start, duration=duration, action=action, speed=1))
    filters += ['[v0][v1][v2][v3]concat=n=4:v=1:a=0,split[a][b]',
                '[a]palettegen=max_colors=96:stats_mode=diff[p]',
                '[b][p]paletteuse=dither=none:diff_mode=rectangle[out]']
    target = MEDIA / 'microduck-sim2sim.gif'
    temporary = MEDIA / 'microduck-sim2sim.tmp.gif'
    command += ['-filter_complex_threads', '1', '-filter_complex', ';'.join(filters),
                '-map', '[out]', '-an', '-loop', '0', '-y', str(temporary)]
    subprocess.run(command, check=True)
    unique = set()
    duration = 0
    with Image.open(temporary) as image:
        frames = image.n_frames
        for index in range(frames):
            image.seek(index)
            duration += image.info['duration']
            unique.add(hashlib.sha256(image.convert('RGB').tobytes()).digest())
    assert 14000 <= duration <= 15000
    assert len(unique) > frames * .9
    assert temporary.stat().st_size < 10_000_000
    temporary.replace(target)
    report = dict(cuts=recipe, seconds=duration/1000, frames=frames, unique_frames=len(unique),
                  bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (MEDIA / 'microduck-sim2sim.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cuts'}, indent=2))


if __name__ == '__main__':
    main()
