"""Encode native timestamped Godot frames; trim/speed up four documented shots."""
from fractions import Fraction
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import av
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--media', type=Path, default=Path('docs/media'))
    args = p.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.nice(19)
    data = json.loads((args.capture / 'hub.json').read_text())
    task = json.loads((args.capture / 'task.json').read_text())
    assert task['success'], 'Publish only a completed, passing cargo run'
    frames = data['frames']
    assert all(a['milliseconds'] < b['milliseconds'] for a,b in zip(frames,frames[1:]))
    camera_audit=[]
    for shot in dict.fromkeys(f.get('shot','interactive') for f in frames):
        selected=[f for f in frames if f.get('shot','interactive')==shot]
        poses={json.dumps([f['camera_position'],f.get('camera_basis'),f['fov']]) for f in selected}
        assert shot!='interactive' and len(poses)==1, 'PV requires a fixed camera within each shot'
        camera_audit.append(dict(shot=shot,frames=len(selected),unique_camera_transforms=len(poses),
                                 position=selected[0]['camera_position'],basis=selected[0]['camera_basis'],fov=selected[0]['fov']))
    raw = args.capture / 'native-wall-time.mp4'
    if not raw.exists():
        with av.open(str(raw), 'w') as container:
            stream = container.add_stream('libx264', rate=30, options={'crf':'18','preset':'fast'})
            stream.width=1920; stream.height=1080; stream.pix_fmt='yuv420p'; stream.thread_count=2
            stream.time_base = Fraction(1,1000000)
            for info in frames:
                with Image.open(info['file']) as image:
                    frame=av.VideoFrame.from_image(image.convert('RGB'))
                frame.pts=round(info['milliseconds']*1000); frame.time_base=stream.time_base
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    with av.open(str(raw)) as container:
        raw_times=[float(frame.pts*frame.time_base) for frame in container.decode(video=0)]
    assert len(raw_times)==len(frames)
    timestamp_error=max(abs(t-info['milliseconds']/1000) for t,info in zip(raw_times,frames))
    assert timestamp_error<=1/30
    args.media.mkdir(parents=True, exist_ok=True)
    cuts = [(5.90,13.82,3.5,'01   抓取零件 · 100 g'),
            (13.82,34.06,4.0,'02   SO101 · 放入蓝色货仓'),
            (34.06,49.20,2.5,'03   收回机械臂 · 货仓夹紧'),
            (49.50,69.34,5.0,'04   夹紧运输 · 越过 18 mm 障碍')]
    recipe=[];filters=[];command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin']
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    for i,(start,end,duration,caption) in enumerate(cuts):
        first=min(frames,key=lambda x:abs(x['sim_seconds']-start))
        last=min(frames,key=lambda x:abs(x['sim_seconds']-end))
        assert len({f['shot'] for f in frames if first['sim_seconds']<=f['sim_seconds']<=last['sim_seconds']})==1
        a=first['milliseconds']/1000; b=last['milliseconds']/1000;speed=(b-a)/duration
        recipe.append(dict(sim_start=first['sim_seconds'],sim_end=last['sim_seconds'],
                           source_start=a,source_end=b,output_seconds=duration,source_speed=speed,caption=caption,shot=first['shot']))
        caption_file=args.capture/f'caption-{i}.txt'
        caption_file.write_text(caption)
        speed_file=args.capture/f'speed-{i}.txt'
        speed_file.write_text(f'Godot 实录 · 固定机位  /  原片 ×{speed:.1f}')
        overlay=args.capture/f'caption-{i}.png'
        plate=Image.new('RGBA',(1920,1080),(0,0,0,0));draw=ImageDraw.Draw(plate)
        draw.rounded_rectangle((40,35,600,118),radius=8,fill=(238,233,217,238),outline=(86,83,91,230),width=2)
        draw.text((60,49),'SAI 001 / 小小维修站',font=ImageFont.truetype(str(font),36,index=2),fill='#36343a')
        draw.rounded_rectangle((40,950,1150,1050),radius=8,fill=(238,233,217,238),outline=(86,83,91,230),width=2)
        draw.text((60,954),caption,font=ImageFont.truetype(str(font),34,index=2),fill='#36343a')
        draw.text((60,1003),speed_file.read_text(),font=ImageFont.truetype(str(font),23,index=2),fill='#625966')
        plate.save(overlay)
        command+=['-threads','1','-i',str(raw),'-loop','1','-framerate','30','-i',str(overlay)]
        filters.append(f'[{i*2}:v]trim=start={a:.6f}:end={b:.6f},setpts=(PTS-STARTPTS)/{speed:.9f},'
                       f'fps=30,scale=1920:1080,setsar=1[shot{i}];[shot{i}][{i*2+1}:v]overlay=shortest=1:format=auto[v{i}]')
    filters.append('[v0][v1][v2][v3]concat=n=4:v=1:a=0[out]')
    mp4=args.media/'sai-workshop-15s.mp4'
    command+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]',
              '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an','-t','15','-y',str(mp4)]
    subprocess.run(command,check=True)
    gif=args.media/'sai-workshop-15s.gif'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1','-i',str(mp4),
        '-filter_complex_threads','1','-filter_complex',
        'fps=12,scale=640:360:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
        '-loop','0','-y',str(gif)],check=True)
    duration=0;unique=set()
    with Image.open(gif) as image:
        count=image.n_frames
        for i in range(count):
            image.seek(i);duration+=image.info['duration'];unique.add(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
    assert 14800<=duration<=15200 and len(unique)>count*.90 and gif.stat().st_size<10_000_000
    report=dict(capture=args.capture.name,
        physics=f"Godot/Jolt {task.get('physics_hz', 'unknown')} Hz; policy motor targets {task.get('controller_hz', 50)} Hz",
        revision='v2-fixed-camera',camera_audit=camera_audit,
        raw_timestamp_quantization_s=1/30,raw_timestamp_max_error_s=timestamp_error,
        cargo_success=True,frame_count=len(frames),source_wall_seconds=frames[-1]['milliseconds']/1000,
        native_sim_seconds=data['seconds'],cuts=recipe,seconds=duration/1000,frames=count,unique_frames=len(unique),
        gif_bytes=gif.stat().st_size,gif_sha256=sha(gif),mp4_sha256=sha(mp4),raw_mp4_sha256=sha(raw),
        hub_sha256=sha(args.capture/'hub.json'),task_sha256=sha(args.capture/'task.json'),
        trace_sha256=sha(args.capture/'sai-trace.jsonl'),
        note='Raw MP4 PTS follow recorded wall timestamps. PV cuts use the explicit original-footage speed above. No pose animation or physics time-scale changes.')
    (args.media/'sai-workshop-15s.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cuts'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
