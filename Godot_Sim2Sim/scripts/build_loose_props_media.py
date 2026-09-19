"""Build a 12-second, actual-speed comparison from the latest three valid captures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import av
from showcase_resource_guard import preflight

ROOT=Path(__file__).resolve().parents[1]
def main():
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.nice(10);preflight()
    runs=[]
    for skill,target in [('kick_left','Bin12g'),('kick_right','Bottle6g'),('kick_left','Ball10g')]:
        candidates=[]
        for p in (ROOT/'results/loose_props').glob('*/report.json'):
            data=json.loads(p.read_text())
            if data['skill']==skill and data['target']==target and not data['headless'] and not data['pauses'] and (p.parent/'raw.mp4').is_file():
                candidates.append(p.parent)
        if not candidates:raise ValueError(f'Missing visible capture: {skill}/{target}')
        runs.append(max(candidates))
    filters=[];edits=[];command=['ffmpeg','-hide_banner','-loglevel','error','-y','-filter_complex_threads','1']
    for i,p in enumerate(runs):
        frames=json.loads((p/'frames.json').read_text())
        entry=min(frames,key=lambda f:abs(f['sim_seconds']-.9))
        start=(entry['milliseconds']-frames[0]['milliseconds'])/1000
        command+=['-threads','1','-i',str(p/'raw.mp4')]
        filters.append(f'[{i}:v]trim=start={start:.6f}:duration=4,setpts=PTS-STARTPTS,fps=30[v{i}]')
        edits.append(dict(source=str(p.relative_to(ROOT)/'raw.mp4'),sha256=hashlib.sha256((p/'raw.mp4').read_bytes()).hexdigest(),start=start,duration=4,speed=1))
    filters.append('[v0][v1][v2]concat=n=3:v=1:a=0[out]')
    output=ROOT/'docs/media/loose-props.mp4'
    command+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True)
    frames=json.loads((runs[1]/'frames.json').read_text())
    shot=min(frames,key=lambda f:abs(f['sim_seconds']-1.36))
    shutil.copyfile(shot['file'],ROOT/'docs/media/loose-props.jpg')
    with av.open(str(output)) as video:
        stream=video.streams.video[0];count=0;previous=-1.;unique=set()
        for frame in video.decode(video=0):
            stamp=float(frame.pts*frame.time_base)
            if stamp<=previous:raise ValueError('Non-increasing frame timestamps')
            previous=stamp;count+=1;unique.add(hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest())
        duration=float(stream.duration*stream.time_base)
        if not (11.9<=duration<=12.1 and count>=350 and len(unique)>100):raise ValueError('Invalid duration or insufficient motion frames')
        record=dict(seconds=duration,width=stream.width,height=stream.height,frames=count,unique_frames=len(unique),speed=1,edits=edits,sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (ROOT/'docs/media/loose-props-edit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
