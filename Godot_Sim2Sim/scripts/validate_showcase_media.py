"""Fully decode published media and record dimensions, timing and checksums."""
import hashlib
import json
from pathlib import Path
import av

ROOT=Path(__file__).resolve().parents[1]
media=ROOT/'docs/media'
results={}
for path in sorted(media.iterdir()):
    if path.suffix not in ('.gif','.mp4'):continue
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        stamps=[];durations=[];unique=set()
        for i,frame in enumerate(container.decode(stream)):
            unique.add(hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).digest())
            stamps.append(float(frame.pts*frame.time_base))
            durations.append(float(frame.duration*frame.time_base) if frame.duration else 0.)
            if path.name=='microduck-service-bay-15s.gif' and i in [0,6,12]:
                frame.to_image().save(ROOT/f'results/showcase/gif-review-{i}.png')
        assert len(stamps)>20 and all(b>a for a,b in zip(stamps,stamps[1:])),path
        span=stamps[-1]-stamps[0]+durations[-1]
        results[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                            'width':stream.width,'height':stream.height,'frames':len(stamps),
                            'duration_seconds':span,'decoded_all_frames':True,
                            'monotonic_timestamps':True,'unique_frames':len(unique),'effective_fps':len(stamps)/span}
        assert len(unique) > len(stamps)*.95, f'Static or mostly repeated frames: {path}'
        assert span <= (15 if path.suffix=='.gif' else 25), f'Edit too long: {path}'
        if path.suffix=='.gif':assert path.stat().st_size <= 10_000_000,path
        print(path.name,results[path.name],flush=True)
(media/'manifest.json').write_text(json.dumps(results,indent=2)+'\n')
