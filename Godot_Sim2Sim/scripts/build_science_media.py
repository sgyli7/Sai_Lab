"""Package unmodified Godot screenshots and encode frames at their recorded times."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import av
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gallery',type=Path)
    parser.add_argument('play',type=Path,nargs='?',help='Optional native recording to encode alongside the gallery')
    parser.add_argument('--movie-name',default='science-station-play.mp4')
    parser.add_argument('--output',type=Path,default=Path('docs/science-station/media'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    gallery=json.loads((args.gallery/'hub.json').read_text())
    selected={}
    for frame in gallery['frames']:
        if frame['shot']!='interactive':selected[frame['shot']]=frame
    required={'arrival','overview','towers','samples','berth','hills'}
    assert required <= selected.keys(), f'Missing preset views: {required-selected.keys()}'
    for name,frame in selected.items():
        shutil.copy2(frame['file'],args.output/(name+'.jpg'))
    if args.play is None:
        metadata=dict(screenshots={k:v['file'] for k,v in selected.items()},
                      source='Unmodified native Godot 1920 × 1080 viewport captures')
        (args.output/'media.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata,indent=2));return
    play=json.loads((args.play/'hub.json').read_text())
    frames=play['frames']
    assert len(frames)>30
    output=av.open(str(args.output/args.movie_name),'w')
    stream=output.add_stream('libx264',rate=30)
    stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
    stream.time_base=Fraction(1,1000);stream.codec_context.time_base=Fraction(1,1000)
    stream.options={'crf':'21','preset':'fast','threads':'2'}
    base=frames[0]['milliseconds']
    for source in frames:
        with Image.open(source['file']) as im:
            frame=av.VideoFrame.from_image(im)
        frame.pts=round(source['milliseconds']-base);frame.time_base=Fraction(1,1000)
        for packet in stream.encode(frame):output.mux(packet)
    for packet in stream.encode():output.mux(packet)
    output.close()
    metadata=dict(screenshots={k:v['file'] for k,v in selected.items()},frames=len(frames),
        wall_seconds=(frames[-1]['milliseconds']-base)/1000,
        simulation_seconds=frames[-1]['hub_seconds']-frames[0]['hub_seconds'],
        source='Native Godot viewport, original wall timestamps, no speed-up or pose animation')
    (args.output/'media.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
