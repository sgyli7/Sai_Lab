"""Edit real-time capture files into a chaptered PV and compact README GIFs.

Requires FFmpeg with libass, libx264, palettegen and paletteuse. No synthesized
motion, speed changes or replacement robot frames are used.
"""
from pathlib import Path
import argparse
import av
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/media'
EDIT = ROOT / 'results/showcase/edit'
SKILLS = [
 ('standing','小小维修站','SERVICE BAY 01','MicroDuck · 九项策略引擎实录'),
 ('walking','出发，巡视工位','WALK / TURN / STOP','WD05 · 前进、转向与停步'),
 ('sitstand','收起身形，再站起来','SIT / STAND','Sitstand_Godot · 坐下与起身'),
 ('ground_pick','低头，贴近地面','GROUND PICK','alpha_ground_pick · 低头拾取动作'),
 ('kick_left','左脚，精准触球','LEFT KICK','K10 · 左脚触球后保持站立'),
 ('kick_right','换一只脚','RIGHT KICK','KR06 · 右脚触球与姿态恢复'),
 ('roulade','翻过去，站回来','FORWARD ROLL','R11 · 前滚一周后自主恢复站立'),
 ('roller','轮足：推进、滑行、刹车','ROLLER / WORK IN PROGRESS','原版 roller · 刹车未达标，仍有反向滑动'),
 ('roller_crouch','压低重心，滑出工位','CROUCH / GLIDE / RISE','原版 roller_crouch · 下蹲滑行后起身'),
]

def run(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def ass(path, title, subtitle, number, duration):
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Text,Noto Sans CJK SC,48,&H00E8E5DD,&H00E8E5DD,&H00252324,&H00252324,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    def line(text, x, y, size, color='E8E5DD', extra=''):
        return 'Dialogue: 0,0:00:00.00,0:02:00.00,Text,,0,0,0,,{\\pos(%d,%d)\\fs%d\\1c&H%s&%s}%s\n' % (x,y,size,color,extra,text)
    content = line('MICRODUCK',64,22,32,'38BDE0',r'\b1\fsp5')
    content += line('SERVICE BAY 01   /   小小维修站',374,28,24)
    content += line(f'{number:02d} / 09',1750,22,32,'9E7289')
    content += line(title,64,938,54,extra=r'\fad(160,0)')
    content += line(subtitle,67,1011,28,'BABEBE')
    content += line('ONNX × GODOT / 实时策略控制',1370,1020,24,'38BDE0')
    path.write_text(header + content)


# Policy-time seconds: trim holds, retain real speed and the actual contact/recovery.
PV_CUTS = [
 ('roulade',.35,3.85), ('walking',2.55,4.1), ('walking',5.1,6.6),
 ('kick_left',-.1,1.), ('kick_right',-.1,1.),
 ('sitstand',0,.95), ('sitstand',5.9,7.2), ('ground_pick',.25,2.9),
 ('roller',1.,2.4), ('roller',7.8,9.2),
 ('roller_crouch',0,.85), ('roller_crouch',2.25,3.55), ('roller_crouch',3.7,4.6),
 ('standing',.5,1.1),
]
GIF_CUTS = [
 ('roulade',.55,3.75), ('walking',2.6,4.1),
 ('kick_left',-.08,.72), ('kick_right',-.08,.72),
 ('sitstand',0,.6), ('sitstand',5.9,6.6),
 ('ground_pick',.35,1.15), ('ground_pick',2.1,2.8),
 ('roller',1.1,1.9), ('roller',8.2,8.8),
 ('roller_crouch',.1,.8), ('roller_crouch',2.5,3.7), ('standing',.5,.9),
]
SHORT_NAMES = {'roulade':'前滚 · 恢复站立','walking':'行走 · 转向',
 'kick_left':'左脚触球','kick_right':'右脚触球','sitstand':'坐下 · 起身',
 'ground_pick':'低头 · 拾取动作','roller':'轮足 · 刹车待改进',
 'roller_crouch':'下蹲 · 滑行 · 起身','standing':'MICRODUCK / 小小维修站'}


def build(prefix):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:raise SystemExit('FFmpeg is required')
    MEDIA.mkdir(parents=True,exist_ok=True); EDIT.mkdir(parents=True,exist_ok=True)
    common=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-filter_threads','2']
    data={name:json.loads((ROOT/'results/showcase'/f'{prefix}_{name}'/'capture.json').read_text()) for name,*_ in SKILLS}
    titles={name:(title,subtitle) for name,title,_,subtitle in SKILLS}
    order=list(dict.fromkeys(c[0] for c in PV_CUTS))
    recipes={}
    def montage(label,cuts,target,brief=False):
        segments=[];recipe=[];offset=0.
        for i,(skill,begin,end) in enumerate(cuts):
            report=data[skill];frames=report['frames'];base=frames[0]['milliseconds']
            def stamp(t):
                f=min(frames,key=lambda f:abs(f['sim_seconds']-(report['entry_seconds']+t)))
                return (f['milliseconds']-base)/1000.
            start=stamp(begin);duration=stamp(end)-start
            directory=ROOT/'results/showcase'/f'{prefix}_{skill}'
            title,subtitle=titles[skill]
            if brief:title,subtitle=SHORT_NAMES[skill],''
            subtitles=EDIT/f'{label}_{i}.ass';ass(subtitles,title,subtitle,order.index(skill)+1,duration)
            output=EDIT/f'{label}_{i}.mp4'
            crop=skill in ('roulade','kick_left','kick_right','roller','roller_crouch','standing')
            framing='crop=1536:864:192:108,scale=1920:1080,' if crop else ''
            filters=(framing+'fps=30,drawbox=x=0:y=0:w=iw:h=82:color=0x24232b:t=fill,'
                     'drawbox=x=0:y=925:w=iw:h=155:color=0x24232b:t=fill,'
                     'drawbox=x=64:y=919:w=130:h=6:color=0xe0bd38:t=fill,'
                     f"ass=filename='{subtitles}':fontsdir='{ROOT / 'godot/atelier/fonts'}'")
            run(common+['-ss',f'{start:.6f}','-i',str(directory/'raw.mp4'),'-t',f'{duration:.6f}',
                        '-vf',filters,'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)])
            with av.open(str(output)) as encoded:
                encoded_duration=encoded.streams.video[0].frames/30.0
            segments.append(output)
            recipe.append({'skill':skill,'source':str((directory/'raw.mp4').relative_to(ROOT)),
                           'source_sha256':hashlib.sha256((directory/'raw.mp4').read_bytes()).hexdigest(),
                           'policy_start_seconds':begin,'policy_end_seconds':end,
                           'source_start_seconds':start,'duration_seconds':duration,
                           'output_start_seconds':offset,'encoded_duration_seconds':encoded_duration,'speed':1.0,'center_crop':crop})
            offset+=encoded_duration
        concat=EDIT/f'{label}.txt';concat.write_text(''.join(f"file '{p}'\n" for p in segments))
        run(common+['-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(target)])
        recipes[label]=recipe
        print(label,round(offset,3),'seconds',flush=True)
    montage('pv_short',PV_CUTS,MEDIA/'microduck-service-bay-pv.mp4')
    montage('preview_short',GIF_CUTS,EDIT/'preview_short.mp4',brief=True)
    gif(common,EDIT/'preview_short.mp4',MEDIA/'microduck-service-bay-15s.gif',640,12)
    for filename,names in [('postures',['sitstand','ground_pick']),('kicks-and-roll',['kick_left','kick_right','roulade']),('wheels',['roller','roller_crouch'])]:
        montage(filename+'_short',[cut for cut in PV_CUTS if cut[0] in names],EDIT/f'{filename}_short.mp4')
        gif(common,EDIT/f'{filename}_short.mp4',MEDIA/f'{filename}.gif',640,12)
    run(common+['-ss','0.5','-i',str(MEDIA/'microduck-service-bay-pv.mp4'),'-frames:v','1',str(MEDIA/'poster.jpg')])
    (MEDIA/'edit.json').write_text(json.dumps({'source_prefix':prefix,'cuts':recipes,
        'fps_mp4':30,'gif_fps_target':12,'silent':True,'revision':'action-first-under-15-seconds',
        'note':'Real captured frames at original wall-clock speed. Hard cuts remove holds; selected wide shots are cropped for action readability. Complete cycles remain in the raw evidence archive.'},ensure_ascii=False,indent=2)+'\n')


def gif(common,source,dest,width,fps):
    # Flat comic colors compress more cleanly without moving dither noise.
    hero = dest.name == 'microduck-service-bay-15s.gif'
    budget = 10_000_000
    profiles = [(640,12,96),(640,12,64),(640,10,64)] if hero else [(width,fps,96),(560,10,64),(480,10,48)]
    for size,rate,colors in profiles:
        filters=(f'fps={rate},scale={size}:-2:flags=lanczos,split[a][b];'
                 f'[a]palettegen=max_colors={colors}:stats_mode=diff[p];'
                 '[b][p]paletteuse=dither=none:diff_mode=rectangle')
        run(common+['-i',str(source),'-filter_complex_threads','2',
                    '-filter_complex',filters,'-loop','0',str(dest)])
        if dest.stat().st_size<=budget:
            break
    if dest.stat().st_size>budget:
        raise RuntimeError(f'GIF exceeds homepage size budget: {dest}')
    print(dest.name,dest.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='04')
    build(parser.parse_args().prefix)
