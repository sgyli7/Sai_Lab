"""CPU video derivatives. Case result/trace/raw videos are never modified.

Default derivatives retain50Hz simulation playback with wall-time subtitles.
--wall-clock additionally resamples measured monotonic time at25fps, preserving
cold initialization and inference pauses by repeating the latest observed frame.
"""
import argparse,hashlib,json,math
from pathlib import Path
import av
from PIL import Image,ImageDraw

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def writer(path,width,height,fps):
 out=av.open(str(path),'w');stream=out.add_stream('libx264',rate=fps);stream.width=width;stream.height=height;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast'}
 return out,stream

def emit(out,stream,im):
 for packet in stream.encode(av.VideoFrame.from_image(im)):out.mux(packet)

def finish(out,stream):
 for packet in stream.encode():out.mux(packet)
 out.close()

def subtitle(im,text):
 im=im.copy();d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,52),fill='black')
 for n,line in enumerate(text):d.text((7,5+n*17),line,fill='white')
 return im

def metadata(path,fps,count,annotated,kind):
 with av.open(str(path)) as probe:
  stream=probe.streams.video[0]
  if stream.frames != count or abs(float(stream.average_rate)-fps)>1e-6:
   raise ValueError('Incomplete/mismatched video frames or rate: '+str(path))
 return {'path':path.name,'fps':fps,'frames':count,'duration_seconds':count/fps,'wall_annotation':annotated,'playback_kind':kind,'sha256':digest(path)}

def export(directory,wall_clock=False,wall_fps=25,output_dir=None):
 p=Path(directory)
 if 'round02' not in p.resolve().parts:raise ValueError('Only round02 evidence may be exported')
 derivatives = Path(output_dir) if output_dir else p
 derivatives.mkdir(parents=True,exist_ok=True)
 result_sha=digest(p/'result.json');result=json.loads((p/'result.json').read_text());rows=[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()]
 for i,row in enumerate(rows):
  if row['control_tick']!=i or (i and row['monotonic_ns']<=rows[i-1]['monotonic_ns']):raise ValueError('Invalid monotonic frame timeline')
 manifest_path=derivatives/'video_manifest.json';videos=json.loads(manifest_path.read_text())['videos'] if manifest_path.exists() else {}
 first=rows[0]['monotonic_ns'];cold=result.get('cold_process_started_monotonic_ns',first);start=min(first,cold);end=rows[-1]['monotonic_ns'];events=result.get('inference_calls',[])
 for name,rawfile in [('head','head_50fps.mp4'),('observer','observer_50fps.mp4')]:
  raw=p/rawfile;dst=derivatives/(name+'_wall_annotated.mp4')
  if not dst.exists():
   inp=av.open(str(raw));vs=inp.streams.video[0];out,stream=writer(dst,vs.width,vs.height,50);count=0
   try:
    for i,frame in enumerate(inp.decode(video=0)):
     row=rows[i];gap=(row['monotonic_ns']-rows[max(i-1,0)]['monotonic_ns'])/1e9
     im=subtitle(frame.to_image(),[f"{result['policy']} {result['case_id']} Final outcome: {'SUCCESS' if result['success'] else 'FAILURE'}",f"SIM playback50fps sim={i/50:.2f}s wall+{(row['monotonic_ns']-start)/1e9:.2f}s gap={gap:.3f}s; pauses compressed"])
     emit(out,stream,im);count+=1
    finish(out,stream)
   finally:inp.close()
   if count!=len(rows):raise ValueError('Video/trace frame counts differ')
  else:count=len(rows)
  videos[name+'_raw']=metadata(raw,50,count,False,'simulation_time_raw')
  if derivatives.resolve()!=p.resolve():videos[name+'_raw']['path']=str(raw.resolve())
  videos[name+'_wall_annotated']=metadata(dst,50,count,True,'simulation_time_with_wall_subtitles')
  if wall_clock:
   wall=derivatives/(name+'_wall_clock.mp4');frames=math.ceil((end-start)/1e9*wall_fps)+1
   if not wall.exists():
    inp=av.open(str(raw));vs=inp.streams.video[0];out,stream=writer(wall,vs.width,vs.height,wall_fps);decoder=iter(inp.decode(video=0));current=next(decoder).to_image();idx=0
    try:
     for frameindex in range(frames):
      now=start+round(frameindex/wall_fps*1e9)
      while idx+1<len(rows) and rows[idx+1]['monotonic_ns']<=now:current=next(decoder).to_image();idx+=1
      phase='PHYSICS / RENDER / CONTROL';shown=current
      if now<first:phase='SIMULATOR COLD START - no earlier camera observation';shown=Image.new('RGB',current.size,'#202020')
      for n,event in enumerate(events):
       if event.get('request_started_monotonic_ns',end+1)<=now<event.get('reply_received_monotonic_ns',0):
        phase='MODEL COLD LOAD + FIRST INFERENCE PAUSED' if n==0 else 'INFERENCE PAUSED'
        if n==0:phase+=f" (reported model load {event.get('model_load_seconds')}s)"
        break
      im=subtitle(shown,[f"{result['policy']} {result['case_id']} Final outcome: {'SUCCESS' if result['success'] else 'FAILURE'} | WALL CLOCK {frameindex/wall_fps:.2f}s sim={idx/50:.2f}s",phase]);emit(out,stream,im)
     finish(out,stream)
    finally:inp.close()
   videos[name+'_wall_clock']=metadata(wall,wall_fps,frames,True,'measured_wall_clock_resampled')
 manifest={'schema':'pi05_g1_round02_video_manifest_v1','case_id':result['case_id'],'policy':result['policy'],'videos':videos,'source_case_dir':str(p.resolve()),'timing_source':'trajectory.monotonic_ns plus inference request/reply monotonic_ns; no GPU or physics rerun','raw_result_sha256':result_sha,'trace_sha256':digest(p/'trajectory.jsonl'),'timing':{'cold_start_to_first_frame_seconds':(first-start)/1e9,'observed_trajectory_wall_seconds':(end-first)/1e9,'total_preserved_wall_seconds':(end-start)/1e9,'model_load_seconds_reported':events[0].get('model_load_seconds') if events else None},'limitations':'wall-clock files repeat latest observation across pauses; initial cold period uses blank placeholder because no image exists. Final cleanup after last observed frame is excluded. SIM subtitle files compress pauses and are not real-time performance evidence.'}
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 if digest(p/'result.json')!=result_sha:raise ValueError('Immutable result changed during CPU export')
 return manifest
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--case-dir',required=True);parser.add_argument('--output-dir');parser.add_argument('--wall-clock',action='store_true');parser.add_argument('--wall-fps',type=int,default=25);args=parser.parse_args()
 if not 1<=args.wall_fps<=50:parser.error('wall-fps must be1..50')
 print(json.dumps(export(args.case_dir,args.wall_clock,args.wall_fps,args.output_dir)))
