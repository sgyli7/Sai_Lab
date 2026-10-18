"""CPU comparison of fixed p0_seed40, three policies, head+observer views.

Uses identical case identity rather than selecting successful rollouts. Two
outputs: synchronized simulation playback and optionally measured wall-clock.
"""
import argparse,bisect,json,math
from pathlib import Path
import av
from PIL import Image,ImageDraw,ImageOps
from .video_export import writer,emit,finish,digest

class Cursor:
 def __init__(self,path):
  self.input=av.open(str(path));self.stream=self.input.streams.video[0];self.fps=float(self.stream.average_rate);self.decoder=iter(self.input.decode(video=0));self.index=-1;self.image=None;self.ended=False
 def at(self,seconds):
  target=int(seconds*self.fps+1e-7)
  while self.index<target and not self.ended:
   try:self.image=next(self.decoder).to_image();self.index+=1
   except StopIteration:self.ended=True
  return self.image

def caption_state(result,trace,global_seconds,wall_clock):
 """Pure timeline semantics for captions; recording ends at last real capture."""
 first=trace[0]['monotonic_ns'];start=result.get('cold_process_started_monotonic_ns',first)
 end=(trace[-1]['monotonic_ns']-start)/1e9
 if wall_clock:
  ws=min(max(0.,global_seconds),end)
  j=max(0,min(len(trace)-1,bisect.bisect_right([x['monotonic_ns'] for x in trace],start+ws*1e9)-1))
  ended=global_seconds>=end
 else:
  j=min(len(trace)-1,max(0,int(global_seconds*50+1e-7)))
  ws=(trace[j]['monotonic_ns']-start)/1e9
  ended=global_seconds>=(trace[-1]['control_tick']/50.)
 outcome='SUCCESS' if result.get('success') is True else ('FAILURE' if result.get('success') is False else 'INCOMPLETE')
 qualified=result.get('qualified_stability_window') or result.get('qualified_window')
 event='No qualification yet' if not ended else 'No qualified window'
 if qualified is not None:
  qt=qualified['end_tick']
  if not 0<=qt<len(trace):raise ValueError('Qualified event tick outside actual recorded trace')
  if j>=qt:
   qw=(trace[qt]['monotonic_ns']-first)/1e9
   event=f'QUALIFIED at sim{qt/50:.2f}s / actual obs-wall{qw:.2f}s'
 return {'tick':j,'simulation_seconds':j/50.,'wall_seconds':ws,'recording_end_wall_seconds':end,
         'ended':ended,'end_label':'END OF RECORDING / LAST FRAME HELD' if ended else '',
         'outcome_label':'Final outcome: '+outcome,'qualification_label':event}

def compare(case_dirs,output_dir,wall_clock=False):
 paths=[Path(x) for x in case_dirs];output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
 if len(paths)!=3:raise ValueError('Exactly3 policy cases required')
 results=[json.loads((p/'result.json').read_text()) for p in paths];traces=[[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()] for p in paths]
 if any(r['case_id']!='p0_seed40' for r in results):raise ValueError('Fixed representative must be p0_seed40 for every policy')
 if len(set(r['policy'] for r in results))!=3:raise ValueError('Three distinct policy identities required')
 derived=[Path(json.loads((p/'config.json').read_text()).get('video_output_dir',p)) for p in paths]
 outputs={}
 for wall in ([False,True] if wall_clock else [False]):
  kind='wall_clock' if wall else 'sim_time';fps=25 if wall else 50;duration=max((t[-1]['monotonic_ns']-r.get('cold_process_started_monotonic_ns',t[0]['monotonic_ns']))/1e9 if wall else len(t)/50 for t,r in zip(traces,results));frames=math.ceil(duration*fps);dst=output/f'p0_seed40_three_policies_{kind}.mp4'
  if dst.exists():raise FileExistsError(dst)
  cursors=[[Cursor((d/f'{view}_wall_clock.mp4') if wall else (p/('head_50fps.mp4' if view=='head' else 'observer_50fps.mp4'))) for view in ['head','observer']] for p,d in zip(paths,derived)]
  out,stream=writer(dst,1920,800,fps)
  try:
   for i in range(frames):
    sec=i/fps;canvas=Image.new('RGB',(1920,800),'#101010');draw=ImageDraw.Draw(canvas)
    for col,(r,t,views) in enumerate(zip(results,traces,cursors)):
     state=caption_state(r,t,sec,wall)
     for row,(view,cursor) in enumerate(zip(['HEAD','OBSERVER'],views)):
      im=cursor.at(sec);thumb=ImageOps.contain(im,(640,340));x=col*640;y=row*400;canvas.paste(thumb,(x+(640-thumb.width)//2,y+60+(340-thumb.height)//2))
      draw.text((x+7,y+4),f"{r['policy']} | {state['outcome_label']} | {view}",fill='white')
      draw.text((x+7,y+21),f"sim={state['simulation_seconds']:.2f}s wall={state['wall_seconds']:.2f}s | {'WALL CLOCK' if wall else 'SIM50Hz (pauses compressed)'}",fill='white')
      draw.text((x+7,y+38),state['qualification_label'],fill='#aaffaa')
      if state['ended']:
       draw.rectangle((x,y+66,x+640,y+92),fill='black')
       draw.text((x+7,y+72),state['end_label'],fill='#ffcc66')
    emit(out,stream,canvas)
   finish(out,stream)
  finally:
   for views in cursors:
    for c in views:c.input.close()
  outputs[kind]={'path':str(dst),'sha256':digest(dst),'fps':fps,'frames':frames,'duration_seconds':frames/fps}
 manifest={'schema':'pi05_g1_round02_fixed_three_policy_comparison_v1','representative_case_id':'p0_seed40','selection':'Predeclared fixed case, no outcome-based selection','layout':'Columns policy order supplied; rows head then observer; short completed runs hold final image','caption_semantics':{'outcome':'Final outcome label is retrospective, not an online completion signal','ended':'Each column wall clock clamps to last actual captured timestamp; held frames explicitly labelled END OF RECORDING / LAST FRAME HELD','qualified':'Shown only once timeline reaches qualified window end; actual obs-wall is measured from first captured frame, excluding initial simulator cold start','wall_clock':'Main wall timestamp includes cold start, clamps at last capture; final cleanup excluded'},'inputs':[{'case_dir':str(p),'policy':r['policy'],'result_sha256':digest(p/'result.json')} for p,r in zip(paths,results)],'outputs':outputs}
 (output/'p0_seed40_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case-dir',action='append',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--wall-clock',action='store_true');a=p.parse_args();print(json.dumps(compare(a.case_dir,a.output_dir,a.wall_clock)))
