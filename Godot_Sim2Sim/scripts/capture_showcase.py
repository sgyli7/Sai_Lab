"""Record the verified nine-model bank in the workshop, using native task contracts."""
from __future__ import annotations
import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import numpy as np
from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch, PreviewDeferred
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.models import NativeAnchor
from sim2sim.research.world import World
from sim2sim.research.evaluate import record, summarize, PROTOCOL_VERSION
from sim2sim.research.roller_tasks import summarize as summarize_roller

ROOT=Path(__file__).resolve().parents[1]
SHOTS={'standing':'wide','walking':'hero','sitstand':'hero','ground_pick':'detail',
       'kick_left':'action','kick_right':'action','roulade':'action','roller':'wide','roller_crouch':'action'}

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)

def encode(out, frames):
    import av
    from PIL import Image
    if len(frames)<2:raise ValueError('Insufficient recorded frames')
    with av.open(str(out/'raw.mp4'),'w') as container:
        stream=container.add_stream('libx264',rate=30)
        stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.thread_count=2
        stream.options={'crf':'18','preset':'fast'}
        start=frames[0]['milliseconds']
        for sample in frames:
            with Image.open(sample['file']) as im:frame=av.VideoFrame.from_image(im.convert('RGB'))
            frame.pts=round(sample['milliseconds']-start);frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)

def capture(args):
    out=ROOT/'results/showcase'/args.tag
    if out.exists():raise ValueError(f'Refusing to overwrite {out}')
    paths=bundle_paths(args.bundle)
    state=preflight(30);state['session_reference']=checkpoint_session(state);state['render_fps']=30
    out.mkdir(parents=True);(out/'resource_before.json').write_text(json.dumps(state,indent=2))
    os.environ.update(MD_MODE='capture',MD_RENDER_FPS='30',MD_WIDTH='1920',MD_HEIGHT='1080',
                      MD_SHOWCASE_SHOT=args.shot or SHOTS[args.skill],SIM2SIM_VISUAL_STYLE='legacy',MD_WORKSHOP_COLLISIONS='0')
    actor=NativeAnchor(paths[args.skill])
    task=TASKS[args.skill]
    if task.robot!='microduck_roller':task=replace(task,robot='microduck_ball_stand_fix')
    stop=threading.Event();pauses=[];world=None;timer=None
    condition='push_coast_brake' if args.skill=='roller' else 'game_seq' if args.skill=='walking' else 'default'
    seconds=args.seconds if args.seconds else task.seconds
    if not 0<seconds<=24:raise ValueError('Captures are limited to 24 simulated seconds')
    # Bound startup as well as rollout. A failed constructor cannot leave a
    # preview running indefinitely; Godot also receives parent-death SIGTERM.
    deadline=threading.Timer(55,lambda:os.kill(os.getpid(),signal.SIGINT))
    deadline.daemon=True;deadline.start()
    try:
        world=World(task,headless=False,time_input_s=actor.time_input_s,heading_input=actor.heading_input,
                    yaw_memory_input=actor.yaw_memory_input,roller_contract=args.skill=='roller',
                    entry_source=paths['roller' if task.robot=='microduck_roller' else 'standing'],
                    scene_override='res://atelier/atelier.tscn')
        proc=world.backend._proc
        def pause(reason):
            pauses.append(reason)
            if proc.poll() is None:proc.terminate()
        timer=threading.Timer(50,pause,args=['showcase preview reached 50-second limit']);timer.daemon=True;timer.start()
        threading.Thread(target=watch,args=(stop,state,pause),daemon=True).start()
        # Configure recording before reset: presentation RPCs cannot insert an
        # unreported physics tick into the captured rollout or the policy handoff.
        world.backend._client.call({'cmd':'atelier_record','directory':str(out/'frames'),'fps':30,'format':'jpg'})
        world.reset(args.seed,condition,randomize=False)
        start=time.monotonic();initial_heading=world.heading.copy();rows=[]
        entry_seconds=1.0 if args.entry=='standing' else 0.0
        if entry_seconds:
            teacher,cmd=world.prepare_standing_entry()
            from sim2sim.obs import build_obs
            for k in range(round(entry_seconds/DT)):
                observation=build_obs(world.state,world.last,cmd,world.home)
                world.step(teacher(observation[None])[0])
                time.sleep(max(0,start+(k+1)*DT-time.monotonic()))
            world.finish_standing_entry();initial_heading=world.heading.copy()
        for k in range(round(seconds/DT)):
            action=actor(world.obs()[None])[0]
            world.step(action);row=record(world,action);row['condition']=condition;rows.append(row)
            time.sleep(max(0,start+entry_seconds+(k+1)*DT-time.monotonic()))
        wall=time.monotonic()-start
        metrics=world.backend._client.call({'cmd':'atelier_metrics'})
        world.backend._client.call({'cmd':'atelier_record','directory':''})
        # The last metrics RPC is outside the rollout. Exclude any later images.
        frames=[f for f in metrics['recording'] if f['sim_seconds']<=entry_seconds+seconds+1e-5]
        result=(summarize_roller(rows,initial_heading,world.roller_target_yaw,0.) if args.skill=='roller'
                else summarize(task,rows,initial_heading))
        report={'skill':args.skill,'seed':args.seed,'condition':condition,'randomize':False,'entry':args.entry,
                'entry_seconds':entry_seconds,'skill_seconds':seconds,'sim_seconds':world.t,'wall_seconds':wall,
                'model_sha256':actor.sha256,'model_path':paths[args.skill],
                'time_input_s':actor.time_input_s,'heading_input':actor.heading_input,'yaw_memory_input':actor.yaw_memory_input,
                'physics':world.physics,'protocol':PROTOCOL_VERSION,'outcome':result,'frames':frames,
                'recording_dropped':metrics['recording_dropped'],'rendering_method':metrics['rendering_method'],
                'visual_profile':metrics['visual_profile'],'shot':os.environ['MD_SHOWCASE_SHOT'],
                'paused':pauses,'resets_after_initial_reset':0,
                'note':'Demonstration start state, not an unseen statistical test; full skill duration retained. Native ONNX control, no pose animation or recovery teleport.'}
        (out/'capture.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
        np.savez_compressed(out/'trajectory.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0] if k!='condition'})
        (out/'source.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in ['godot/atelier','src/sim2sim/research','src/sim2sim/backends']
            for p in (ROOT/base).glob('*') if p.is_file()},indent=2)+'\n')
        print(args.skill,json.dumps(result),flush=True)
    except Exception as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'pauses':pauses},indent=2)+'\n')
        raise
    finally:
        deadline.cancel()
        stop.set()
        if timer:timer.cancel()
        if world:
            proc=world.backend._proc
            world.close()
            log=getattr(proc,'_sim2sim_log_path',None)
            if log and Path(log).exists():shutil.copyfile(log,out/'godot.log')
    if pauses:raise PreviewDeferred(str(pauses))
    if (out/'godot.log').exists() and any(line.startswith(('ERROR:', 'SCRIPT ERROR:')) for line in (out/'godot.log').read_text().splitlines()):
        raise RuntimeError('Godot reported errors; inspect godot.log before promoting this capture')
    encode(out,frames)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=ROOT/'bundles/delivery_v3')
    parser.add_argument('--skill',choices=TASKS,required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--shot',choices=['hero','wide','detail','action'])
    parser.add_argument('--entry',choices=['reset','standing'],default='standing')
    parser.add_argument('--seed',type=int,default=61000);parser.add_argument('--seconds',type=float)
    args=parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():parser.error('Use a simple result tag')
    with preview_lease():capture(args)

if __name__=='__main__':main()
