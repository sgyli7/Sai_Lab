"""Run one real kick policy against a selected light rigid body (no added impulse)."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import threading
import time
import numpy as np
from capture_showcase import encode, serial
from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.evaluate import record
from sim2sim.obs import build_obs

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'box':(1,'Bin12g'),'bottle':(2,'Bottle6g'),'ball':(3,'Ball10g')}

def capture(args):
    out=ROOT/'results/loose_props'/f'{time.strftime("%Y%m%d-%H%M%S")}_{args.skill}_{args.target}'
    out.mkdir(parents=True)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.nice(10)
    os.environ.update(MD_MODE='capture',MD_WIDTH='1920',MD_HEIGHT='1080',MD_RENDER_FPS='30',
        MD_WORKSHOP_COLLISIONS='1',MD_STATIC_COURSE='0',MD_PROP_TELEMETRY='1',SIM2SIM_VISUAL_STYLE='legacy')
    os.environ.pop('MD_SHOWCASE_SHOT',None)
    paths=bundle_paths(ROOT/'bundles/delivery_v3');actor=NativeAnchor(paths[args.skill])
    world=None;stop=threading.Event();pauses=[]
    deadline=threading.Timer(55,lambda:os.kill(os.getpid(),signal.SIGINT));deadline.daemon=True
    with preview_lease():
        baseline=preflight(30);baseline['session_reference']=checkpoint_session(baseline);baseline['render_fps']=30
        deadline.start()
        try:
            world=World(replace(TASKS[args.skill],robot='microduck_ball_stand_fix'),headless=args.headless,
                time_input_s=actor.time_input_s,heading_input=actor.heading_input,yaw_memory_input=actor.yaw_memory_input,
                entry_source=paths['standing'],scene_override='res://atelier/atelier.tscn')
            def pause(reason):
                pauses.append(reason);world.backend._proc.terminate()
            threading.Thread(target=watch,args=(stop,baseline,pause),daemon=True).start()
            client=world.backend._client
            index,target=TARGETS[args.target]
            client.call(dict(cmd='atelier_prop_target',index=index))
            if not args.headless:
                p=np.array([.55,.34,.56]);target_point=np.array([.11,.13,0])
                z=p-target_point;z/=np.linalg.norm(z);x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
                client.call(dict(cmd='atelier_camera_lock',camera=dict(position=p.tolist(),basis=[x.tolist(),y.tolist(),z.tolist()],fov=49.,near=.015,far=100.)))
                client.call(dict(cmd='atelier_record',directory=str(out/'frames'),fps=30))
            world.reset(61000,randomize=False)
            teacher,cmd=world.prepare_standing_entry();start=time.monotonic()
            for k in range(50):
                world.step(teacher(build_obs(world.state,world.last,cmd,world.home)[None])[0])
                if not args.headless:time.sleep(max(0,start+(k+1)*DT-time.monotonic()))
            world.finish_standing_entry()
            samples=[];rows=[];touches=[]
            for k in range(250):
                action=actor(world.obs()[None])[0];world.step(action);rows.append(record(world,action))
                sample=next(p for p in world.state.extra['raw']['workshop_props'] if p['name']==target)
                samples.append(dict(time=world.t,**sample))
                for name,body in world.features['bodies'].items():
                    for contact in body['contact_events']:
                        if contact['body']==target:touches.append(dict(time=world.t,robot_body=name,**contact))
                if not args.headless:time.sleep(max(0,start+1+(k+1)*DT-time.monotonic()))
            positions=np.array([s['position'] for s in samples]);rotations=np.array([s['quaternion_xyzw'] for s in samples])
            angles=2*np.arccos(np.clip(abs(rotations@rotations[0]),0,1))
            displacement=float(np.linalg.norm((positions[-1]-positions[0])[[0,2]]))
            report=dict(skill=args.skill,target=target,mass=samples[0]['mass'],model_sha256=actor.sha256,
                seed=61000,entry_seconds=1,skill_seconds=5,headless=args.headless,
                displacement_m=displacement,max_rotation_degrees=float(np.rad2deg(angles).max()),
                max_object_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(1-2*(rotations[:,0]**2+rotations[:,2]**2),-1,1))).max()),
                max_speed_mps=max(np.linalg.norm(s['linear_velocity']) for s in samples),
                max_angular_speed_rad_s=max(np.linalg.norm(s['angular_velocity']) for s in samples),
                robot_fell=any(r['tilt']>70 or r['z']<.045 for r in rows),robot_final_tilt=rows[-1]['tilt'],
                robot_final_height=rows[-1]['z'],contacts=touches,pauses=pauses,
                note='One policy-triggered target placement, then native Jolt; no artificial impulse, animation or mid-action pose correction. Not a general success-rate benchmark.')
            if not args.headless:
                metrics=client.call(dict(cmd='atelier_metrics'));client.call(dict(cmd='atelier_record',directory=''))
                (out/'frames.json').write_text(json.dumps(metrics['recording'],indent=2))
            # Also exercise the normal reset path, after the measured action.
            world.reset(61000,randomize=False)
            reset_props=world.state.extra['raw']['workshop_props']
            report['reset_speeds_zero']=all(np.linalg.norm(p['linear_velocity'])==0 and np.linalg.norm(p['angular_velocity'])==0 for p in reset_props)
            (out/'report.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
            (out/'props.json').write_text(json.dumps(samples,default=serial,indent=2)+'\n')
            np.savez_compressed(out/'trajectory.npz',**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
        finally:
            deadline.cancel();stop.set()
            if world:
                proc=world.backend._proc;world.close();(out/'godot.log').write_text(Path(proc._sim2sim_log_path).read_text())
        if pauses:raise RuntimeError(str(pauses))
        if not args.headless:encode(out,metrics['recording'])
    print(json.dumps({k:v for k,v in report.items() if k!='contacts'},default=serial))
    print('Contact records:',len(touches),'output:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=TARGETS,required=True)
    parser.add_argument('--skill',choices=['kick_left','kick_right'],default='kick_left')
    parser.add_argument('--headless',action='store_true')
    capture(parser.parse_args())
