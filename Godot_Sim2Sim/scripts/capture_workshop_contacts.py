"""Bounded, real ONNX walking probe on the workshop's small step lane.

A single explicit test start at the lane entrance; no mid-run pose correction.
This observes contact/falls, and does not certify uneven-terrain locomotion.
"""
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from dataclasses import replace
import mujoco
import numpy as np
from capture_showcase import encode, serial
from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.world import World
from sim2sim.research.evaluate import record
from sim2sim.obs import build_obs

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'results/workshop_contacts' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    os.environ.update(MD_MODE='capture', MD_WIDTH='1920', MD_HEIGHT='1080',
                      MD_RENDER_FPS='30', MD_WORKSHOP_COLLISIONS='1', MD_STATIC_COURSE='1', SIM2SIM_VISUAL_STYLE='legacy')
    os.environ.pop('MD_SHOWCASE_SHOT', None)
    paths = bundle_paths(ROOT / 'bundles/delivery_v3')
    actor = NativeAnchor(paths['walking'])
    world = None
    stop = threading.Event()
    pauses = []
    deadline = threading.Timer(55, lambda: os.kill(os.getpid(), signal.SIGINT))
    deadline.daemon = True
    with preview_lease():
        baseline = preflight(30)
        baseline['session_reference'] = checkpoint_session(baseline)
        baseline['render_fps'] = 30
        deadline.start()
        try:
            world = World(replace(TASKS['walking'], robot='microduck_ball_stand_fix'), headless=False,
                          entry_source=paths['standing'], yaw_memory_input=actor.yaw_memory_input,
                          scene_override='res://atelier/atelier.tscn')
            def pause(reason):
                pauses.append(reason)
                world.backend._proc.terminate()
            threading.Thread(target=watch, args=(stop, baseline, pause), daemon=True).start()
            client = world.backend._client
            target = np.array([.02, .12, 1.48]); p = np.array([.83,.50,1.99])
            z = p-target; z /= np.linalg.norm(z)
            x = np.cross([0,1,0],z); x /= np.linalg.norm(x); y = np.cross(z,x)
            client.call(dict(cmd='atelier_camera_lock', camera=dict(position=p.tolist(),
                basis=[x.tolist(),y.tolist(),z.tolist()], fov=48., near=.015, far=100.)))
            client.call(dict(cmd='atelier_record', directory=str(out/'frames'), fps=30))
            world.reset(61000, randomize=False)
            # Set the initial approach pose in the test harness only.
            world.sampler.sample(world.rng, yaw_range=(-math.pi/2,-math.pi/2), joint_noise_rad=0.)
            d,m = world.mj.data,world.mj.model
            d.qpos[world.mj.free_qposadr:world.mj.free_qposadr+2] = [.02,-1.04]
            mujoco.mj_forward(m,d)
            world.state = world.backend.reset(ctrl=world.home, bodies=world.mj.body_poses_mujoco(), report_bodies=world.report_names)
            world.features = world.measure(reset=True)
            teacher,cmd = world.prepare_standing_entry()
            start = time.monotonic()
            for k in range(50):
                world.step(teacher(build_obs(world.state,world.last,cmd,world.home)[None])[0])
                time.sleep(max(0,start+(k+1)*DT-time.monotonic()))
            world.finish_standing_entry()
            world.command_tape = np.zeros((350,13),np.float32)
            world.command_tape[:,0] = .12
            rows=[]; contacts=[]
            for k in range(350):
                action=actor(world.obs()[None])[0];world.step(action)
                rows.append(record(world,action))
                for body,b in world.features['bodies'].items():
                    for contact in b['contact_events']:
                        if contact['body'].startswith(('Step_','Ramp_','Threshold_','Stop_')):
                            contacts.append(dict(time=world.t,robot_body=body,**contact))
                time.sleep(max(0,start+1+(k+1)*DT-time.monotonic()))
            metrics=client.call(dict(cmd='atelier_metrics'))
            client.call(dict(cmd='atelier_record',directory=''))
            report=dict(seconds=8,seed=61000,speed=.12,initial_godot_xz=[.02,1.04],
                initial_yaw=-math.pi/2,model_sha256=actor.sha256,contacts=contacts,
                fell=any(r['tilt']>70 or r['z']<.045 for r in rows),final_xy=rows[-1]['xy'],
                max_tilt=max(r['tilt'] for r in rows),pauses=pauses,
                frames=metrics['recording'],note='Real policy; one lane-entry reset; not a terrain success-rate evaluation.')
            (out/'report.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
            np.savez_compressed(out/'trajectory.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        finally:
            deadline.cancel();stop.set()
            if world:
                proc=world.backend._proc;world.close()
                (out/'godot.log').write_text(Path(proc._sim2sim_log_path).read_text())
        if pauses:raise RuntimeError(str(pauses))
        encode(out,metrics['recording'])
    print(json.dumps({k:v for k,v in report.items() if k not in ('frames','contacts')},default=serial))
    print('Real obstacle contacts:',len(contacts),'output:',out)

if __name__=='__main__':
    main()
