"""Six matched views and a short native-policy orbit; bounded guarded capture."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import numpy as np
from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.world import World
from capture_showcase import encode, serial

ROOT = Path(__file__).resolve().parents[1]
# Positions are world metres, held identical across the before/after captures.
VIEWS = {
    'overview': ([3.7, 2.55, 4.3], [0, .52, -1.0], 49),
    'follow': ([.65, .43, .91], [0, .21, -.20], 49),
    'courtyard': ([1.8, .92, 2.2], [-.25, .63, -1.35], 58),
    'workbench': ([.22, .78, -.24], [-.70, .68, -1.45], 58),
    'lane': ([2.65, .45, .48], [1.50, .66, -2.30], 59),
    'roofline': ([1.65, 1.50, .95], [-.30, 1.04, -2.20], 58),
}

def camera(client, p, target, fov):
    p, target = np.asarray(p, dtype=float), np.asarray(target, dtype=float)
    z = p-target; z /= np.linalg.norm(z)
    x = np.cross([0,1,0], z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    client.call(dict(cmd='atelier_camera_lock', camera=dict(position=p.tolist(),
        basis=[x.tolist(), y.tolist(), z.tolist()], fov=fov, near=.015, far=30.)))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    parser.add_argument('--before', action='store_true'); args=parser.parse_args()
    if not args.tag.replace('-','').replace('_','').isalnum():parser.error('Simple tag required')
    out=ROOT/'results/atmosphere'/args.tag
    if out.exists():raise ValueError('Refusing to overwrite capture')
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:])); os.nice(10)
    os.environ.update(MD_MODE='capture', MD_WIDTH='1920', MD_HEIGHT='1080', MD_RENDER_FPS='30',
        MD_YARD_DRESSING='0' if args.before else '1', MD_WORKSHOP_COLLISIONS='1', MD_STATIC_COURSE='0')
    os.environ.pop('MD_SHOWCASE_SHOT', None)
    paths=bundle_paths(ROOT/'bundles/delivery_v3'); actor=NativeAnchor(paths['standing'])
    stop=threading.Event(); pauses=[]; world=None
    timer=threading.Timer(55, lambda:os.kill(os.getpid(),signal.SIGINT));timer.daemon=True
    with preview_lease():
        baseline=preflight(30);baseline['render_fps']=30;baseline['session_reference']=checkpoint_session(baseline)
        out.mkdir(parents=True);timer.start()
        try:
            world=World(replace(TASKS['standing'], robot='microduck_ball_stand_fix'), headless=False,
                scene_override='res://atelier/atelier.tscn')
            def pause(reason):
                pauses.append(reason)
                if world.backend._proc.poll() is None:world.backend._proc.terminate()
            threading.Thread(target=watch,args=(stop,baseline,pause),daemon=True).start()
            c=world.backend._client;c.call(dict(cmd='atelier_view',view='overview'))
            world.reset(61000,randomize=False)
            start=time.monotonic()
            for k in range(50):
                world.step(actor(world.obs()[None])[0]);time.sleep(max(0,start+(k+1)*DT-time.monotonic()))
            for name,(p,t,fov) in VIEWS.items():
                camera(c,p,t,fov)
                c.call(dict(cmd='screenshot',path=str(out/(name+'.png'))))
            camera(c,*VIEWS['courtyard'])
            c.call(dict(cmd='atelier_record',directory=str(out/'frames'),fps=30,format='jpg'))
            start=time.monotonic()
            for k in range(300):
                world.step(actor(world.obs()[None])[0])
                # Slow camera arc, real scene depth and native standing motion.
                a=k/299.*.22;p=np.asarray(VIEWS['courtyard'][0])+[a, .04*np.sin(a*8), -a*.5]
                if k%5==0:camera(c,p,VIEWS['courtyard'][1],58)
                time.sleep(max(0,start+(k+1)*DT-time.monotonic()))
            metrics=c.call(dict(cmd='atelier_metrics'));c.call(dict(cmd='atelier_record',directory=''))
            report=dict(before=args.before,views=VIEWS,metrics=metrics,pauses=pauses,resource=baseline,
                model_sha256=actor.sha256,source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT/'godot/atelier').glob('*') if p.is_file()})
            (out/'report.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
        finally:
            stop.set();timer.cancel()
            if world:
                proc=world.backend._proc;world.close()
                (out/'godot.log').write_text(Path(proc._sim2sim_log_path).read_text())
        if pauses:raise RuntimeError(str(pauses))
        encode(out,metrics['recording'])
    print(out)

if __name__=='__main__':main()
