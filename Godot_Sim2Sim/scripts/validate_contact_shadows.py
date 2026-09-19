"""Check real GPU contact shadows and stationary feet in main + workshop.

Run each renderer separately with the isolated showcase environment. No physics
or caster transform is changed between the current and historical-bias images.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import numpy as np
from PIL import Image
from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor

ROOT = Path(__file__).resolve().parents[1]


def camera(client, position, target, fov):
    p, t = np.asarray(position, float), np.asarray(target, float)
    z = p - t
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    client.call(dict(cmd='atelier_camera_lock', camera=dict(position=p.tolist(),
                basis=[x.tolist(), y.tolist(), z.tolist()], fov=fov)))


def run(args):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(15)
    out = ROOT / 'results/contact_shadows' / args.renderer / args.scene
    out.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / 'godot/tests' / f'contact_shadow_runtime_{os.getpid()}'
    runtime.mkdir()
    source = 'godot/main.tscn' if args.scene == 'main' else 'godot/atelier/atelier.tscn'
    base = 'res://physics_server.gd' if args.scene == 'main' else 'res://atelier/atelier_server.gd'
    probe = (ROOT / 'godot/tests/contact_shadow_probe.gd').read_text()
    (runtime / 'server.gd').write_text(probe.replace('res://atelier/atelier_server.gd', base))
    scene = (ROOT / source).read_text().replace(base, f'res://tests/{runtime.name}/server.gd')
    (runtime / 'scene.tscn').write_text(scene)
    os.environ.update(MD_MODE='capture', MD_WIDTH='1280', MD_HEIGHT='720', MD_RENDER_FPS='30',
                      GODOT_RENDERING_DRIVER=args.renderer, MD_NEIGHBOURHOOD='1',
                      SIM2SIM_VISUAL_STYLE='microduck' if args.scene == 'main' else 'legacy')
    actor = NativeAnchor(bundle_paths(ROOT / 'bundles/delivery_v3')['standing'])
    world = None
    stop = threading.Event()
    pauses = []
    timer = threading.Timer(55, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.daemon = True
    try:
        with preview_lease():
            baseline = preflight()
            baseline['session_reference'] = checkpoint_session(baseline)
            baseline['render_fps'] = 30
            timer.start()
            try:
                world = World(replace(TASKS['standing'], robot='microduck_ball_stand_fix'), headless=False,
                              scene_override=f'res://tests/{runtime.name}/scene.tscn')
                def pause(reason):
                    pauses.append(reason)
                    if world.backend._proc.poll() is None:
                        world.backend._proc.terminate()
                threading.Thread(target=watch, args=(stop, baseline, pause), daemon=True).start()
                world.reset(61000, randomize=False)
                for _ in range(60):
                    world.step(actor(world.obs()[None])[0])
                client = world.backend._client
                camera(client, [.65, .36, -.7], [.10, .09, 0], 48)
                client.call(dict(cmd="shadow_probe", fixture=True))
                time.sleep(.1)
                client.call(dict(cmd="screenshot", path=str(out / "settled.png")))
                report = {}
                for name, params in [('current', {}), ('historical', {'shadow_bias': .1}),
                                     ('unshadowed', {'shadow_enabled': False})]:
                    report[name] = client.call(dict(cmd='shadow_probe', fixture=True, **params))
                    time.sleep(.1)
                    assert client.call(dict(cmd='screenshot', path=str(out / f'{name}.png')))['ok']
                report['resource'] = baseline
                report['pauses'] = pauses
            finally:
                stop.set()
                timer.cancel()
                if world:
                    proc = world.backend._proc
                    world.close()
                    (out / 'godot.log').write_text(Path(proc._sim2sim_log_path).read_text())
            if pauses:
                raise RuntimeError(pauses)
    finally:
        shutil.rmtree(runtime)
    # Compare a point 5 mm into the small sphere's expected ground shadow to
    # nearby lit ground, correcting for backend-specific unshadowed exposure.
    reference = np.asarray(Image.open(out / 'unshadowed.png').convert('RGB'), dtype=float)
    for name in ['current', 'historical']:
        pixels = np.asarray(Image.open(out / f'{name}.png').convert('RGB'), dtype=float)
        samples = {s['mm']: np.rint(s['pixel']).astype(int) for s in report[name]['samples']}
        def delta(mm):
            x, y = samples[mm]
            return float((reference[y, x] - pixels[y, x]).mean())
        report[name]['contact_contrast'] = delta(5) - delta(70)
    feet = [f for f in report['current']['feet'] if '/ankle_' in f['path']]
    report['sole_min_y'] = {side: min(f['min_y'] for f in feet if f'ankle_{side}/' in f['path'])
                            for side in ['left', 'right']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['current']['feet'] == report['historical']['feet'], 'Comparison moved a visual mesh'
    assert all(abs(y) < .001 for y in report['sole_min_y'].values()), 'Sole is not on the physical floor'
    assert report['current']['contact_contrast'] > 15, 'Grounded small sphere has no contact shadow'
    if args.renderer == 'vulkan':
        assert report['historical']['contact_contrast'] < 5, 'Historical bug did not reproduce'
    print(json.dumps(dict(scene=args.scene, renderer=args.renderer,
                          current=report['current']['contact_contrast'],
                          historical=report['historical']['contact_contrast'],
                          sole_min_y=report['sole_min_y'], output=str(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', choices=['opengl3', 'vulkan'], required=True)
    parser.add_argument('--scene', choices=['main', 'workshop'], required=True)
    run(parser.parse_args())
