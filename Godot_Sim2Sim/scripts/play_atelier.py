"""Open the workshop with the verified candidate bank and current native controls."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from showcase_resource_guard import preflight, checkpoint_session, preview_lease, watch
from sim2sim.research.bundle import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'bundles/delivery_v3')
    parser.add_argument('--roller', action='store_true')
    parser.add_argument('--flat', action='store_true', help='Historical flat demo: no workshop prop collision or contact course')
    parser.add_argument('--steps', action='store_true', help='Use the previous static course instead of loose rigid bodies')
    parser.add_argument('--tour', action='store_true', help='Scenic camera; controls remain active')
    args = parser.parse_args()
    if args.flat and args.steps:
        parser.error("--flat and --steps are mutually exclusive")
    verify(args.bundle)
    os.environ.update(MICRODUCK_POLICIES=str(args.bundle.resolve() / 'models'),
                      MD_MODE='tour' if args.tour else 'play', MD_RENDER_FPS='30',
                      MD_WIDTH='1920', MD_HEIGHT='1080', SIM2SIM_VISUAL_STYLE='legacy')
    os.environ.pop('MD_SHOWCASE_SHOT', None)
    os.environ['MD_WORKSHOP_COLLISIONS'] = '0' if args.flat else '1'
    os.environ['MD_STATIC_COURSE'] = '1' if args.steps else '0'
    command = [sys.executable, '-m', 'sim2sim.play', '--scene', 'res://atelier/atelier.tscn',
               '--robot', str(ROOT / 'robots/microduck_ball_stand_fix.json')]
    if args.roller:
        command.append('--roller')
    with preview_lease():
        state = preflight(30)
        state['session_reference'] = checkpoint_session(state)
        state['render_fps'] = 30
        stop = threading.Event()
        proc = subprocess.Popen(command, start_new_session=True)
        def pause(reason):
            print('Workshop paused:', reason, flush=True)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        threading.Thread(target=watch, args=(stop, state, pause), daemon=True).start()
        try:
            return proc.wait()
        except KeyboardInterrupt:
            return 130
        finally:
            stop.set()
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()  # The Godot child has a parent-death SIGTERM handler.
                    proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
