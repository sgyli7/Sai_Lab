"""Check exposed coplanar surfaces and actual Jolt contacts in the workshop."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from sim2sim.godot_proc import _headless_overlay, godot_bin
from showcase_resource_guard import preview_lease, preflight

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'results/workshop_validation'
    out.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    with preview_lease():
        preflight()
        overlay = _headless_overlay(ROOT / 'godot')
        try:
            env = dict(os.environ, MD_AUDIT_OUTPUT=str(out / 'surfaces.json'),
                       MD_CONTACT_REPORT=str(out / 'contacts.json'), MD_LOOSE_REPORT=str(out / 'loose_props.json'), MD_YARD_REPORT=str(out / 'yard.json'), MD_WORKSHOP_COLLISIONS='1', MD_STATIC_COURSE='1')
            for name in ('workshop_surface_probe', 'workshop_contacts', 'workshop_loose_props', 'workshop_yard'):
                result = subprocess.run([godot_bin(), '--headless', '--fixed-fps', '200',
                    '--single-threaded-scene', '--path', str(overlay), '--script',
                    f'res://tests/{name}.gd'], env=env, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=30)
                (out / f'{name}.log').write_text(result.stdout)
                print(result.stdout, end='')
                if result.returncode or any(x in result.stdout for x in ('SCRIPT ERROR:', 'ERROR:', 'WARNING:')):
                    raise RuntimeError(f'{name} failed or emitted an engine warning')
            subprocess.run([sys.executable, str(ROOT / 'scripts/check_workshop_surfaces.py'),
                            str(out / 'surfaces.json')], check=True)
        finally:
            shutil.rmtree(overlay)

if __name__ == '__main__':
    main()
