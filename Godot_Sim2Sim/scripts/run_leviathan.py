"""Launch independent map 03 with the Leviathan review bundle."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading

GAME = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = GAME.parents[1] / "RobotDesign/Leviathan_001"


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    p.add_argument("--runtime-dir", type=Path, default=GAME / "results/leviathan/runtime")
    p.add_argument("--output", type=Path, default=GAME / "results/leviathan/play")
    p.add_argument("--plan", type=Path)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--robot", choices=("microduck", "roller", "sai"), default="sai")
    p.add_argument("--spawn", choices=("under", "platform", "cockpit"), default="under")
    p.add_argument("--display-driver", choices=("x11", "wayland"), help="Optional desktop backend, e.g. x11 for OS keyboard validation")
    p.add_argument("--reference-runtime", action="store_true", help="Diagnostic frozen script backends")
    p.add_argument("--time-scale", type=float, default=1., help="Initial time rate, 0.1 to 3.0; also adjustable in game")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    delivery = a.design / "artifacts/delivery/leviathan-review-2026-09-14"
    baseline = json.loads((delivery / "training/baseline.json").read_text())
    engine = Path(baseline["engine"]["path"])
    if digest(engine) != baseline["engine"]["sha256"]:
        raise RuntimeError("Frozen engine hash mismatch")
    for name, expected in baseline["model_files_sha256"].items():
        if digest(delivery / "model" / name) != expected:
            raise RuntimeError(f"Frozen asset hash mismatch: {name}")
    source = delivery / "training/godot-source"
    runtime = a.runtime_dir.resolve()
    if not runtime.is_relative_to((GAME / "results/leviathan").resolve()) or runtime == (GAME / "results/leviathan").resolve():
        raise ValueError("Map 03 runtime must be a subdirectory of results/leviathan; map 01/02 paths are protected")
    if not .1 <= a.time_scale <= 3.:
        raise ValueError("Time rate must be between 0.1 and 3.0")
    shutil.copytree(source, runtime, dirs_exist_ok=True)
    for folder in ("hub", "science_station", "tests", "atelier", "visuals", "standalone"):
        shutil.copytree(GAME / "godot" / folder, runtime / folder, dirs_exist_ok=True)
    shutil.copy2(GAME / "godot/main.tscn", runtime / "main.tscn")
    microduck_bundle = GAME / "results/workshop-hub/runtime/runtime_assets"
    if not (microduck_bundle / "deployment.json").is_file():
        raise RuntimeError("Prepare Robot_Godot_Sim2Sim once before opening scene 03")
    shutil.copytree(microduck_bundle, runtime / "runtime_assets", dirs_exist_ok=True)
    shutil.copytree(GAME / "results/workshop-hub/runtime/generated", runtime / "generated", dirs_exist_ok=True)
    microduck_native = GAME / "godot/native"
    double_binaries = [microduck_native / f"bin/libmicroduck_policy.linux.double.{kind}.arm64.so" for kind in ("debug", "release")]
    if not all(path.is_file() for path in double_binaries):
        subprocess.run([
            sys.executable, str(GAME / "native/bootstrap.py"), "--double-only",
            "--double-engine", str(engine),
            "--double-godotcpp", str(a.design / "native/.deps/godot-cpp"),
        ], check=True)
    shutil.copytree(microduck_native, runtime / "native", dirs_exist_ok=True)
    shutil.copytree(GAME / "integrations/leviathan/godot", runtime, dirs_exist_ok=True)
    # The default wheel-name suffix heuristic strips authored motion groups.
    import_path = runtime / "leviathan/leviathan.glb.import"
    if import_path.exists():
        text = import_path.read_text().replace("nodes/use_name_suffixes=true", "nodes/use_name_suffixes=false").replace("nodes/use_node_type_suffixes=true", "nodes/use_node_type_suffixes=false")
    else:
        text = '[remap]\nimporter="scene"\nimporter_version=1\ntype="PackedScene"\n[deps]\nsource_file="res://leviathan/leviathan.glb"\n[params]\nnodes/use_name_suffixes=false\nnodes/use_node_type_suffixes=false\n'
    import_path.write_text(text)
    native = runtime / "leviathan/native/bin/libleviathan_jel_double.so"
    if digest(native) != baseline["native_extension_sha256"]:
        raise RuntimeError("Frozen native extension hash mismatch")
    if digest(runtime / "leviathan/physics_runtime.gd") != baseline["godot_runtime_sha256"]:
        raise RuntimeError("Frozen vehicle runtime hash mismatch")
    for name in ("physics.json", "leviathan.glb", "visual_motion.json"):
        if digest(runtime / "leviathan" / name) != baseline["model_files_sha256"][name]:
            raise RuntimeError(f"Staged frozen asset mismatch: {name}")
    if not a.reference_runtime:
        backend=json.loads((GAME / "integrations/leviathan/backend.json").read_text())
        backend_root=GAME / "integrations/leviathan/backend"
        for name, expected in backend["sha256"].items():
            if digest(backend_root / name) != expected:
                raise RuntimeError(f"Qualified computational backend changed: {name}; revalidation required")
        # Existing qualified computational backends; mass, collisions, joints,
        # solver iterations and 0.5 ms physical timestep remain frozen.
        source_files = ("physics_runtime.gd", "jel_forces.gd", "visual_binding.gd", "imported_vehicle.gd")
        for name in source_files:
            text = (backend_root / name).read_text()
            for dependency in source_files:
                text = text.replace("res://" + dependency, "res://leviathan/" + dependency)
            text = text.replace("res://native/leviathan_jel.gdextension", "res://leviathan/native/leviathan_jel.gdextension")
            (runtime / "leviathan" / name).write_text(text)
        shutil.copy2(backend_root / "native/bin/libleviathan_jel_double.so", native)
    project = runtime / "project.godot"
    project.write_text(project.read_text().replace('run/main_scene="res://main.tscn"', 'run/main_scene="res://hub/main.tscn"'))
    (runtime / ".godot").mkdir(exist_ok=True)
    cache=runtime / ".godot/global_script_class_cache.cfg"
    if not cache.exists():cache.write_text("list=[]\n")
    profile = runtime / "leviathan/host_physics_profile.json"
    subprocess.run([str(engine), "--headless", "--path", str(runtime), "--script", "res://hub/dump_physics.gd"],
                   env=dict(os.environ, HUB_PHYSICS_DUMP=str(profile)), check=True)
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    options = dict(scene="polar_range", polar=True, drive_speed=.5, fast_check=a.headless,
                   native_batch=not a.reference_runtime, time_scale=a.time_scale, choose_scene=False, robot=a.robot, polar_spawn=a.spawn, task="drive", output=str(a.output), record=a.record,
                   leviathan=json.loads((source / "station_interface.json").read_text()),
                   plan=json.loads(a.plan.read_text()) if a.plan else {})
    (runtime / "hub/options.json").write_text(json.dumps(options, indent=2))
    # Read the qualified controller snapshot; never mutate the frozen delivery.
    sys.path.insert(0, str(delivery / "training/controller"))
    import sim2sim
    sim2sim.__path__.append(str(GAME / "src/sim2sim"))
    from sim2sim.workshop import serve, import_runtime_assets
    from sai_agent.paths import resource_root
    editor = a.design / "artifacts/engine-desktop-double-build/godot-ed1daf0bf/bin/godot.linuxbsd.editor.double.arm64"
    import_runtime_assets(str(editor), runtime)
    (a.output / "source.json").write_text(json.dumps(dict(baseline=baseline["baseline_id"],
        engine_sha256=digest(engine), mass_kg=baseline["total_mass_kg"],
        vehicle_runtime_sha256=digest(runtime / "leviathan/physics_runtime.gd"),
        native_extension_sha256=digest(native), native_batch=not a.reference_runtime,
        physics_hz=2000, solver_iterations=[32,4], time_scale=a.time_scale,
        physics_manifest_sha256=digest(runtime / "leviathan/physics.json"),
        host={str(f.relative_to(GAME)):digest(f) for folder in ("integrations/leviathan/godot/hub", "integrations/leviathan/godot/polar_range")
              for f in (GAME / folder).iterdir() if f.suffix in (".gd", ".gdshader")}), indent=2))
    if a.prepare_only:return
    os.environ.update(MD_MODE="hub", MD_WORKSHOP_COLLISIONS="1", SIM2SIM_VISUAL_STYLE="legacy")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0)); listener.listen(4); listener.settimeout(.25)
        stop = threading.Event()
        command = [str(engine), "--path", str(runtime), "res://hub/main.tscn", "--disable-vsync",
                   "--rendering-method", "forward_plus", "--rendering-driver", "vulkan"]
        if a.display_driver:command += ["--display-driver", a.display_driver]
        if a.headless:command += ["--headless", "--fixed-fps", "30"]
        command += ["--", f"--port={listener.getsockname()[1]}"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            service = pool.submit(serve, listener, stop, resource_root(), None)
            child = subprocess.Popen(command)
            try:
                while child.poll() is None:
                    stop.wait(.1)
                    if service.done():service.result()
                if child.returncode:raise RuntimeError(f"Godot exited {child.returncode}")
            finally:
                stop.set()
                if child.poll() is None:child.terminate(); child.wait(timeout=10)
                service.result(timeout=10)


if __name__ == "__main__":main()
