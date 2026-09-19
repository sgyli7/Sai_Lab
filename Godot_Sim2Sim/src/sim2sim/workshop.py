"""One persistent workshop window: native MicroDuck and release Sai controllers."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
from sim2sim.paths import sim2sim_root
from sim2sim.sai_driving import DEFAULT_DRIVE_SPEED, drive_speed
from sim2sim.sai_timebase import DEFAULT_PHYSICS_HZ, SUPPORTED_PHYSICS_HZ, patch_generated_runtime

ROOT = sim2sim_root()


def import_runtime_assets(godot: str, runtime: Path) -> None:
    command = [godot, "--headless", "--editor", "--path", str(runtime), "--import", "--quit"]
    for attempt in range(2):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if result.returncode != -signal.SIGABRT or attempt == 1:
            result.check_returncode()


def prepare(runtime: Path, godot: str) -> Path:
    from sai_agent.paths import resource_root
    from sai_agent.cli import prepare_godot
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "godot", runtime, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".godot", "vendor", "*.import", "scenery", "textures", "materials",
                                                   "grass", "levels", "objects", "forest_generated", "addons"))
    # Bound engine worker allocation independently of host logical CPU count.
    # Scheduling only: no time scale, solver, policy or motor parameter changes.
    project = runtime / "project.godot"
    text = project.read_text().replace('config/name="Microduck Sim2Sim"',
        'config/name="Robot Godot Workshop"\nconfig/icon="res://atelier/icon.svg"', 1)
    project.write_text(text + "\n[threading]\nworker_pool/max_threads=2\n")
    bundle = resource_root()
    core = runtime.parent / "sai-release"
    prepare_godot(bundle, core)
    patch_generated_runtime(core)
    (runtime / "sai_release").mkdir(exist_ok=True)
    # Release code assumes an origin-centred scene in this diagnostic only.
    source = (core / "main.gd").read_text()
    old = "base.global_transform.affine_inverse()*robot.item.position"
    assert source.count(old) == 1
    (runtime / "sai_release/main.gd").write_text(source.replace(old,
        "base.global_transform.affine_inverse()*robot.item.global_position"))
    for name in ("robot.gd", "item_observation.gd"):
        shutil.copy2(core / name, runtime / name)
    shutil.copytree(core / "sai_agent", runtime / "sai_agent", dirs_exist_ok=True)
    # Frozen in-process Sai locomotion bundle.  Runtime reads these resources
    # directly from Godot; the upstream package remains only a preparation-time
    # source for robot meshes/specification and the explicit Python oracle.
    shutil.copytree(ROOT / "src/sim2sim/assets/sai/upstream", runtime / "sai_policy", dirs_exist_ok=True)
    shutil.copy2(ROOT / "src/sim2sim/assets/sai/flat-motion-v1.onnx", runtime / "sai_policy/flat-motion-v1.onnx")
    shutil.copy2(ROOT / "src/sim2sim/assets/sai/flat-motion-v1.json", runtime / "sai_policy/flat-motion-v1.json")
    shutil.copy2(ROOT / "src/sim2sim/assets/sai/suspension-v2.json", runtime / "sai_policy/suspension-v2.json")
    shutil.copy2(bundle / "models/tasks/pick-place.json", runtime / "sai_policy/pick-place.json")
    shutil.copy2(bundle / "models/tasks/so101-axes.json", runtime / "sai_policy/so101-axes.json")
    shutil.copy2(bundle / "policies/legacy-crawl57.json", runtime / "sai_policy/legacy-crawl57.json")
    # Capture engine-resolved defaults, not guessed Jolt equivalents.
    profiles = {}
    for name, project in (("microduck", runtime), ("sai", core)):
        shutil.copy2(ROOT / "godot/hub/dump_physics.gd", project / "dump_physics.gd")
        output = runtime.parent / f"{name}-physics.json"
        subprocess.run([godot, "--headless", "--path", str(project), "--script", "res://dump_physics.gd"],
                       env=dict(os.environ, HUB_PHYSICS_DUMP=str(output)), check=True)
        profiles[name] = json.loads(output.read_text())
    (runtime / "hub/physics_profiles.json").write_text(json.dumps(profiles, indent=2))
    if not (runtime / "runtime_assets/deployment.json").is_file():
        raise SystemExit("Prepare MicroDuck's native model bundle first; see docs/workshop-hub.md.")
    from sim2sim.default_sprint import apply_default_sprint
    apply_default_sprint(runtime)
    import_runtime_assets(godot, runtime)
    return bundle


def serve(listener, stop, bundle, trace):
    from sim2sim.workshop_grab import WorkshopController
    from sai_agent.cargo_godot import CargoGodotController
    while not stop.is_set():
        try:
            client, _ = listener.accept()
        except socket.timeout:
            continue
        with client:
            client.settimeout(.25)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buffer = b""
            controller = None
            config = None
            while not stop.is_set():
                try:
                    packet = client.recv(65536)
                except socket.timeout:
                    continue
                if not packet:
                    break
                buffer += packet
                if len(buffer) > 1048576:
                    raise ValueError("Oversized Sai state")
                finished = False
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    state = json.loads(line)
                    if state.get("finish"):
                        finished = True
                        break
                    if controller is None:
                        config = state["hub_config"]
                        skill = config.get("skill", "")
                        profile = bundle / "policies/experimental" / f"{skill}.json" if skill else None
                        controller = CargoGodotController() if config["task"] == "cargo" else WorkshopController(bundle, stair_profile=profile)
                        if config["task"] != "cargo":
                            print("SAI_DEPLOYMENT " + json.dumps(dict(policy=controller.flat_policy_id,
                                  sha256=controller.flat_policy_sha256, driving_profile="sai-driving-20260913",
                                  module=str(Path(__file__).resolve()))), flush=True)
                    response = controller.command(state)
                    client.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
                    if trace:
                        trace.write(json.dumps(dict(config=config, state=state, command=response), separators=(",", ":")) + "\n")
                if finished:
                    break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("workshop", "science_station"), default="workshop")
    parser.add_argument("--choose-scene", action="store_true", help="Show the desktop scene picker")
    parser.add_argument("--drive-speed", type=drive_speed, default=DEFAULT_DRIVE_SPEED,
                        help="Sai flat cruise speed, including crouch, in m/s (default: 0.5)")
    parser.add_argument("--robot", choices=("microduck", "roller", "sai"), default="microduck")
    parser.add_argument("--task", default="drive", choices=("drive", "sort", "cargo18", "cargo25", "up20", "down20", "up40", "down40", "up60", "down60"))
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT") or shutil.which("godot"))
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "results/workshop-hub/runtime")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast-check", action="store_true", help="Unpaced headless plan; physics step and policy rates stay unchanged")
    parser.add_argument("--plan", type=Path, help="Timed integration test / capture plan")
    parser.add_argument("--output", type=Path, default=ROOT / "results/workshop-hub/play")
    parser.add_argument("--record", action="store_true", help="Timestamped native game frames and policy trace")
    parser.add_argument("--sai-controller", choices=("native", "python"), default="native",
                        help="Native in-process locomotion (default), or explicit Python/TCP oracle for manipulation")
    parser.add_argument("--sai-physics-hz", type=int, choices=SUPPORTED_PHYSICS_HZ, default=DEFAULT_PHYSICS_HZ,
                        help="Sai Jolt physics rate; the ONNX controller remains exactly 50 Hz")
    parser.add_argument("--sai-stair-profile", default="",
                        help="Experimental profile name under sai_policy/experimental for ascending stair tasks")
    parser.add_argument("--allow-quarantined-stair-profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the self-contained Godot runtime and exit")
    args = parser.parse_args(argv)
    if args.fast_check and not (args.headless and args.plan):
        parser.error("--fast-check requires --headless and --plan")
    if args.scene == "science_station" and args.task not in ("drive", "sort"):
        parser.error("Science station supports free exploration and scene pickup; use workshop for task courses")
    if not args.godot_bin:
        parser.error("Godot 4.7.2 is required")
    if args.sai_stair_profile:
        profile_path = (ROOT / "src/sim2sim/assets/sai/upstream/experimental"
                        / f"{args.sai_stair_profile}.json")
        if not profile_path.is_file():
            parser.error(f"Unknown Sai stair profile: {args.sai_stair_profile}")
        profile_status = str(json.loads(profile_path.read_text()).get("status", ""))
        if profile_status.startswith("quarantined-") and not args.allow_quarantined_stair_profile:
            parser.error(f"Sai stair profile {args.sai_stair_profile} is quarantined: {profile_status}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(SIM2SIM_VISUAL_STYLE="legacy", MD_WORKSHOP_COLLISIONS="1", MD_MODE="hub")
    bundle = prepare(args.runtime_dir.resolve(), args.godot_bin)
    if args.choose_scene or args.scene == "science_station":
        project = args.runtime_dir / "project.godot"
        project.write_text(project.read_text().replace('config/name="Robot Godot Workshop"',
                                                       'config/name="Robot Godot Worlds"'))
    options = dict(scene=args.scene, drive_speed=args.drive_speed, fast_check=args.fast_check, choose_scene=args.choose_scene and not args.headless and not args.plan, robot=args.robot, task=args.task, output=str(args.output), record=args.record, sai_controller=args.sai_controller, sai_stair_profile=args.sai_stair_profile, sai_physics_hz=args.sai_physics_hz,
                   plan=json.loads(args.plan.read_text()) if args.plan else {})
    (args.runtime_dir / "hub/options.json").write_text(json.dumps(options))
    if args.prepare_only:
        print(args.runtime_dir.resolve())
        return 0
    if args.sai_controller == "native":
        command = [args.godot_bin, "--path", str(args.runtime_dir.resolve()), "res://hub/main.tscn", "--disable-vsync", "--max-fps", "30"]
        if args.headless: command += ["--headless"]
        if args.fast_check: command += ["--fixed-fps", "30"]
        return subprocess.call(command)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(.25)
        stop = threading.Event()
        trace = (args.output / "sai-trace.jsonl").open("w") if args.record or args.plan else None
        command = [args.godot_bin, "--path", str(args.runtime_dir.resolve()), "res://hub/main.tscn", "--disable-vsync", "--max-fps", "30"]
        if args.headless:
            command += ["--headless"]
        if args.fast_check:
            command += ["--fixed-fps", "30"]
        command += ["--", f"--port={listener.getsockname()[1]}"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            service = pool.submit(serve, listener, stop, bundle, trace)
            child = subprocess.Popen(command)
            started = time.monotonic()
            try:
                while child.poll() is None:
                    stop.wait(.1)
                    if service.done():
                        service.result()
                    if args.plan and time.monotonic() - started > max(180, float(options["plan"].get("seconds",72))*6+60):
                        raise TimeoutError("Workshop plan exceeded its wall-time budget")
                result = child.returncode
            finally:
                stop.set()
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                service.result(timeout=5)
                if trace:
                    trace.close()
        if result == 74:
            return subprocess.call([str(ROOT / "run-leviathan003.sh")])
        if result == 73:
            return subprocess.call([str(ROOT / "run-leviathan.sh"), "--reference-runtime"])
        return result


if __name__ == "__main__":
    raise SystemExit(main())
