"""Interactive Godot viewer: hold-to-move + skill buttons, ONNX stays in Python."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.backends.mujoco_backend import MujocoBackend
from sim2sim.fall import fallen as pose_fallen
from sim2sim.godot_proc import GODOT_PROJECT, godot_bin, sim2sim_root
from sim2sim.obs import DEFAULT_HOME, build_obs
from sim2sim.paths import policies_dir
from sim2sim.play_input import PlayBrain, TwistLimits, relaunch_argv, wall_dt
from sim2sim.policy import OnnxPolicy, PolicyNumericError, PolicyShapeError
from sim2sim.policy_time import time_command
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.runner import apply_home_qpos, load_robot_cfg


ROOT = sim2sim_root()
POL = policies_dir()

FALL_RESET_S = 0.8
PUSH_MAX = 1.0


def _opt(path: Path) -> Path | None:
    return path if path.is_file() else None


def policy_search_dirs() -> list[Path]:
    """Repo policies/ plus MICRODUCK_POLICIES / sibling checkout. First hit wins."""
    cands = [
        POL,
        policies_dir(),
        ROOT / "policies",
        ROOT.parent / "policies",
        Path.home() / "Projects/MicroDuck/policies",
    ]
    env = os.environ.get("MICRODUCK_POLICIES")
    if env:
        cands.insert(0, Path(env))
    out: list[Path] = []
    seen: set[str] = set()
    for d in cands:
        try:
            r = Path(d).resolve()
        except OSError:
            continue
        key = str(r)
        if r.is_dir() and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _find_named(name: str) -> Path | None:
    for d in policy_search_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


ROLLER_LIMITS = TwistLimits(vmax_x=0.6, vmin_x=-0.5, vmax_y=0.0, vmin_y=0.0, vmax_ang=1.0)


def _prefer(godot_name: str, fallback: Path) -> Path | None:
    found = _find_named(godot_name)
    if found is not None:
        return found
    return _find_named(Path(fallback).name) or _opt(fallback)


def policy_paths(
    *, local_ppo: bool, roller: bool = False, walking: Path | None = None
) -> dict[str, Path | None]:
    if roller:
        return {
            "walking": _prefer("Roller_Godot.onnx", POL / "roller.onnx"),
            "standing": None,
            "roller_crouch": _prefer("RollerCrouch_Godot.onnx", POL / "roller_crouch.onnx"),
            "sitstand": None,
            "ground_pick": None,
            "kick_left": None,
            "kick_right": None,
            "roulade": None,
        }
    if walking is not None:
        walk_path = Path(walking)
    elif local_ppo:
        walk_path = _find_named("local_velocity_walk_run_idle.onnx") or (
            POL / "local-ppo/local_velocity_walk_run_idle.onnx"
        )
    else:
        walk_path = _prefer("Walk_Godot.onnx", POL / "alpha_walking.onnx") or (POL / "alpha_walking.onnx")
    return {
        "walking": _opt(walk_path),
        "standing": _prefer("Stand_Godot.onnx", POL / "alpha_stand.onnx"),
        "sitstand": _prefer("Sitstand_Godot.onnx", POL / "alpha_sitstand.onnx"),
        "ground_pick": _prefer("GroundPick_Godot.onnx", POL / "alpha_ground_pick.onnx"),
        "kick_left": _prefer("KickLeft_Godot.onnx", POL / "ball_kick_left.onnx"),
        "kick_right": _prefer("KickRight_Godot.onnx", POL / "ball_kick_right.onnx"),
        "roulade": _prefer("Roulade_Godot.onnx", POL / "roulade.onnx"),
    }


def ensure_godot_scene(cfg: dict) -> Path:
    spec = Path(cfg["godot_spec"])
    tscn = spec.with_name("robot.tscn")
    scene_ready = spec.is_file() and tscn.is_file()
    meshes = list((spec.parent / "meshes").glob("*.obj"))
    if scene_ready and all(p.with_suffix(p.suffix + ".import").is_file() for p in meshes):
        return spec
    mjcf = Path(cfg["mjcf"])
    if not mjcf.is_file():
        raise SystemExit(f"mjcf missing: {mjcf}")
    from mjcf2godot.convert import convert

    out = spec.parent
    if not scene_ready:
        print(f"converting {mjcf} → {out}")
        convert(mjcf, out)
    cmd = [godot_bin(), "--headless", "--path", str(GODOT_PROJECT), "--editor", "--import", "--quit"]
    print("godot import:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not spec.is_file() or not tscn.is_file():
        raise SystemExit(f"convert/import did not write {spec} / {tscn}")
    return spec


def capture_home_poses(cfg: dict) -> list[dict]:
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    z0 = float(cfg.get("reset_z", 0.125))
    mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
    apply_home_qpos(mj, home, z=z0)
    poses = mj.body_poses_mujoco()
    mj.close()
    for pose in poses:
        if pose["name"] == "ball":
            pose["pos"] = [5.0, 5.0, 0.035]
    return poses


def kick_ball_position(st, skill: str) -> list[float]:
    """Official ball offset in the robot's current heading frame at trigger."""
    w, x, y, z = st.base_quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    side = 0.042 if skill == "kick_left" else -0.042
    return [float(st.base_pos[0] + c * 0.09 - s * side),
            float(st.base_pos[1] + s * 0.09 + c * side), 0.035]


def load_bank(paths: dict[str, Path | None], home_len: int) -> dict[str, OnnxPolicy]:
    bank: dict[str, OnnxPolicy] = {}
    for name, path in paths.items():
        if path is None:
            continue
        bundle = OnnxPolicy(path)
        try:
            bundle.check_dims(home_len)
        except PolicyShapeError as e:
            raise SystemExit(f"policy {name} shape: {e}") from e
        bank[name] = bundle
        print(f"  loaded {name}: {path}")
    if "walking" not in bank and "standing" not in bank and "sitstand" not in bank:
        print(
            "ERROR: no walking/standing/sitstand ONNX found under policies/.\n"
            "错误：policies/ 下没有可用的 walking/standing/sitstand ONNX。\n"
            f"Roller mode requires: {POL / "roller.onnx"}\n"
            f"Walk mode requires:   {POL / "alpha_walking.onnx"}",
            flush=True,
        )
        raise SystemExit(2)
    return bank


def pick_session(bank: dict[str, OnnxPolicy], policy: str) -> OnnxPolicy:
    if policy in bank:
        return bank[policy]
    if "walking" in bank:
        return bank["walking"]
    return next(iter(bank.values()))


def fallen(st, timeout_acc: float, dt: float) -> tuple[bool, float]:
    if pose_fallen(st.base_quat_wxyz, st.base_pos):
        timeout_acc += dt
        if timeout_acc >= FALL_RESET_S:
            return True, 0.0
        return False, timeout_acc
    return False, 0.0


def random_push() -> np.ndarray:
    xy = np.random.normal(size=2)
    n = float(np.linalg.norm(xy))
    if n < 1e-9:
        xy = np.array([1.0, 0.0])
        n = 1.0
    xy = xy / n * PUSH_MAX
    return np.array([xy[0], xy[1], 0.0], dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    # Sai has active wheels, cargo sliders and a distinct 82D contract. Route it
    # to its pinned articulated adapter before the MicroDuck-specific loader.
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--robot")
    profile, remaining = profile_parser.parse_known_args(raw_argv)
    if profile.robot == "Sai_Agent_001":
        try:
            from sai_agent.cli import main as sai_main
        except ImportError as exc:
            raise SystemExit("Install the Sai profile first: uv sync --extra sai") from exc
        return sai_main(["godot", *remaining])
    p = argparse.ArgumentParser(description="Keyboard/HUD play loop on Godot/Jolt")
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--local-ppo", action="store_true", help="shortcut: local_ppo walking ONNX")
    p.add_argument("--walking", type=Path, default=None, help="override walking ONNX path")
    p.add_argument("--sprint", type=Path, help="optional walking sprint actor selected by left Shift+W")
    p.add_argument("--control-config",type=Path,help="Explicit versioned controller configuration used by standalone replay")
    p.add_argument(
        "--roller",
        action="store_true",
        help="roller-skate XML + roller.onnx (same as infer_policy --roller)",
    )
    p.add_argument(
        "--scene",
        type=str,
        default="res://main.tscn",
        help="Godot scene to run (default: flat main.tscn; rough forest: res://scenes/rough_forest_play.tscn)",
    )
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if args.roller:
        args.robot = ROOT / "robots/microduck_roller.json"
    elif args.robot.resolve() == (ROOT / "robots/microduck.json").resolve():
        args.robot = ROOT / "robots/microduck_ball.json"

    cfg = load_robot_cfg(args.robot)
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    scale = float(cfg.get("action_scale", 1.0))
    decimation = int(cfg.get("decimation", 4))
    dt = float(cfg.get("timestep", 0.005))
    dt_ctrl = decimation * dt
    spec = ensure_godot_scene(cfg)

    print("== sim2sim-play ==" + ("  [rollers]" if args.roller else "") + f"  scene={args.scene}")
    if args.walking is not None and not args.walking.is_file():
        raise SystemExit(f"walking ONNX missing: {args.walking}")
    paths = policy_paths(local_ppo=args.local_ppo, roller=args.roller, walking=args.walking)
    if args.sprint is not None and not args.roller:
        if not args.sprint.is_file():raise SystemExit(f'sprint ONNX missing: {args.sprint}')
        paths['sprint']=args.sprint
    bank = load_bank(paths, home_len=int(home.size))
    if 'sprint' in bank:
        actor=bank['sprint']
        if actor.obs_dim!=61 or actor.time_input_s or actor.heading_input or actor.yaw_memory_input or actor.task_input:
            raise PolicyShapeError('Sprint requires the 61D walking contract')
    task_contacts = None
    if any(actor.task_state is not None for actor in bank.values()):
        if not args.roller:
            raise PolicyShapeError('Roller task-state policies require the roller robot')
        from sim2sim.train.reset_poses import HomePoseSampler
        from sim2sim.research.world import roller_support_groups
        sampler = HomePoseSampler(cfg)
        try:
            task_contacts = roller_support_groups(sampler.mj.model,
                [body['name'] for body in json.loads(spec.read_text())['bodies']])
        finally:
            sampler.mj.close()
    if args.roller:
        lim = ROLLER_LIMITS
        use_stand = True
    else:
        walk = bank.get("walking")
        lim = walk.twist_limits if walk is not None else TwistLimits()
        use_stand = True if walk is None else walk.has_standing_partner
    from dataclasses import replace
    from sim2sim.motion_control import MotionControl
    controls={} if args.control_config is None else json.loads(args.control_config.read_text())
    motion_settings=controls.get('roller' if args.roller else 'walk',{})
    lim=replace(lim,**motion_settings.get('twist_limits',{}))
    motion=MotionControl(motion_settings)
    brain = PlayBrain(
        has_sprint=not args.roller and 'sprint' in bank,
        has_walking="walking" in bank,
        has_standing="standing" in bank and use_stand,
        has_sitstand="sitstand" in bank,
        has_pick="ground_pick" in bank,
        has_kick_left="kick_left" in bank,
        has_kick_right="kick_right" in bank,
        has_roulade="roulade" in bank,
        has_roller_crouch="roller_crouch" in bank,
        has_stand_hold="standing" in bank,
        lim=lim,
    )
    poses = capture_home_poses(cfg)
    backend = GodotBackend(
        spec,
        timestep=dt,
        headless=False,
        scene=args.scene,
        base_body=cfg.get("base_body", "trunk_base"),
        current_limit_a=cfg.get("current_limit_a", 1.75),
    )
    print("\n点 Godot 窗口后按住键（和 MuJoCo infer_policy 相同）：")
    if args.roller:
        print("  W/↑ 滑行   S/↓ 刹车   A/← 左转   D/→ 右转   空格 Idle")
        print("  无侧移（Q/E 无效）  vmax_x=0.6")
        print("  2/Y 下蹲滑行后起身（5 秒）")
        print("  6 切回路走+技能   0 重置   P 推一把   Esc 退出")
    else:
        print("  W/↑ 前进   S/↓ 后退   A/← 左转   D/→ 右转   Q/E 平移   空格 Idle")
        print("  1/G 捡地   2/Y 坐下   3/K 左踢   4/L 右踢   5/R 前滚")
        print("  6 切到轮滑   0 重置   P 推一把   Esc 退出")
    print("  窗口底部也有同样的按钮。\n")

    last_action = np.zeros(int(home.size), dtype=np.float32)
    maneuver_heading = np.array([1.,0.])
    active_policy = None
    held: set[str] = set()
    press_order: list[str] = []
    taps: list[str] = []
    fall_acc = 0.0
    report_bodies = None if task_contacts is None else sum(task_contacts, [])
    st = backend.reset(ctrl=home, bodies=poses, report_bodies=report_bodies)
    next_t = time.perf_counter()
    hz_n = 0
    hz_t0 = time.perf_counter()
    infer_ms = 0.0
    step_ms = 0.0
    try:
        while True:
            out = brain.tick(held, taps, dt_ctrl, press_order=press_order)
            if out.quit:
                print("quit")
                break
            if out.switch_robot:
                want_roller = not args.roller
                next_paths = policy_paths(
                    local_ppo=args.local_ppo, roller=want_roller, walking=args.walking
                )
                if want_roller and next_paths.get("walking") is None:
                    miss = POL / "roller.onnx"
                    print(
                        "ERROR: cannot switch to roller — policies/roller.onnx is missing.\n"
                        "错误：无法切换到轮滑模式 — 缺少 policies/roller.onnx。\n"
                        f"Expected at: {miss}\n"
                        "Download BEST_roller.onnx from pollen-robotics/microduck-simulator "
                        "(app/public/policies/) and save as policies/roller.onnx.",
                        flush=True,
                    )
                    raise SystemExit(2)
                print(
                    "Switching to roller-skate robot (wheels XML + roller.onnx)..."
                    if want_roller
                    else "Switching to walking robot (feet + skills)..."
                )
                backend.close()
                argv = relaunch_argv(sys.argv, want_roller=want_roller, executable=sys.executable)
                print(f"Relaunch {'roller' if want_roller else 'walk'}: {' '.join(argv)}")
                os.execv(sys.executable, argv)
                raise SystemExit(f"execv failed: {argv}")
            do_reset = out.reset
            if not do_reset and not brain.sit and not brain._busy():
                did_fall, fall_acc = fallen(st, fall_acc, dt_ctrl)
                if did_fall:
                    print("auto-reset: fallen")
                    do_reset = True
            if do_reset:
                brain.reset_motion()
                motion.reset()
                active_policy = None
                last_action[:] = 0.0
                fall_acc = 0.0
                st = backend.reset(ctrl=home, bodies=poses, report_bodies=report_bodies)
                held, taps = set(), []
                next_t = time.perf_counter()
                continue
            if out.push:
                backend.nudge(random_push())
            selected_policy = 'sprint' if out.sprint else out.policy
            sess = pick_session(bank, selected_policy)
            if active_policy != selected_policy:
                sess.reset_context(); active_policy = selected_policy
            cmd = out.command
            if sess.time_input_s:
                duration = brain.roulade_duration if out.policy == 'roulade' else (
                    brain.kick_duration if out.policy in ('kick_left','kick_right') else 0.)
                if not duration or sess.time_input_s != duration:
                    raise PolicyShapeError("Time-input policy does not match this maneuver")
                rotation = quat_wxyz_to_mat(st.base_quat_wxyz)
                if out.started_skill == out.policy:
                    yaw = np.arctan2(rotation[1,0],rotation[0,0])
                    maneuver_heading = np.array([np.cos(yaw),np.sin(yaw)])
                cmd = time_command(duration - brain.behavior_t, sess.time_input_s,
                                   rotation if sess.heading_input else None,maneuver_heading)
            skill='roller' if args.roller and out.policy=='walking' else selected_policy
            cmd=motion.command(cmd,st,skill,dt_ctrl)
            obs = build_obs(st, last_action, cmd, home=home)
            from sim2sim.policy_state import inject_state
            obs=inject_state(obs,st,sess.state_input)
            if sess.task_state is not None:
                from sim2sim.policy_task_state import contacts_from_raw
                obs = sess.task_state.observe(obs,
                    contacts_from_raw(st.extra['raw'], task_contacts), st.t)
            t_inf = time.perf_counter()
            try:
                action = sess.infer(obs)
            except PolicyNumericError as e:
                print(f"policy numeric error: {e}")
                brain.reset_motion()
                motion.reset()
                active_policy = None
                last_action[:] = 0.0
                fall_acc = 0.0
                st = backend.reset(ctrl=home, bodies=poses, report_bodies=report_bodies)
                held, taps = set(), []
                next_t = time.perf_counter()
                continue
            infer_ms += (time.perf_counter() - t_inf) * 1000.0
            last_action = action.astype(np.float32, copy=True)
            ctrl = home + last_action * scale
            t_step = time.perf_counter()
            ball = kick_ball_position(st, out.started_skill) if out.started_skill in ("kick_left", "kick_right") else None
            st = backend.step(ctrl, n_substeps=decimation, hud=out.status, place_ball=ball,
                report='research' if task_contacts is not None else None)
            step_ms += (time.perf_counter() - t_step) * 1000.0
            raw = st.extra.get("raw") or {}
            held = {str(x) for x in (raw.get("held") or [])}
            press_order = [str(x) for x in (raw.get("held_order") or [])]
            taps = [str(x) for x in (raw.get("taps") or [])]
            ts = raw.get("time_scale", 1.0)
            next_t += wall_dt(dt_ctrl, ts)
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                next_t = time.perf_counter()
            hz_n += 1
            now = time.perf_counter()
            if now - hz_t0 >= 2.0:
                target = 1.0 / wall_dt(dt_ctrl, ts)
                print(
                    f"play {hz_n / (now - hz_t0):.1f} Hz  (1×=50, 滑条目标 {target:.1f})  "
                    f"infer={infer_ms/hz_n:.1f}ms  godot_step={step_ms/hz_n:.1f}ms"
                )
                hz_n = 0
                hz_t0 = now
                infer_ms = 0.0
                step_ms = 0.0
    except (ConnectionError, RuntimeError) as e:
        print(f"godot closed: {e}")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
