"""Independent A/B evaluation for the 8 non-walking Godot-tuned skills."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.fall import fallen
from sim2sim.obs import DEFAULT_HOME, build_obs, command_13
from sim2sim.paths import policies_dir, sim2sim_root
from sim2sim.play import capture_home_poses, ensure_godot_scene
from sim2sim.policy import OnnxPolicy
from sim2sim.runner import load_robot_cfg
from sim2sim.train.rewards import sit_target_q

ROOT = sim2sim_root()
POL = policies_dir()
MAX_WORKERS = 8


SKILLS: tuple[dict[str, Any], ...] = (
    {
        "name": "standing",
        "a": POL / "alpha_stand.onnx",
        "b": POL / "Stand_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 10.0,
        "kind": "idle",
    },
    {
        "name": "sitstand",
        "a": POL / "alpha_sitstand.onnx",
        "b": POL / "Sitstand_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 8.0,
        "kind": "sitstand",
    },
    {
        "name": "ground_pick",
        "a": POL / "alpha_ground_pick.onnx",
        "b": POL / "GroundPick_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 4.0,
        "kind": "pick",
    },
    {
        "name": "kick_left",
        "a": POL / "ball_kick_left.onnx",
        "b": POL / "KickLeft_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 3.0,
        "kind": "kick",
        "kick_foot": 0,
    },
    {
        "name": "kick_right",
        "a": POL / "ball_kick_right.onnx",
        "b": POL / "KickRight_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 3.0,
        "kind": "kick",
        "kick_foot": 1,
    },
    {
        "name": "roulade",
        "a": POL / "roulade.onnx",
        "b": POL / "Roulade_Godot.onnx",
        "robot": ROOT / "robots/microduck.json",
        "seconds": 5.0,
        "kind": "roulade",
    },
    {
        "name": "roller",
        "a": POL / "roller.onnx",
        "b": POL / "Roller_Godot.onnx",
        "robot": ROOT / "robots/microduck_roller.json",
        "seconds": 8.0,
        "kind": "roller",
    },
    {
        "name": "roller_crouch",
        "a": POL / "roller_crouch.onnx",
        "b": POL / "RollerCrouch_Godot.onnx",
        "robot": ROOT / "robots/microduck_roller.json",
        "seconds": 8.0,
        "kind": "idle",
    },
)


def _cmd_for(kind: str, t: float, seconds: float, *, sit: bool = False) -> np.ndarray:
    if kind == "sitstand":
        return command_13(np.array([1.0 if sit else 0.0, 0.0, 0.0], dtype=np.float32))
    if kind == "pick":
        phase = (t / max(seconds, 1e-6)) % 1.0
        return command_13(np.array([math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase), 0.0]))
    if kind == "roller":
        return command_13(np.array([0.3, 0.0, 0.0], dtype=np.float32))
    return command_13(np.zeros(3, dtype=np.float32))


def _run_episode(
    *,
    onnx: Path,
    robot: Path,
    kind: str,
    seconds: float,
    seed: int,
    sit: bool = False,
    kick_foot: int = 1,
) -> dict[str, Any]:
    cfg = load_robot_cfg(robot)
    spec = ensure_godot_scene(cfg)
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    scale = float(cfg.get("action_scale", 1.0))
    dt_phys = float(cfg.get("timestep", 0.005))
    decimation = int(cfg.get("decimation", 4))
    dt = dt_phys * decimation
    policy = OnnxPolicy(onnx)
    policy.check_dims(int(home.size))
    poses = capture_home_poses(cfg)
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-math.pi, math.pi)) if kind in ("idle", "roller") else 0.0
    backend = GodotBackend(
        spec,
        timestep=dt_phys,
        headless=True,
        base_body=cfg.get("base_body", "trunk_base"),
        current_limit_a=float(cfg.get("current_limit_a", 1.75)),
    )
    last = np.zeros(int(home.size), dtype=np.float32)
    z_hist: list[float] = []
    q_hist: list[np.ndarray] = []
    gyro_y = 0.0
    max_foot_h = 0.0
    fell = False
    xy0: np.ndarray | None = None
    xy_last = np.zeros(2, dtype=np.float64)
    speed_xy_sum = 0.0
    # Sitting / rolling through 70° is the task; do not cut the episode on `fallen`.
    abort_on_fall = kind not in ("sitstand", "roulade")
    try:
        st = backend.reset(ctrl=home, bodies=poses)
        n = int(round(seconds / dt))
        for i in range(n):
            t = i * dt
            cmd = _cmd_for(kind, t, seconds, sit=sit)
            obs = build_obs(st, last, cmd, home)
            act = policy.infer(obs)
            ctrl = home + act * scale
            st = backend.step(ctrl, n_substeps=decimation, report="lite")
            last = act
            z_hist.append(float(st.base_pos[2]))
            q_hist.append(np.asarray(st.q, dtype=np.float32).copy())
            gyro_y += abs(float(st.base_angvel_local[1])) * dt
            xy = np.asarray(st.base_pos[:2], dtype=np.float64)
            if xy0 is None:
                xy0 = xy.copy()
            xy_last = xy
            lv = np.asarray(st.base_linvel[:2], dtype=np.float64)
            speed_xy_sum += float(np.hypot(lv[0], lv[1]))
            feet = (st.extra or {}).get("feet") or []
            if kick_foot < len(feet):
                pos = feet[kick_foot].get("pos") or [0, 0, 0]
                max_foot_h = max(max_foot_h, float(pos[2]))
            if fallen(st.base_quat_wxyz, st.base_pos):
                fell = True
                if abort_on_fall:
                    break
    finally:
        backend.close()
    z = np.asarray(z_hist, dtype=np.float64) if z_hist else np.array([float("nan")])
    q = np.stack(q_hist) if q_hist else np.zeros((1, 14), dtype=np.float32)
    sit_q = sit_target_q(home)
    n_obs = max(len(z_hist), 1)
    xy_disp = float(np.hypot(*(xy_last - xy0))) if xy0 is not None else float("nan")
    out = {
        "fell": bool(fell),
        "sit": bool(sit),
        "survival_s": float(len(z_hist) * dt),
        "mean_trunk_z": float(np.nanmean(z)),
        "min_trunk_z": float(np.nanmin(z)),
        "final_trunk_z": float(z[-1]),
        "yaw_progress_rad": float(gyro_y),
        "max_foot_z": float(max_foot_h),
        "xy_disp": xy_disp,
        "mean_xy_speed": float(speed_xy_sum / n_obs),
        "pose_err_home": float(np.mean(np.abs(q[-1] - home))) if len(q) else float("nan"),
        "pose_err_sit": float(np.mean(np.abs(q[-1] - sit_q))) if len(q) else float("nan"),
        "onnx": str(onnx),
        "kind": kind,
        "seed": int(seed),
    }
    if kind == "pick" and z_hist:
        n_half = max(1, len(z_hist) // 2)
        out["approach_min_z"] = float(np.min(z[:n_half]))
    return out


def _mean_bool(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(r[key]) for r in rows]))


def eval_skill(spec: dict[str, Any], *, seeds: int, workers: int) -> dict[str, Any]:
    a_path, b_path = Path(spec["a"]), Path(spec["b"])
    if not a_path.is_file():
        return {"name": spec["name"], "skip": f"missing A {a_path}"}
    if not b_path.is_file():
        return {"name": spec["name"], "skip": f"missing B {b_path}"}
    jobs = []
    for label, path in (("A", a_path), ("B", b_path)):
        for seed in range(int(seeds)):
            sit = spec["kind"] == "sitstand" and (seed % 2 == 0)
            jobs.append((label, path, seed, sit))
    rows: dict[str, list[dict]] = {"A": [], "B": []}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {
            pool.submit(
                _run_episode,
                onnx=path,
                robot=Path(spec["robot"]),
                kind=str(spec["kind"]),
                seconds=float(spec["seconds"]),
                seed=seed,
                sit=sit,
                kick_foot=int(spec.get("kick_foot", 1)),
            ): label
            for label, path, seed, sit in jobs
        }
        for fut in as_completed(futs):
            rows[futs[fut]].append(fut.result())
    report = {"name": spec["name"], "kind": spec["kind"], "seeds": int(seeds), "A": {}, "B": {}}
    for lab in ("A", "B"):
        got = rows[lab]
        report[lab] = {
            "fell_rate": _mean_bool(got, "fell"),
            "mean_trunk_z": _mean_bool(got, "mean_trunk_z"),
            "min_trunk_z": _mean_bool(got, "min_trunk_z"),
            "survival_s": _mean_bool(got, "survival_s"),
            "pose_err_home": _mean_bool(got, "pose_err_home"),
            "pose_err_sit": _mean_bool(got, "pose_err_sit"),
            "yaw_progress_rad": _mean_bool(got, "yaw_progress_rad"),
            "max_foot_z": _mean_bool(got, "max_foot_z"),
            "xy_disp": _mean_bool(got, "xy_disp"),
            "mean_xy_speed": _mean_bool(got, "mean_xy_speed"),
            "n": len(got),
        }
        if spec["kind"] == "pick":
            report[lab]["approach_min_z"] = _mean_bool(got, "approach_min_z")
        if spec["kind"] == "sitstand":
            sit_rows = [r for r in got if r.get("sit")]
            stand_rows = [r for r in got if not r.get("sit")]
            report[lab]["sit"] = {
                "n": len(sit_rows),
                "mean_trunk_z": _mean_bool(sit_rows, "mean_trunk_z"),
                "final_trunk_z": _mean_bool(sit_rows, "final_trunk_z"),
                "pose_err_sit": _mean_bool(sit_rows, "pose_err_sit"),
            }
            report[lab]["stand"] = {
                "n": len(stand_rows),
                "mean_trunk_z": _mean_bool(stand_rows, "mean_trunk_z"),
                "final_trunk_z": _mean_bool(stand_rows, "final_trunk_z"),
                "pose_err_home": _mean_bool(stand_rows, "pose_err_home"),
                "fell_rate": _mean_bool(stand_rows, "fell"),
            }
    a_fell, b_fell = report["A"]["fell_rate"], report["B"]["fell_rate"]
    report["b_fewer_falls"] = bool(b_fell <= a_fell)
    return report


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Godot skill A/B", "", "A = factory ONNX. B = `*_Godot.onnx`.", ""]
    lines.append("| skill | A fell | B fell | A z | B z | B fewer falls | notes |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for r in results:
        if r.get("skip"):
            lines.append(f"| {r['name']} | — | — | — | — | — | {r['skip']} |")
            continue
        extra = f"kind={r['kind']}"
        if r["kind"] == "idle":
            extra += (
                f"; pose_home A={r['A']['pose_err_home']:.3f} B={r['B']['pose_err_home']:.3f}"
                f"; |ωy|dt A={r['A']['yaw_progress_rad']:.3f} B={r['B']['yaw_progress_rad']:.3f}"
            )
        elif r["kind"] == "sitstand" and "sit" in r["A"] and "sit" in r["B"]:
            extra += (
                f"; sit_z A={r['A']['sit']['final_trunk_z']:.3f} B={r['B']['sit']['final_trunk_z']:.3f}"
                f"; pose_sit A={r['A']['sit']['pose_err_sit']:.3f} B={r['B']['sit']['pose_err_sit']:.3f}"
                f"; stand_z A={r['A']['stand']['final_trunk_z']:.3f} B={r['B']['stand']['final_trunk_z']:.3f}"
                f"; pose_home A={r['A']['stand']['pose_err_home']:.3f} B={r['B']['stand']['pose_err_home']:.3f}"
            )
        elif r["kind"] == "pick":
            extra += (
                f"; approach_min_z A={r['A'].get('approach_min_z', float('nan')):.3f}"
                f" B={r['B'].get('approach_min_z', float('nan')):.3f}"
                f"; pose_home A={r['A']['pose_err_home']:.3f} B={r['B']['pose_err_home']:.3f}"
            )
        elif r["kind"] == "kick":
            extra += f"; max_foot_z A={r['A']['max_foot_z']:.3f} B={r['B']['max_foot_z']:.3f}"
        elif r["kind"] == "roller":
            extra += (
                f"; xy_disp A={r['A']['xy_disp']:.3f} B={r['B']['xy_disp']:.3f}"
                f"; xy_speed A={r['A']['mean_xy_speed']:.3f} B={r['B']['mean_xy_speed']:.3f}"
            )
        elif r["kind"] == "roulade":
            extra += (
                f"; yaw_progress A={r['A']['yaw_progress_rad']:.3f} B={r['B']['yaw_progress_rad']:.3f}"
                f"; pose_home A={r['A']['pose_err_home']:.3f} B={r['B']['pose_err_home']:.3f}"
            )
        lines.append(
            f"| {r['name']} | {r['A']['fell_rate']:.2f} | {r['B']['fell_rate']:.2f} | "
            f"{r['A']['mean_trunk_z']:.3f} | {r['B']['mean_trunk_z']:.3f} | "
            f"{'yes' if r['b_fewer_falls'] else 'no'} | {extra} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim2sim-eval-skill")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--skill", action="append", default=None, help="repeatable filter by SKILLS.name")
    p.add_argument("--out", type=Path, default=ROOT / "results/skill_godot_eval/report.md")
    args = p.parse_args(argv)
    wanted = set(args.skill) if args.skill else None
    results = []
    for spec in SKILLS:
        if wanted and spec["name"] not in wanted:
            continue
        print(f"== {spec['name']} ==", flush=True)
        results.append(eval_skill(spec, seeds=args.seeds, workers=args.workers))
    write_report(results, args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
