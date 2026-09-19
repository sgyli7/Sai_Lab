"""Independent kick_left / kick_right headless regression (soft known-fail by default).

Does NOT hard-fail ./run.sh walk semantics. Godot kick fall is currently expected
(KNOWN_FAIL). Use --mode hard only after plant fix is authorized green.

Examples:
  cd sim2sim && uv run sim2sim-kick-gate
  uv run sim2sim-kick-gate --skills kick_left --mode soft
  ./run_kick_gate.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sim2sim.compare import compare_pair
from sim2sim.godot_proc import sim2sim_root
from sim2sim.paths import policies_dir
from sim2sim.policy import OnnxPolicy
from sim2sim.runner import load_robot_cfg, make_backend, run_rollout
from sim2sim.skill_metrics import classify_kick_pair, fall_timeline, pair_rmse

ROOT = sim2sim_root()
POL = policies_dir()

DEFAULT_ONNX = {
    "kick_left": POL / "ball_kick_left.onnx",
    "kick_right": POL / "ball_kick_right.onnx",
}

# Align PlayBrain: kick ~4s, vel=0
KICK_SCHEDULE = [{"name": "kick", "seconds": 4.0, "vel": [0.0, 0.0, 0.0]}]


def run_one(skill: str, onnx: Path, cfg: dict, out_dir: Path) -> tuple[dict, dict, dict]:
    policy = OnnxPolicy(onnx)
    infos: dict[str, dict] = {}
    trajs: dict[str, dict] = {}
    for backend_name in ("mujoco", "godot"):
        print(f"=== {skill} / {backend_name} ===", flush=True)
        backend = make_backend(backend_name, cfg, headless=True)
        try:
            traj = run_rollout(backend, policy, cfg, schedule=KICK_SCHEDULE)
        finally:
            backend.close()
        npz = out_dir / f"{skill}_{backend_name}.npz"
        np.savez_compressed(npz, **traj)
        info = fall_timeline(traj)
        info["skill"] = skill
        info["backend"] = backend_name
        info["npz"] = str(npz)
        info["onnx"] = str(onnx)
        infos[backend_name] = info
        trajs[backend_name] = traj
        print(
            json.dumps(
                {
                    k: info[k]
                    for k in (
                        "skill",
                        "backend",
                        "fell",
                        "fall_t_gate",
                        "fall_t_play",
                        "z_min",
                        "z_final",
                        "max_tilt_deg",
                        "xy_m",
                        "action_peak",
                        "q_max_abs_delta",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    pair = pair_rmse(trajs["mujoco"], trajs["godot"])
    thr = cfg.get("thresholds", {})
    cmp = compare_pair(trajs["mujoco"], trajs["godot"], thr)
    pair["compare_hard_fail"] = cmp["hard_fail"]
    pair["compare_warnings"] = cmp["warnings"]
    return infos["mujoco"], infos["godot"], pair


def write_report(rows: list[dict], pairs: list[dict], out_dir: Path, mode: str) -> Path:
    lines = [
        "# Kick headless gate",
        "",
        f"mode: `{mode}` (soft = Godot kick fall is KNOWN_FAIL, does not poison `./run.sh`)",
        "",
        "| skill | backend | fell | fall_t_gate | fall_t_play | z_min | max_tilt | xy_m | action_peak | q_Δmax |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['skill']} | {r['backend']} | {r['fell']} | {r['fall_t_gate']} | {r['fall_t_play']} | "
            f"{r['z_min']:.3f} | {r['max_tilt_deg']:.1f} | {r['xy_m']:.3f} | {r['action_peak']:.3f} | "
            f"{r['q_max_abs_delta']:.3f} |"
        )
    lines += ["", "## MJ vs GD", "| skill | q_rmse | xy_rmse | status | notes |", "|---|---|---|---|---|"]
    for p in pairs:
        notes = ", ".join(p.get("known_fail", []) + p.get("hard_fail", []) + p.get("warnings", [])) or "—"
        lines.append(
            f"| {p['skill']} | {p['q_rmse']:.4f} | {p['xy_rmse']:.4f} | {p['status']} | {notes} |"
        )
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kick left/right headless Sim2Sim gate (soft known-fail)")
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/kick_gate")
    p.add_argument(
        "--skills",
        nargs="+",
        default=["kick_left", "kick_right"],
        choices=list(DEFAULT_ONNX.keys()),
    )
    p.add_argument(
        "--mode",
        choices=("soft", "hard"),
        default="soft",
        help="soft: Godot-only fall → KNOWN_FAIL exit 0; hard: same → HARD_FAIL exit 1",
    )
    args = p.parse_args(argv)

    cfg = load_robot_cfg(args.robot)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    known_fail = args.mode == "soft"

    rows: list[dict] = []
    pairs: list[dict] = []
    any_hard = False
    any_known = False

    for skill in args.skills:
        onnx = Path(DEFAULT_ONNX[skill])
        if not onnx.exists():
            print(f"MISSING onnx {onnx}", flush=True)
            return 2
        mj_info, gd_info, pair = run_one(skill, onnx, cfg, out_dir)
        rows.extend([mj_info, gd_info])
        gate = classify_kick_pair(mj_info, gd_info, known_fail=known_fail)
        entry = {"skill": skill, **pair, **gate}
        pairs.append(entry)
        any_hard = any_hard or bool(gate["hard_fail"])
        any_known = any_known or bool(gate["known_fail"])
        print(
            f"GATE {skill}: status={gate['status']} q_rmse={pair['q_rmse']:.4f} "
            f"xy_rmse={pair['xy_rmse']:.4f}",
            flush=True,
        )

    summary = {
        "mode": args.mode,
        "cases": rows,
        "pairs": pairs,
        "note": (
            "Independent of ./run.sh. Soft mode records Godot kick fall as KNOWN_FAIL "
            "and exits 0 so LOCAL_GATE / walk SIM2SIM_RUN stay green."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report = write_report(rows, pairs, out_dir, args.mode)
    print(report.read_text(), flush=True)

    if any_hard:
        print(f"KICK_GATE: HARD FAIL  results={out_dir}", flush=True)
        return 1
    if any_known:
        print(f"KICK_GATE: KNOWN_FAIL (soft, exit 0)  results={out_dir}", flush=True)
        return 0
    print(f"KICK_GATE: PASS  results={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
