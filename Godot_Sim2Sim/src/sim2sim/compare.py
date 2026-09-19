"""Compare two .npz rollouts. Writes report.md, metrics.json, PNG plots; exit 1 on hard-fail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim2sim.fall import fallen, tilt_deg
from sim2sim.godot_proc import sim2sim_root

ROOT = sim2sim_root()


def _fft_cadence(q: np.ndarray, dt: float) -> float:
    if len(q) < 16:
        return 0.0
    x = q - q.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), dt)
    spec[0] = 0
    i = int(np.argmax(spec))
    return float(freqs[i])


def metrics_one(traj: dict) -> dict:
    pos = np.asarray(traj["base_pos"])
    quat = np.asarray(traj["base_quat"])
    q = np.asarray(traj["q"])
    t = np.asarray(traj["t"])
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.02
    z = pos[:, 2]
    tilt = np.array([tilt_deg(q) for q in quat], dtype=np.float64)
    fell = bool(any(fallen(q, p) for q, p in zip(quat, pos)))
    xy = np.linalg.norm(pos[-1, :2] - pos[0, :2])
    return {
        "steps": int(len(t)),
        "xy_m": float(xy),
        "z_min": float(z.min()),
        "z_final": float(z[-1]),
        "max_tilt_deg": float(tilt.max()),
        "fell": fell,
        "q_std": q.std(axis=0).tolist(),
        "cadence_hz_left_knee": _fft_cadence(q[:, 3], dt) if q.shape[1] > 3 else 0.0,
        "action_rms": float(np.sqrt(np.mean(np.asarray(traj["action"]) ** 2))) if "action" in traj else 0.0,
    }


def compare_pair(a: dict, b: dict, thresholds: dict) -> dict:
    ma, mb = metrics_one(a), metrics_one(b)
    na, nb = int(len(np.asarray(a["t"]))), int(len(np.asarray(b["t"])))
    hard = []
    soft = []
    if na != nb:
        hard.append(f"length_mismatch mujoco={na} godot={nb}")
        q_rmse = xy_rmse = z_rmse = float("nan")
    else:
        qa, qb = np.asarray(a["q"]), np.asarray(b["q"])
        pa, pb = np.asarray(a["base_pos"]), np.asarray(b["base_pos"])
        q_rmse = float(np.sqrt(np.mean((qa - qb) ** 2)))
        xy_rmse = float(np.sqrt(np.mean(np.sum((pa[:, :2] - pb[:, :2]) ** 2, axis=1))))
        z_rmse = float(np.sqrt(np.mean((pa[:, 2] - pb[:, 2]) ** 2)))
    if ma["fell"] and not mb["fell"]:
        hard.append("mujoco_fell_godot_stood")
    if mb["fell"] and not ma["fell"]:
        hard.append("godot_fell_mujoco_stood")
    if mb["fell"] and ma["fell"]:
        soft.append("both_fell")
    if na == nb:
        if q_rmse > float(thresholds.get("q_rmse_hard", 1.5)):
            hard.append(f"q_rmse={q_rmse:.3f}")
        elif q_rmse > float(thresholds.get("q_rmse_warn", 0.3)):
            soft.append(f"q_rmse={q_rmse:.3f}")
        if xy_rmse > float(thresholds.get("xy_rmse_hard", 2.0)):
            hard.append(f"xy_rmse={xy_rmse:.3f}")
        elif xy_rmse > float(thresholds.get("xy_rmse_warn", 0.25)):
            soft.append(f"xy_rmse={xy_rmse:.3f}")
        # Catch lockstep freeze (Godot restarts from rest every 20 ms): MJ walks,
        # GD shuffles ~0.14 m, xy_rmse stays under the 2.5 m hard cap.
        xy_stall = float(thresholds.get("xy_stall_ratio", 0.5))
        xy_ref = float(thresholds.get("xy_stall_ref_m", 0.8))
        if (not ma["fell"]) and ma["xy_m"] >= xy_ref and mb["xy_m"] < xy_stall * ma["xy_m"]:
            hard.append(f"godot_xy_stalled={mb['xy_m']:.3f}_vs_{ma['xy_m']:.3f}")
    return {
        "mujoco": ma,
        "godot": mb,
        "q_rmse": q_rmse,
        "xy_rmse": xy_rmse,
        "z_rmse": z_rmse,
        "hard_fail": hard,
        "warnings": soft,
    }


def _plots(a: dict, b: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ta, tb = np.asarray(a["t"]), np.asarray(b["t"])
    pa, pb = np.asarray(a["base_pos"]), np.asarray(b["base_pos"])
    qa, qb = np.asarray(a["q"]), np.asarray(b["q"])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(ta, pa[:, 2], label="mujoco")
    ax[0, 0].plot(tb, pb[:, 2], label="godot")
    ax[0, 0].set_title("base z")
    ax[0, 0].legend()
    ax[0, 1].plot(pa[:, 0], pa[:, 1], label="mujoco")
    ax[0, 1].plot(pb[:, 0], pb[:, 1], label="godot")
    ax[0, 1].set_title("xy path")
    ax[0, 1].axis("equal")
    ax[1, 0].plot(ta, qa[:, 2], label="mj hip_pitch")
    ax[1, 0].plot(tb, qb[:, 2], label="gd hip_pitch")
    ax[1, 0].set_title("left_hip_pitch")
    ax[1, 1].plot(ta, qa[:, 3], label="mj knee")
    ax[1, 1].plot(tb, qb[:, 3], label="gd knee")
    ax[1, 1].set_title("left_knee")
    fig.tight_layout()
    fig.savefig(out_dir / "compare.png", dpi=120)
    plt.close(fig)


def write_report(cmp: dict, out_dir: Path, title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Metrics")
    lines.append("| | MuJoCo | Godot |")
    lines.append("|---|---|---|")
    for k in ("xy_m", "z_min", "z_final", "max_tilt_deg", "fell", "cadence_hz_left_knee", "action_rms"):
        lines.append(f"| {k} | {cmp['mujoco'][k]} | {cmp['godot'][k]} |")
    lines.append("")
    lines.append(f"- q_rmse: {cmp['q_rmse']:.4f} rad")
    lines.append(f"- xy_rmse: {cmp['xy_rmse']:.4f} m")
    lines.append(f"- z_rmse: {cmp['z_rmse']:.4f} m")
    lines.append("")
    lines.append("## Gate")
    if cmp["hard_fail"]:
        lines.append("HARD FAIL: " + ", ".join(cmp["hard_fail"]))
    else:
        lines.append("HARD FAIL: none")
    if cmp["warnings"]:
        lines.append("WARN: " + ", ".join(cmp["warnings"]))
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mujoco", type=Path, required=True)
    p.add_argument("--godot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--title", default="Sim2Sim compare")
    args = p.parse_args(argv)
    cfg = json.loads(args.robot.read_text()) if args.robot.exists() else {}
    thr = cfg.get("thresholds", {})
    a = dict(np.load(args.mujoco, allow_pickle=True))
    b = dict(np.load(args.godot, allow_pickle=True))
    cmp = compare_pair(a, b, thr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(cmp, indent=2))
    _plots(a, b, args.out)
    write_report(cmp, args.out, args.title)
    print((args.out / "report.md").read_text())
    return 1 if cmp["hard_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
