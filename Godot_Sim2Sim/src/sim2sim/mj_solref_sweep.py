#!/usr/bin/env python3
"""MuJoCo solref/solimp sweep for C3 drop calibration.

Proves (or disproves) that the C3 gap (MJ z_min=0.046 vs Godot z_min=0.031)
is 100% due to the soft contact model in MuJoCo (solref/solimp), by
progressively hardening the contact on the compiled MjModel and observing
whether z_min drops to ~0.031 and t=0.92 tilt drops to ~30°.

Usage:
    cd /path/to/MicroDuck-Godot-Simi2Sim
    export PYTHONPATH="$PWD/src"
    uv run python -m sim2sim.mj_solref_sweep

Does NOT modify calib.py, run.sh, or the XML; contact override is applied
AFTER MjModel is compiled, on an in-memory copy only.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.godot_proc import sim2sim_root
from sim2sim.obs import DEFAULT_HOME
from sim2sim.runner import apply_home_qpos, load_robot_cfg, make_backend

ROOT = sim2sim_root()

# ─── Sweep configurations ────────────────────────────────────────────────────

SWEEPS = [
    {
        "name": "S0_baseline",
        "desc": "XML default solref/solimp (MJ baseline)",
        "solref": None,  # keep XML values
        "solimp": None,
    },
    {
        "name": "S1_timeconst_2dt",
        "desc": "timeconst 2*dt=0.01 (half of default)",
        "solref": np.array([0.01, 1.0]),
        "solimp": None,  # keep default solimp
    },
    {
        "name": "S2_timeconst_dt",
        "desc": "timeconst dt=0.005 (equal to one timestep)",
        "solref": np.array([0.005, 1.0]),
        "solimp": None,
    },
    {
        "name": "S3_tight",
        "desc": "timeconst dt + solimp width 1e-5 (~Jolt 0.2mm slop level)",
        "solref": np.array([0.005, 1.0]),
        "solimp": np.array([0.99, 0.9999, 1e-5, 0.5, 2.0]),
    },
    {
        "name": "S4_stiffness",
        "desc": "Direct stiffness form solref=(-1e6,-1e4) + tight solimp",
        "solref": np.array([-1e6, -1e4]),
        "solimp": np.array([0.99, 0.9999, 1e-5, 0.5, 2.0]),
    },
]

# Probe time points to report (seconds)
PROBE_TIMES = {0.12, 0.40, 0.80, 0.92, 1.00, 1.14, 1.26, 1.50, 2.00, 2.50}
SIM_SECONDS = 2.5
N_SUBSTEPS = 4


# ─── Helpers ─────────────────────────────────────────────────────────────────

def tilt_deg(quat_wxyz: np.ndarray) -> float:
    w, x, y, zq = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    gz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, gz))))


def geom_body_name(model: mujoco.MjModel, geom_id: int) -> str:
    bid = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def classify_geom(model: mujoco.MjModel, geom_id: int) -> str:
    gn = geom_name(model, geom_id)
    bn = geom_body_name(model, geom_id)
    if "foot_collision" in gn:
        return "foot"
    if bn == "jaw_soft":
        return "jaw"
    if "hip" in bn.lower():
        return "hip"
    return "other"


def contact_stats(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Contact analysis at the current step."""
    n = int(data.ncon)
    max_pen = 0.0
    foot_n = 0
    jaw_n = 0
    hip_n = 0
    xs: list[float] = []

    for i in range(n):
        c = data.contact[i]
        pen = float(-c.dist)
        if pen > max_pen:
            max_pen = pen
        g1 = int(c.geom[0])
        g2 = int(c.geom[1])
        k1 = classify_geom(model, g1)
        k2 = classify_geom(model, g2)
        cats = {k1, k2}
        if "foot" in cats:
            foot_n += 1
            xs.append(float(c.pos[0]))
        if "jaw" in cats:
            jaw_n += 1
            xs.append(float(c.pos[0]))
        if "hip" in cats:
            hip_n += 1

    x_span = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
    x_min = min(xs) if xs else float("nan")
    x_max = max(xs) if xs else float("nan")
    return {
        "ncon": n,
        "max_pen_mm": max_pen * 1000.0,
        "foot_n": foot_n,
        "jaw_n": jaw_n,
        "hip_n": hip_n,
        "x_span_mm": x_span * 1000.0,
        "cx_min": x_min,
        "cx_max": x_max,
    }


# ─── Main sweep ──────────────────────────────────────────────────────────────

def run_one(cfg: dict, sw: dict) -> dict:
    """Run one 2.5 s C3 drop with the given solref/solimp override."""
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
    z0 = float(cfg.get("reset_z", 0.125)) + 0.05  # 0.175, same as c3_drop / probe

    backend = make_backend("mujoco", cfg, headless=True)
    m: mujoco.MjModel = backend.model
    d: mujoco.MjData = backend.data

    # ── apply solref/solimp override ──────────────────────────────────────
    orig_solref = m.geom_solref.copy()
    orig_solimp = m.geom_solimp.copy()

    if sw["solref"] is not None:
        m.geom_solref[:] = sw["solref"]
    if sw["solimp"] is not None:
        m.geom_solimp[:] = sw["solimp"]

    # Also print the actual floor geom values to confirm override landed
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_gid >= 0:
        actual_solref = m.geom_solref[floor_gid].copy()
        actual_solimp = m.geom_solimp[floor_gid].copy()
    else:
        actual_solref = m.geom_solref[0].copy()
        actual_solimp = m.geom_solimp[0].copy()

    # ── reset ─────────────────────────────────────────────────────────────
    apply_home_qpos(backend, home.astype(np.float32), z=z0)
    backend.reset(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=home)

    # ── simulate 2.5 s ────────────────────────────────────────────────────
    n_steps = int(round(SIM_SECONDS / (N_SUBSTEPS * backend.dt)))
    z_min = z0
    z_min_t = 0.0
    probes: list[dict] = []

    # First-contact tracking
    first_contact_done = False
    first_contact_pen = 0.0
    first_contact_ncon = 0
    first_contact_stats: dict = {}

    for _ in range(n_steps):
        st = backend.step(home, n_substeps=N_SUBSTEPS)
        t = st.t
        z = float(st.base_pos[2])

        if z < z_min:
            z_min = z
            z_min_t = t

        # first contact detection
        if not first_contact_done and int(d.ncon) > 0:
            cs = contact_stats(m, d)
            if cs["foot_n"] > 0 and cs["max_pen_mm"] > 0.001:
                first_contact_done = True
                first_contact_pen = cs["max_pen_mm"]
                first_contact_ncon = int(d.ncon)
                first_contact_stats = cs

        # probe times
        t_key = round(t + 1e-9, 2)
        if t_key in PROBE_TIMES or abs(t - 2.50) < 1e-6:
            cs = contact_stats(m, d)
            probes.append({
                "t": t,
                "z": z,
                "tilt": tilt_deg(st.base_quat_wxyz),
                "vz": float(st.base_linvel[2]),
                "jaw_x": float(d.xipos[backend.base_body_id][0]),  # approx; jaw is child
                **cs,
            })

    backend.close()

    return {
        "sweep": sw["name"],
        "desc": sw["desc"],
        "actual_solref": actual_solref.tolist(),
        "actual_solimp": actual_solimp.tolist(),
        "z_min": z_min,
        "z_min_t": z_min_t,
        "first_pen_mm": first_contact_pen,
        "first_ncon": first_contact_ncon,
        "first_stats": first_contact_stats,
        "probes": probes,
    }


def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Sweep':<22} {'solref':>14} {'1st_pen(mm)':>12} {'z_min':>8} {'t(zmin)':>8} {'jaw_x@0.92':>12}")
    print("=" * 90)
    for r in results:
        sr = r["actual_solref"]
        sr_str = f"({sr[0]:.3g},{sr[1]:.3g})"
        pen = r["first_pen_mm"]
        # find probe nearest t=0.92
        p92 = min(r["probes"], key=lambda p: abs(p["t"] - 0.92), default=None)
        jaw_x_str = f"{p92['jaw_x']:.4f}" if p92 else "n/a"
        print(
            f"{r['sweep']:<22} {sr_str:>14} {pen:>12.3f} {r['z_min']:>8.4f}"
            f" {r['z_min_t']:>8.3f} {jaw_x_str:>12}"
        )
    print("=" * 90)

    print("\n── Probe timeline per sweep ──")
    for r in results:
        print(f"\n  [{r['sweep']}] first_pen={r['first_pen_mm']:.3f}mm  z_min={r['z_min']:.4f}@t={r['z_min_t']:.3f}")
        hdr = f"  {'t':>5} {'z':>8} {'tilt':>7} {'vz':>7} {'ncon':>6} {'pen_mm':>9} {'foot_n':>7} {'jaw_n':>6} {'hip_n':>6} {'xspan_mm':>9}"
        print(hdr)
        for p in r["probes"]:
            print(
                f"  {p['t']:>5.2f} {p['z']:>8.4f} {p['tilt']:>7.1f} {p['vz']:>7.3f}"
                f" {p['ncon']:>6} {p['max_pen_mm']:>9.3f} {p['foot_n']:>7} {p['jaw_n']:>6}"
                f" {p['hip_n']:>6} {p['x_span_mm']:>9.1f}"
            )


def main() -> int:
    cfg = load_robot_cfg(ROOT / "robots/microduck.json")

    results = []
    for sw in SWEEPS:
        print(f"\n>>> Running {sw['name']}: {sw['desc']}")
        r = run_one(cfg, sw)
        results.append(r)
        p92 = min(r["probes"], key=lambda p: abs(p["t"] - 0.92), default=None)
        tilt92 = p92["tilt"] if p92 else float("nan")
        print(
            f"    first_pen={r['first_pen_mm']:.3f}mm  z_min={r['z_min']:.4f}@t={r['z_min_t']:.3f}"
            f"  tilt@0.92={tilt92:.1f}°  foot@0.92={p92['foot_n'] if p92 else '?'}"
            f"  jaw@0.92={p92['jaw_n'] if p92 else '?'}"
            f"  hip@0.92={p92['hip_n'] if p92 else '?'}"
        )

    print_results(results)

    # ── Verdict ──────────────────────────────────────────────────────────────
    s0 = next(r for r in results if r["sweep"] == "S0_baseline")
    s4 = results[-1]
    print("\n── Verdict ──")
    pen_drop = s0["first_pen_mm"] - s4["first_pen_mm"]
    zmin_drop = s0["z_min"] - s4["z_min"]
    print(f"  Penetration S0→S4: {s0['first_pen_mm']:.3f}mm → {s4['first_pen_mm']:.3f}mm  (Δ={pen_drop:.3f}mm)")
    print(f"  z_min        S0→S4: {s0['z_min']:.4f} → {s4['z_min']:.4f}  (Δ={zmin_drop:.4f})")

    p92_s0 = min(s0["probes"], key=lambda p: abs(p["t"] - 0.92), default=None)
    p92_s4 = min(s4["probes"], key=lambda p: abs(p["t"] - 0.92), default=None)
    if p92_s0 and p92_s4:
        print(f"  tilt@0.92    S0→S4: {p92_s0['tilt']:.1f}° → {p92_s4['tilt']:.1f}°")
        print(f"  hip@0.92     S0→S4: {p92_s0['hip_n']} → {p92_s4['hip_n']}")

    if s4["z_min"] < 0.035 and (p92_s4 and p92_s4["tilt"] < 45):
        print("\n  ✓ MAIN BRANCH: C3 gap is 100% solref/solimp. Godot 0.031 matches hardened MJ.")
    else:
        print("\n  ✗ SECONDARY BRANCH: Hardened MJ z_min still > 0.035 — second cause exists.")
        print("    Continue: check condim/friction cone / contact count distribution.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
