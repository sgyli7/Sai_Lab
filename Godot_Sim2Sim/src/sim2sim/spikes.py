"""Phase 0 spikes: S1 lockstep, S2 mass/layers, S3 hinge PD vs MuJoCo."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.godot_proc import sim2sim_root, spawn_godot, stop_godot

ROOT = sim2sim_root()
SPIKE_XML = """
<mujoco model="pd_hinge">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler" iterations="50"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-3.1416 3.1416"
             damping="0.053" armature="0" frictionloss="0"/>
      <inertial pos="0 0 -0.1" mass="0.2" diaginertia="0.0004 0.0004 0.00005"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" kp="0.55" kv="0" forcerange="-0.96 0.96" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def spike_s1() -> dict:
    proc, port, client = spawn_godot("res://spikes/lockstep.tscn")
    try:
        hello = client.call({"cmd": "hello"})
        hz = int(hello["ticks_per_second"])
        dts = []
        ticks0 = None
        n_cmd = 10
        sub = 4
        for i in range(n_cmd):
            msg = client.call({"cmd": "step", "n_substeps": sub})
            dts.append(float(msg["dt"]))
            ticks0 = int(msg["ticks"]) if ticks0 is None else ticks0
            ticks = int(msg["ticks"])
        expected_ticks = n_cmd * sub
        ok_ticks = ticks == expected_ticks
        dt_mean = float(np.mean(dts))
        ok_dt = abs(dt_mean - 0.005) < 1e-4 and abs(hz - 200) < 1
        return {
            "name": "S1_lockstep",
            "ok": bool(ok_ticks and ok_dt),
            "hz": hz,
            "ticks": ticks,
            "expected_ticks": expected_ticks,
            "dt_mean": dt_mean,
            "hello": hello,
        }
    finally:
        stop_godot(proc, client)


def spike_s2() -> dict:
    proc, port, client = spawn_godot("res://spikes/mass_layers.tscn")
    try:
        client.call({"cmd": "hello"})
        # 1.0 s of free spin about Godot Y (MuJoCo Z)
        n = int(1.0 / 0.005)
        msg = client.call({"cmd": "step", "n_substeps": n})
        om = np.asarray(msg["omega"], dtype=float)
        # principal-axis spin should stay on Y, |w| ~ 5
        ok_spin = abs(om[1] - 5.0) < 0.35 and abs(om[0]) < 0.2 and abs(om[2]) < 0.2
        contacts_ab = int(msg["contacts_ab"])
        contacts_cd = int(msg["contacts_cd"])
        # A vs B: layer 1 vs 2, overlapping, should NOT contact
        # C vs D: both layer 1, overlapping, SHOULD contact
        ok_layers = contacts_ab == 0 and contacts_cd > 0
        return {
            "name": "S2_mass_layers",
            "ok": bool(ok_spin and ok_layers),
            "omega": om.tolist(),
            "contacts_ab": contacts_ab,
            "contacts_cd": contacts_cd,
            "ok_spin": bool(ok_spin),
            "ok_layers": bool(ok_layers),
        }
    finally:
        stop_godot(proc, client)


def _mujoco_pd_traj(seconds: float = 1.5, target: float = 0.5) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(SPIKE_XML)
        xml = f.name
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qs = []
    n = int(seconds / 0.005)
    d.ctrl[0] = target
    for _ in range(n):
        mujoco.mj_step(m, d)
        qs.append(float(d.qpos[0]))
    return np.asarray(qs)


def spike_s3() -> dict:
    mj = _mujoco_pd_traj()
    proc, port, client = spawn_godot("res://spikes/pd_hinge.tscn")
    try:
        client.call({"cmd": "hello"})
        client.call({"cmd": "reset"})
        gd = []
        # 1.5 s, substeps=1 to sample every physics tick
        for _ in range(len(mj)):
            msg = client.call({"cmd": "step", "ctrl": [0.5], "n_substeps": 1})
            gd.append(float(msg["q"][0]))
        gd = np.asarray(gd)
        n = min(len(mj), len(gd))
        rmse = float(np.sqrt(np.mean((mj[:n] - gd[:n]) ** 2)))
        # If apply_torque PD is in the same ballpark as MuJoCo position actuator.
        ok = rmse < 0.25 and abs(gd[n - 1] - mj[n - 1]) < 0.35
        return {
            "name": "S3_pd_hinge",
            "ok": bool(ok),
            "rmse": rmse,
            "q_end_mujoco": float(mj[n - 1]),
            "q_end_godot": float(gd[n - 1]),
            "drive": "apply_torque_pd",
        }
    finally:
        stop_godot(proc, client)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("s1", "s2", "s3"), default=None)
    args = p.parse_args(argv)
    results = []
    fns = [("s1", spike_s1), ("s2", spike_s2), ("s3", spike_s3)]
    failed = False
    for key, fn in fns:
        if args.only and args.only != key:
            continue
        print(f"== spike {key} ==")
        r = fn()
        results.append(r)
        print(r)
        if not r.get("ok"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
