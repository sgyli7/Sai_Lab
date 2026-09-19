"""Calibration tests C1–C3: joint PD, STAND hold, drop. No policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim2sim.obs import DEFAULT_HOME
from sim2sim.runner import apply_home_qpos, load_robot_cfg, make_backend
from sim2sim.godot_proc import sim2sim_root

ROOT = sim2sim_root()


def _roll_pd(backend, ctrl: np.ndarray, seconds: float, decimation: int = 4) -> dict:
    n = int(round(seconds / (decimation * backend.dt)))
    qs, pos, t = [], [], []
    st = None
    for _ in range(n):
        st = backend.step(ctrl, n_substeps=decimation)
        qs.append(st.q.copy())
        pos.append(st.base_pos.copy())
        t.append(st.t)
    return {
        "t": np.asarray(t),
        "q": np.asarray(qs),
        "base_pos": np.asarray(pos),
        "final": st,
    }


def c1_step(backend, cfg: dict, joint_index: int = 2) -> dict:
    """Pinned-base single-joint step (default left_hip_pitch)."""
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
    z0 = float(cfg.get("reset_z", 0.125))
    if backend.name == "mujoco":
        apply_home_qpos(backend, home.astype(np.float32), z=z0)
        backend.reset(qpos=backend.data.qpos.copy(), qvel=backend.data.qvel.copy(), ctrl=home, pin_base=True)
        poses = backend.body_poses_mujoco()
        _ = poses
    else:
        from sim2sim.backends.mujoco_backend import MujocoBackend

        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, home.astype(np.float32), z=z0)
        poses = mj.body_poses_mujoco()
        mj.close()
        backend.reset(ctrl=home, bodies=poses, pin_base=True)
    target = home.copy()
    target[joint_index] = home[joint_index] + 0.3
    out = _roll_pd(backend, target, seconds=1.5)
    q_end = float(out["q"][-1, joint_index])
    err = abs(q_end - float(target[joint_index]))
    return {
        "name": "C1_joint_step",
        "joint_index": joint_index,
        "target": float(target[joint_index]),
        "q_end": q_end,
        "abs_err": err,
        "q_traj": out["q"][:, joint_index],
        "t": out["t"],
    }


def c2_stand(backend, cfg: dict) -> dict:
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
    z0 = float(cfg.get("reset_z", 0.125))
    if backend.name == "mujoco":
        apply_home_qpos(backend, home.astype(np.float32), z=z0)
        backend.reset(qpos=backend.data.qpos.copy(), qvel=backend.data.qvel.copy(), ctrl=home, pin_base=False)
    else:
        from sim2sim.backends.mujoco_backend import MujocoBackend

        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, home.astype(np.float32), z=z0)
        poses = mj.body_poses_mujoco()
        mj.close()
        backend.reset(ctrl=home, bodies=poses)
    out = _roll_pd(backend, home, seconds=2.0)
    z = out["base_pos"][:, 2]
    qerr = np.linalg.norm(out["q"] - home[None, :], axis=1)
    return {
        "name": "C2_stand_pd",
        "z_min": float(z.min()),
        "z_final": float(z[-1]),
        "z0": z0,
        "qerr_mean": float(qerr.mean()),
        "qerr_final": float(qerr[-1]),
        "fell": bool(z.min() < 0.08),
        "t": out["t"],
        "z": z,
    }


def c3_drop(backend, cfg: dict) -> dict:
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
    z0 = float(cfg.get("reset_z", 0.125)) + 0.05
    if backend.name == "mujoco":
        apply_home_qpos(backend, home.astype(np.float32), z=z0)
        backend.reset(qpos=backend.data.qpos.copy(), qvel=backend.data.qvel.copy(), ctrl=home)
    else:
        from sim2sim.backends.mujoco_backend import MujocoBackend

        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, home.astype(np.float32), z=z0)
        poses = mj.body_poses_mujoco()
        mj.close()
        backend.reset(ctrl=home, bodies=poses)
    out = _roll_pd(backend, home, seconds=1.5)
    z = out["base_pos"][:, 2]
    return {
        "name": "C3_drop",
        "z0": z0,
        "z_min": float(z.min()),
        "z_final": float(z[-1]),
        "t": out["t"],
        "z": z,
    }


def run_calib(kind: str, cfg: dict) -> dict:
    backend = make_backend(kind, cfg, headless=True)
    try:
        r1 = c1_step(backend, cfg)
        r2 = c2_stand(backend, cfg)
        r3 = c3_drop(backend, cfg)
    finally:
        backend.close()
    return {"backend": kind, "C1": _jsonable(r1), "C2": _jsonable(r2), "C3": _jsonable(r3)}


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--backend", choices=("mujoco", "godot"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    cfg = load_robot_cfg(args.robot)
    result = run_calib(args.backend, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("t", "z", "q_traj")} if isinstance(v, dict) else v for k, v in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
