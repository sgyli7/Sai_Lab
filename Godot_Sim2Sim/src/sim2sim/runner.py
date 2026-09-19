"""ONNX policy rollout on a physics backend. Writes a .npz trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim2sim.obs import DEFAULT_HOME, build_obs, command_13
from sim2sim.paths import expand_cfg, sim2sim_root
from sim2sim.policy import OnnxPolicy
from sim2sim.policy_time import time_command
from sim2sim.coords import quat_wxyz_to_mat

ROOT = sim2sim_root()


def load_robot_cfg(path: Path) -> dict:
    return expand_cfg(json.loads(Path(path).read_text()))


def apply_home_qpos(backend, home: np.ndarray, z: float = 0.125) -> None:
    """Match infer_policy main(): freejoint at z, joints at HOME, ctrl=HOME."""
    import mujoco

    if backend.name != "mujoco":
        return
    m, d = backend.model, backend.data
    adr = backend.free_qposadr
    d.qpos[adr : adr + 3] = [0.0, 0.0, z]
    d.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(backend.joint_qpos_indices):
        d.qpos[qpos_idx] = home[i]
    d.ctrl[:] = home
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def make_backend(kind: str, cfg: dict, *, headless: bool = True):
    mjcf = Path(cfg["mjcf"])
    if kind == "mujoco":
        from sim2sim.backends.mujoco_backend import MujocoBackend

        return MujocoBackend(
            mjcf,
            timestep=cfg.get("timestep", 0.005),
            current_limit_a=cfg.get("current_limit_a", 1.75),
            base_body=cfg.get("base_body", "trunk_base"),
        )
    if kind == "godot":
        from sim2sim.backends.godot_backend import GodotBackend

        spec = Path(cfg.get("godot_spec", str(ROOT / "godot/generated/microduck/robot_spec.json")))
        return GodotBackend(
            spec,
            timestep=cfg.get("timestep", 0.005),
            headless=headless,
            base_body=cfg.get("base_body", "trunk_base"),
            current_limit_a=cfg.get("current_limit_a", 1.75),
        )
    raise ValueError(kind)


def run_rollout(
    backend,
    policy: OnnxPolicy,
    cfg: dict,
    *,
    schedule: list[dict] | None = None,
) -> dict:
    policy.reset_context()
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    scale = float(cfg.get("action_scale", 1.0))
    decimation = int(cfg.get("decimation", 4))
    z0 = float(cfg.get("reset_z", 0.125))
    schedule = schedule or cfg["schedule"]

    if backend.name == "mujoco":
        apply_home_qpos(backend, home, z=z0)
        st = backend.reset(qpos=backend.data.qpos.copy(), qvel=backend.data.qvel.copy(), ctrl=home)
        poses = backend.body_poses_mujoco()
    else:
        # Godot: poses come from a companion MuJoCo reset
        from sim2sim.backends.mujoco_backend import MujocoBackend

        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, home, z=z0)
        poses = mj.body_poses_mujoco()
        mj.close()
        st = backend.reset(ctrl=home, bodies=poses)

    last_action = np.zeros(policy.act_dim, dtype=np.float32)
    initial_rotation = quat_wxyz_to_mat(st.base_quat_wxyz)
    initial_yaw = np.arctan2(initial_rotation[1,0],initial_rotation[0,0])
    initial_heading = np.array([np.cos(initial_yaw),np.sin(initial_yaw)])
    elapsed = 0.
    logs = {
        "t": [],
        "q": [],
        "qd": [],
        "base_pos": [],
        "base_quat": [],
        "base_linvel": [],
        "base_angvel_local": [],
        "action": [],
        "ctrl": [],
        "obs": [],
        "cmd": [],
        "phase": [],
    }

    for phase in schedule:
        cmd_vel = np.array(phase.get("vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        seconds = float(phase["seconds"])
        n = int(round(seconds / (decimation * backend.dt)))
        command = command_13(cmd_vel)
        for _ in range(n):
            if getattr(policy,'time_input_s',0.):
                rotation = quat_wxyz_to_mat(st.base_quat_wxyz)
                command = time_command(elapsed,policy.time_input_s,
                    rotation if policy.heading_input else None,initial_heading)
            obs = build_obs(st, last_action, command, home=home)
            action = policy.infer(obs)
            last_action = action.copy()
            ctrl = home + action * scale
            st = backend.step(ctrl, n_substeps=decimation)
            elapsed += decimation * backend.dt
            logs["t"].append(st.t)
            logs["q"].append(st.q.copy())
            logs["qd"].append(st.qd.copy())
            logs["base_pos"].append(st.base_pos.copy())
            logs["base_quat"].append(st.base_quat_wxyz.copy())
            logs["base_linvel"].append(st.base_linvel.copy())
            logs["base_angvel_local"].append(st.base_angvel_local.copy())
            logs["action"].append(action.copy())
            logs["ctrl"].append(np.asarray(ctrl, dtype=np.float64))
            logs["obs"].append(obs.copy())
            logs["cmd"].append(command.copy())
            logs["phase"].append(phase.get("name", ""))

    out = {k: np.asarray(v) for k, v in logs.items() if k != "phase"}
    out["phase"] = np.asarray(logs["phase"])
    out["backend"] = np.asarray(backend.name)
    out["policy"] = np.asarray(str(policy.path))
    out["dt_control"] = np.asarray(decimation * backend.dt)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--backend", choices=("mujoco", "godot"), required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--windowed", action="store_true")
    args = p.parse_args(argv)
    cfg = load_robot_cfg(args.robot)
    policy = OnnxPolicy(args.onnx)
    backend = make_backend(args.backend, cfg, headless=not args.windowed)
    try:
        traj = run_rollout(backend, policy, cfg)
    finally:
        backend.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **traj)
    pos = traj["base_pos"]
    print(
        f"wrote {args.out} steps={len(traj['t'])} "
        f"xy={np.linalg.norm(pos[-1,:2]-pos[0,:2]):.3f}m "
        f"z_min={pos[:,2].min():.3f} z_final={pos[-1,2]:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
