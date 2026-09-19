"""MuJoCo FK sampler for Godot reset body poses (one companion backend)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim2sim.backends.mujoco_backend import MujocoBackend
from sim2sim.obs import DEFAULT_HOME
from sim2sim.runner import apply_home_qpos


def joint_limits(mj: MujocoBackend) -> tuple[np.ndarray, np.ndarray]:
    """Per-actuator hinge ranges; unlimited joints get ±π."""
    lo = np.empty(mj.nu, dtype=np.float64)
    hi = np.empty(mj.nu, dtype=np.float64)
    for i in range(mj.nu):
        j = int(mj.model.actuator_trnid[i, 0])
        if int(mj.model.jnt_limited[j]):
            lo[i] = float(mj.model.jnt_range[j, 0])
            hi[i] = float(mj.model.jnt_range[j, 1])
        else:
            lo[i] = -math.pi
            hi[i] = math.pi
    return lo, hi


class HomePoseSampler:
    """Holds one ``MujocoBackend(current_limit_a=0)`` for FK of randomized home poses.

    Godot ``reset`` should still send ``ctrl=HOME`` (not the noisy q), matching play.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32).reshape(-1)
        self.reset_z = float(cfg.get("reset_z", 0.125))
        mjcf = Path(cfg["mjcf"])
        self.mj = MujocoBackend(
            mjcf,
            timestep=cfg.get("timestep", 0.005),
            current_limit_a=0.0,
            base_body=cfg.get("base_body", "trunk_base"),
        )
        self.joint_lo, self.joint_hi = joint_limits(self.mj)
        self._nominal_poses = self.nominal()
        self.support_names = support_body_pair(self._nominal_poses)
        self.ankle_z_nominal = _support_z_from_poses(self._nominal_poses, self.support_names)

    def nominal(self) -> list[dict]:
        """Deterministic home FK; same as ``play.capture_home_poses``."""
        apply_home_qpos(self.mj, self.home, z=self.reset_z)
        return self.mj.body_poses_mujoco()

    def sample(
        self,
        rng: np.random.Generator,
        *,
        yaw_range: tuple[float, float] = (-math.pi, math.pi),
        joint_noise_rad: float = 0.05,
        z: float | None = None,
        q_base: np.ndarray | None = None,
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        z0 = self.reset_z if z is None else float(z)
        yaw = float(rng.uniform(yaw_range[0], yaw_range[1]))
        base = self.home if q_base is None else np.asarray(q_base, dtype=np.float32).reshape(-1)
        noise = rng.uniform(-float(joint_noise_rad), float(joint_noise_rad), size=base.shape)
        q0 = np.clip(base.astype(np.float64) + noise, self.joint_lo, self.joint_hi).astype(
            np.float32
        )
        ctrl0 = self.home.astype(np.float32, copy=True)
        apply_home_qpos(self.mj, q0, z=z0)
        adr = self.mj.free_qposadr
        d = self.mj.data
        d.qpos[adr : adr + 3] = [0.0, 0.0, z0]
        d.qpos[adr + 3 : adr + 7] = [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]
        d.ctrl[:] = ctrl0
        d.qvel[:] = 0
        import mujoco

        mujoco.mj_forward(self.mj.model, d)
        poses = self.mj.body_poses_mujoco()
        return poses, q0, ctrl0

    def close(self) -> None:
        self.mj.close()


def support_body_pair(poses: list[dict]) -> tuple[str, str]:
    """Walk ankles, or roller ankle_l_v1 / tires if the walking names are absent."""
    names = {str(p.get("name")) for p in poses}
    for pair in (
        ("ankle_left", "ankle_right"),
        ("ankle_l_v1", "ankle_r_v1"),
        ("tire", "tire_3"),
    ):
        if pair[0] in names and pair[1] in names:
            return pair
    raise KeyError(f"no support bodies in poses: {sorted(names)}")


def _support_z_from_poses(poses: list[dict], names: tuple[str, str]) -> np.ndarray:
    by_name = {p["name"]: p for p in poses}
    zs = []
    for name in names:
        if name not in by_name:
            raise KeyError(f"nominal poses missing {name}")
        zs.append(float(by_name[name]["pos"][2]))
    return np.asarray(zs, dtype=np.float32)


def _ankle_z_from_poses(poses: list[dict]) -> np.ndarray:
    return _support_z_from_poses(poses, support_body_pair(poses))
