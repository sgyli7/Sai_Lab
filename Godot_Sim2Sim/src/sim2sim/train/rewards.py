"""Vectorized walk rewards (numpy). Port of mjlab/microduck_rl velocity terms.

Whole-body ``angular_momentum`` is skipped: Godot lite step has no angmom sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

# Actuator order (robot_walk.xml): left 0-4, neck/head 5-8, right 9-13.
LEG_JOINT_IDX = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int64)
HEAD_JOINT_IDX = np.array([5, 6, 7, 8], dtype=np.int64)

# variable_posture stds from microduck_velocity_env_cfg (legs only).
_STD_STANDING_LEGS = np.array(
    [0.1, 0.05, 0.15, 0.15, 0.1, 0.1, 0.05, 0.15, 0.15, 0.1], dtype=np.float32
)
_STD_WALKING_LEGS = np.array(
    [0.3, 0.05, 0.4, 0.4, 0.25, 0.3, 0.05, 0.4, 0.4, 0.25], dtype=np.float32
)

TERM_NAMES: tuple[str, ...] = (
    "track_lin_vel",
    "track_ang_vel",
    "upright",
    "air_time",
    "pose_legs",
    "foot_clearance",
    "foot_swing_height",
    "action_rate_l2",
    "foot_slip",
    "body_ang_vel",
    "head_pose_tracking",
    "head_pose_bias",
    "dof_pos_limits",
    "posture_pose",
    "posture_height",
    "mouth_proximity",
    "pick_return",
    "approach_height",
    "kick_swing",
    "roulade_progress",
    "roulade_land",
)

# sitstand SITTING_TARGET_OVERRIDES on HOME (microduck_sitstand_env_cfg.py).
SIT_JOINT_OVERRIDES: dict[int, float] = {
    1: 0.0,
    2: -0.4079,
    3: 1.35,
    4: 0.0,
    10: 0.0,
    11: 0.4079,
    12: -1.35,
    13: 0.0,
}
SIT_Z = 0.060
STAND_Z = 0.115


@dataclass
class RewardConfig:
    """Weights and kernel params. Weight 0 disables a term."""

    track_lin_vel: float = 2.0
    track_lin_vel_std2: float = 0.1
    track_ang_vel: float = 2.0
    track_ang_vel_std2: float = 0.5
    upright: float = 2.0
    upright_std2: float = 0.05
    air_time: float = 3.0
    air_time_min: float = 0.125
    air_time_max: float = 0.3
    air_time_cmd_threshold: float = 0.01
    pose_legs: float = 1.0
    pose_walking_threshold: float = 0.01
    foot_clearance: float = -2.0
    foot_clearance_target: float = 0.02
    foot_swing_height: float = -0.25
    foot_swing_target: float = 0.02
    action_rate_l2: float = -0.1
    foot_slip: float = -0.1
    body_ang_vel: float = -0.05
    head_pose_tracking: float = 2.0
    head_pose_tracking_std: float = 0.5
    head_pose_bias: float = 0.0
    head_pose_bias_tau_s: float = 1.0
    dof_pos_limits: float = -1.0
    posture_pose: float = 0.0
    posture_pose_std: float = 0.15
    posture_height: float = 0.0
    posture_height_std: float = 0.03
    sit_z: float = SIT_Z
    stand_z: float = STAND_Z
    mouth_proximity: float = 0.0
    mouth_std: float = 0.03
    pick_return: float = 0.0
    approach_height: float = 0.0
    approach_z: float = 0.075
    approach_z_std: float = 0.04
    kick_swing: float = 0.0
    kick_window_s: float = 1.2
    kick_foot: str = "right"
    roulade_progress: float = 0.0
    roulade_land: float = 0.0
    roulade_complete_rad: float = 4.5
    roulade_rate_cap: float = 3.0
    scale_by_dt: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> RewardConfig:
        if not raw:
            return cls()
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: raw[k] for k in raw if k in allowed}
        return cls(**kwargs)


def track_lin_vel(
    v_cmd_xy: np.ndarray, v_body_xy: np.ndarray, std2: float = 0.1
) -> np.ndarray:
    """exp(-|v_cmd_xy - v_body_xy|^2 / std2)."""
    err = np.asarray(v_cmd_xy, dtype=np.float32) - np.asarray(v_body_xy, dtype=np.float32)
    sq = np.sum(err * err, axis=-1)
    return np.exp(-sq / float(std2)).astype(np.float32)


def track_ang_vel(wz_cmd: np.ndarray, gyro_z: np.ndarray, std2: float = 0.5) -> np.ndarray:
    """exp(-(wz_cmd - gyro_z)^2 / std2)."""
    d = np.asarray(wz_cmd, dtype=np.float32).reshape(-1) - np.asarray(gyro_z, dtype=np.float32).reshape(
        -1
    )
    return np.exp(-(d * d) / float(std2)).astype(np.float32)


def upright(grav: np.ndarray, std2: float = 0.05) -> np.ndarray:
    """mjlab upright: exp(-|grav_xy|^2 / std2). Upright projected gravity is (0,0,-1)."""
    g = np.asarray(grav, dtype=np.float32)
    xy = g[..., 0] * g[..., 0] + g[..., 1] * g[..., 1]
    return np.exp(-xy / float(std2)).astype(np.float32)


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_yaw_frame(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate world linvel into the yaw (heading) frame. vz is unchanged."""
    yaw = yaw_from_quat_wxyz(quat)
    c = np.cos(yaw)
    s = np.sin(yaw)
    v = np.asarray(v_world, dtype=np.float64).reshape(-1, 3)
    bx = c * v[:, 0] + s * v[:, 1]
    by = -s * v[:, 0] + c * v[:, 1]
    return np.stack([bx, by, v[:, 2]], axis=-1).astype(np.float32)


def sit_target_q(home: np.ndarray) -> np.ndarray:
    q = np.asarray(home, dtype=np.float32).reshape(14).copy()
    for idx, val in SIT_JOINT_OVERRIDES.items():
        q[idx] = float(val)
    return q


def pick_phase_from_cmd(cmd13: np.ndarray) -> np.ndarray:
    c = np.asarray(cmd13, dtype=np.float32).reshape(-1, 13)
    ph = np.arctan2(c[:, 1], c[:, 0]) / (2.0 * np.pi)
    return np.mod(ph, 1.0).astype(np.float32)


def posture_pose(
    q: np.ndarray,
    home: np.ndarray,
    sit_q: np.ndarray,
    sit_flag: np.ndarray,
    std: float = 0.15,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    sit_q = np.asarray(sit_q, dtype=np.float32).reshape(14)
    w = np.clip(np.asarray(sit_flag, dtype=np.float32).reshape(-1), 0.0, 1.0)[:, None]
    target = w * sit_q[None, :] + (1.0 - w) * home[None, :]
    err = q[:, LEG_JOINT_IDX] - target[:, LEG_JOINT_IDX]
    z = np.mean((err * err) / (float(std) ** 2), axis=1)
    return np.exp(-z).astype(np.float32)


def posture_height(
    z: np.ndarray,
    sit_flag: np.ndarray,
    sit_z: float = SIT_Z,
    stand_z: float = STAND_Z,
    std: float = 0.03,
) -> np.ndarray:
    w = np.clip(np.asarray(sit_flag, dtype=np.float32).reshape(-1), 0.0, 1.0)
    tgt = w * float(sit_z) + (1.0 - w) * float(stand_z)
    d = np.asarray(z, dtype=np.float32).reshape(-1) - tgt
    return np.exp(-((d / float(std)) ** 2)).astype(np.float32)


def mouth_proximity(mouth_z: np.ndarray, phase: np.ndarray, std: float = 0.03) -> np.ndarray:
    z = np.asarray(mouth_z, dtype=np.float32).reshape(-1)
    ph = np.asarray(phase, dtype=np.float32).reshape(-1)
    approach = (ph < 0.5).astype(np.float32)
    prox = np.exp(-((z / float(std)) ** 2)).astype(np.float32)
    prox = np.where(np.isfinite(z), prox, 0.0)
    return (prox * approach).astype(np.float32)


def approach_height(
    base_z: np.ndarray,
    phase: np.ndarray,
    target: float = 0.075,
    std: float = 0.04,
) -> np.ndarray:
    """Dense crouch during pick approach (phase < 0.5). Mouth gaussian is too sharp at stand z."""
    z = np.asarray(base_z, dtype=np.float32).reshape(-1)
    ph = np.asarray(phase, dtype=np.float32).reshape(-1)
    approach = (ph < 0.5).astype(np.float32)
    prox = np.exp(-(((z - float(target)) / float(std)) ** 2)).astype(np.float32)
    return (prox * approach).astype(np.float32)


def pick_return_pose(
    q: np.ndarray,
    home: np.ndarray,
    phase: np.ndarray,
    std: float = 0.15,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    ret = (np.asarray(phase, dtype=np.float32).reshape(-1) >= 0.5).astype(np.float32)
    err = q[:, LEG_JOINT_IDX] - home[LEG_JOINT_IDX]
    z = np.mean((err * err) / (float(std) ** 2), axis=1)
    return (np.exp(-z) * ret).astype(np.float32)


def kick_swing_reward(
    foot_height: np.ndarray,
    foot_xy_speed: np.ndarray,
    ep_t: np.ndarray,
    *,
    foot: str = "right",
    window_s: float = 1.2,
) -> np.ndarray:
    h = np.asarray(foot_height, dtype=np.float32).reshape(-1, 2)
    v = np.asarray(foot_xy_speed, dtype=np.float32).reshape(-1, 2)
    t = np.asarray(ep_t, dtype=np.float32).reshape(-1)
    idx = 0 if str(foot) == "left" else 1
    gate = (t < float(window_s)).astype(np.float32)
    return ((np.maximum(h[:, idx], 0.0) + v[:, idx]) * gate).astype(np.float32)


def cmd_speed(cmd13: np.ndarray) -> np.ndarray:
    c = np.asarray(cmd13, dtype=np.float32).reshape(-1, 13)
    lin = np.sqrt(c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1])
    return (lin + np.abs(c[:, 2])).astype(np.float32)


def pose_legs(
    q: np.ndarray,
    home: np.ndarray,
    speed: np.ndarray,
    walking_threshold: float = 0.01,
    std_standing: np.ndarray | None = None,
    std_walking: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian on leg joints vs HOME; tighter std when |cmd| < walking_threshold."""
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    speed = np.asarray(speed, dtype=np.float32).reshape(-1)
    std_s = _STD_STANDING_LEGS if std_standing is None else np.asarray(std_standing, dtype=np.float32)
    std_w = _STD_WALKING_LEGS if std_walking is None else np.asarray(std_walking, dtype=np.float32)
    stand = (speed < float(walking_threshold)).astype(np.float32)[:, None]
    std = stand * std_s[None, :] + (1.0 - stand) * std_w[None, :]
    err = q[:, LEG_JOINT_IDX] - home[LEG_JOINT_IDX]
    z = np.mean((err * err) / (std * std), axis=1)
    return np.exp(-z).astype(np.float32)


def head_pose_tracking(
    q: np.ndarray,
    home: np.ndarray,
    cmd13: np.ndarray,
    std: float = 0.5,
) -> np.ndarray:
    """Mean over 4 head joints of exp(-(err/std)^2). cmd head slots are deltas from HOME."""
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    cmd = np.asarray(cmd13, dtype=np.float32).reshape(-1, 13)
    actual = q[:, HEAD_JOINT_IDX] - home[HEAD_JOINT_IDX]
    err = actual - cmd[:, 3:7]
    per = np.exp(-((err / float(std)) ** 2))
    return np.mean(per, axis=1).astype(np.float32)


def action_rate_l2(action: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32)
    p = np.asarray(last_action, dtype=np.float32)
    d = a - p
    return np.sum(d * d, axis=-1).astype(np.float32)


def dof_pos_limits(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    below = np.maximum(lo - q, 0.0)
    above = np.maximum(q - hi, 0.0)
    return np.sum(below + above, axis=-1).astype(np.float32)


def body_ang_vel(gyro_xy: np.ndarray) -> np.ndarray:
    g = np.asarray(gyro_xy, dtype=np.float32).reshape(-1, 2)
    return np.sum(g * g, axis=-1).astype(np.float32)


def foot_slip(
    contact: np.ndarray,
    foot_xy_speed: np.ndarray,
    speed: np.ndarray,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float32).reshape(-1, 2)
    v = np.asarray(foot_xy_speed, dtype=np.float32).reshape(-1, 2)
    active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(np.float32)
    return (np.sum((v * v) * c, axis=1) * active).astype(np.float32)


def foot_clearance(
    height: np.ndarray,
    foot_xy_speed: np.ndarray,
    speed: np.ndarray,
    target: float = 0.02,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    h = np.asarray(height, dtype=np.float32).reshape(-1, 2)
    v = np.asarray(foot_xy_speed, dtype=np.float32).reshape(-1, 2)
    active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(np.float32)
    cost = np.sum(np.abs(h - float(target)) * v, axis=1)
    return (cost * active).astype(np.float32)


def air_time_reward(
    current_air_time: np.ndarray,
    cmd_speed_n: np.ndarray,
    tmin: float = 0.125,
    tmax: float = 0.3,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    """mjlab ``feet_air_time``: +1 per foot per step while current air time ∈ (tmin, tmax).

    Source: mjlab 1.3.0 ``mjlab/tasks/velocity/mdp/rewards.py::feet_air_time``.
    Not a touchdown pulse — that form capped at ~0.005/step and lost to action_rate.
    """
    air = np.asarray(current_air_time, dtype=np.float32)
    in_win = (air > float(tmin)) & (air < float(tmax))
    r = np.sum(in_win, axis=-1).astype(np.float32)
    spd = np.asarray(cmd_speed_n, dtype=np.float32).reshape(-1)
    r *= (spd > float(cmd_threshold)).astype(np.float32)
    return r


class AirTimeTracker:
    """Per-foot air/contact timers at control rate. Starts in contact (home stand)."""

    def __init__(self, num_envs: int, n_feet: int = 2) -> None:
        self.num_envs = int(num_envs)
        self.n_feet = int(n_feet)
        self.air_time = np.zeros((self.num_envs, self.n_feet), dtype=np.float32)
        self.contact_time = np.zeros((self.num_envs, self.n_feet), dtype=np.float32)
        self.in_contact = np.ones((self.num_envs, self.n_feet), dtype=bool)

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.air_time[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.in_contact[env_ids] = True

    def step(self, contact: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance timers. Returns (first_contact, last_air_time) for this step."""
        now = np.asarray(contact, dtype=bool).reshape(self.num_envs, self.n_feet)
        first = now & ~self.in_contact
        last_air = np.where(first, self.air_time, 0.0).astype(np.float32)
        dt = float(dt)
        air = self.air_time + dt
        ct = self.contact_time + dt
        self.air_time = np.where(now, 0.0, air).astype(np.float32)
        self.contact_time = np.where(now, ct, 0.0).astype(np.float32)
        self.in_contact = now
        return first, last_air


class HeadPoseBias:
    """1 s EMA of head-joint error; raw term is mean |EMA| (penalty, weight negative)."""

    def __init__(self, num_envs: int, n_joints: int = 4, tau_s: float = 1.0, dt: float = 0.02) -> None:
        self.ema = np.zeros((int(num_envs), int(n_joints)), dtype=np.float32)
        self.alpha = min(1.0, float(dt) / max(float(tau_s), 1e-6))

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.ema[env_ids] = 0.0

    def step(self, err: np.ndarray) -> np.ndarray:
        e = np.asarray(err, dtype=np.float32)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * e
        return np.mean(np.abs(self.ema), axis=-1).astype(np.float32)


class SwingPeakTracker:
    def __init__(self, num_envs: int, n_feet: int = 2) -> None:
        self.peak = np.zeros((int(num_envs), int(n_feet)), dtype=np.float32)

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.peak[env_ids] = 0.0

    def step(
        self,
        in_air: np.ndarray,
        height: np.ndarray,
        first_contact: np.ndarray,
        target: float,
        speed: np.ndarray,
        cmd_threshold: float,
    ) -> np.ndarray:
        air = np.asarray(in_air, dtype=bool)
        h = np.asarray(height, dtype=np.float32)
        self.peak = np.where(air, np.maximum(self.peak, h), self.peak).astype(np.float32)
        fc = np.asarray(first_contact, dtype=bool)
        tgt = max(float(target), 1e-8)
        err = self.peak / tgt - 1.0
        cost = np.sum((err * err) * fc.astype(np.float32), axis=1)
        active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(
            np.float32
        )
        self.peak = np.where(fc, 0.0, self.peak).astype(np.float32)
        return (cost * active).astype(np.float32)


@dataclass
class RewardInputs:
    q: np.ndarray
    gyro: np.ndarray
    grav: np.ndarray
    base_linvel_yaw: np.ndarray
    cmd13: np.ndarray
    action: np.ndarray
    last_action: np.ndarray
    contact: np.ndarray
    foot_height: np.ndarray
    foot_xy_speed: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    home: np.ndarray
    base_z: np.ndarray | None = None
    mouth_z: np.ndarray | None = None
    ep_t: np.ndarray | None = None


class RewardComputer:
    """Stateful batch rewards + episode sums for rsl_rl extras['log']."""

    def __init__(
        self,
        cfg: RewardConfig,
        num_envs: int,
        dt: float,
        home: np.ndarray,
        joint_lo: np.ndarray,
        joint_hi: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.home = np.asarray(home, dtype=np.float32).reshape(-1)
        self.joint_lo = np.asarray(joint_lo, dtype=np.float32).reshape(-1)
        self.joint_hi = np.asarray(joint_hi, dtype=np.float32).reshape(-1)
        self.air = AirTimeTracker(self.num_envs)
        self.bias = HeadPoseBias(
            self.num_envs, tau_s=cfg.head_pose_bias_tau_s, dt=self.dt
        )
        self.swing = SwingPeakTracker(self.num_envs)
        self.sit_q = sit_target_q(self.home)
        self.pitch_acc = np.zeros(self.num_envs, dtype=np.float32)
        self.pitch_max = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_sums = {n: np.zeros(self.num_envs, dtype=np.float32) for n in TERM_NAMES}

    def reset(self, env_ids: np.ndarray | list[int]) -> dict[str, float]:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        extras: dict[str, float] = {}
        if ids.size:
            for n in TERM_NAMES:
                extras[n] = float(np.mean(self.episode_sums[n][ids]))
                self.episode_sums[n][ids] = 0.0
            self.air.reset(ids)
            self.bias.reset(ids)
            self.swing.reset(ids)
            self.pitch_acc[ids] = 0.0
            self.pitch_max[ids] = 0.0
        return extras

    def zero_sums(self, env_ids: np.ndarray | list[int]) -> None:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if ids.size:
            for n in TERM_NAMES:
                self.episode_sums[n][ids] = 0.0
            self.air.reset(ids)
            self.bias.reset(ids)
            self.swing.reset(ids)
            self.pitch_acc[ids] = 0.0
            self.pitch_max[ids] = 0.0

    def compute(self, inp: RewardInputs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.cfg
        n = self.num_envs
        speed = cmd_speed(inp.cmd13)
        first, last_air = self.air.step(inp.contact, self.dt)
        in_air = ~np.asarray(inp.contact, dtype=bool)
        q = np.asarray(inp.q, dtype=np.float32).reshape(n, -1)
        home = self.home
        cmd = np.asarray(inp.cmd13, dtype=np.float32).reshape(n, 13)
        head_err = (q[:, HEAD_JOINT_IDX] - home[HEAD_JOINT_IDX]) - cmd[:, 3:7]

        raw: dict[str, np.ndarray] = {
            "track_lin_vel": track_lin_vel(
                cmd[:, :2], inp.base_linvel_yaw[:, :2], cfg.track_lin_vel_std2
            ),
            "track_ang_vel": track_ang_vel(cmd[:, 2], inp.gyro[:, 2], cfg.track_ang_vel_std2),
            "upright": upright(inp.grav, cfg.upright_std2),
            "air_time": air_time_reward(
                self.air.air_time,
                speed,
                tmin=cfg.air_time_min,
                tmax=cfg.air_time_max,
                cmd_threshold=cfg.air_time_cmd_threshold,
            ),
            "pose_legs": pose_legs(q, home, speed, cfg.pose_walking_threshold),
            "foot_clearance": foot_clearance(
                inp.foot_height,
                inp.foot_xy_speed,
                speed,
                target=cfg.foot_clearance_target,
                cmd_threshold=cfg.air_time_cmd_threshold,
            ),
            "foot_swing_height": self.swing.step(
                in_air,
                inp.foot_height,
                first,
                cfg.foot_swing_target,
                speed,
                cfg.air_time_cmd_threshold,
            ),
            "action_rate_l2": action_rate_l2(inp.action, inp.last_action),
            "foot_slip": foot_slip(
                inp.contact, inp.foot_xy_speed, speed, cfg.air_time_cmd_threshold
            ),
            "body_ang_vel": body_ang_vel(inp.gyro[:, :2]),
            "head_pose_tracking": head_pose_tracking(
                q, home, cmd, std=cfg.head_pose_tracking_std
            ),
            "head_pose_bias": self.bias.step(head_err),
            "dof_pos_limits": dof_pos_limits(q, self.joint_lo, self.joint_hi),
        }
        z_n = np.zeros(n, dtype=np.float32)
        phase = pick_phase_from_cmd(cmd) if (cfg.mouth_proximity or cfg.pick_return) else z_n
        mouth = (
            np.asarray(inp.mouth_z, dtype=np.float32).reshape(-1)
            if inp.mouth_z is not None
            else np.full(n, np.nan, dtype=np.float32)
        )
        base_z = (
            np.asarray(inp.base_z, dtype=np.float32).reshape(-1)
            if inp.base_z is not None
            else z_n
        )
        ep_t = (
            np.asarray(inp.ep_t, dtype=np.float32).reshape(-1)
            if inp.ep_t is not None
            else z_n
        )
        gyro = np.asarray(inp.gyro, dtype=np.float32).reshape(n, 3)
        self.pitch_acc = self.pitch_acc + gyro[:, 1] * self.dt
        cur = np.abs(self.pitch_acc)
        progress = np.maximum(cur - self.pitch_max, 0.0)
        cap = float(cfg.roulade_rate_cap) * self.dt
        progress = np.minimum(progress, cap)
        self.pitch_max = np.maximum(self.pitch_max, cur)
        land = (self.pitch_max >= float(cfg.roulade_complete_rad)).astype(np.float32) * raw[
            "upright"
        ]
        raw.update(
            {
                "posture_pose": posture_pose(
                    q, home, self.sit_q, cmd[:, 0], std=cfg.posture_pose_std
                )
                if cfg.posture_pose
                else z_n,
                "posture_height": posture_height(
                    base_z,
                    cmd[:, 0],
                    sit_z=cfg.sit_z,
                    stand_z=cfg.stand_z,
                    std=cfg.posture_height_std,
                )
                if cfg.posture_height
                else z_n,
                "mouth_proximity": mouth_proximity(mouth, phase, std=cfg.mouth_std)
                if cfg.mouth_proximity
                else z_n,
                "pick_return": pick_return_pose(q, home, phase, std=cfg.posture_pose_std)
                if cfg.pick_return
                else z_n,
                "approach_height": approach_height(
                    base_z, phase, target=cfg.approach_z, std=cfg.approach_z_std
                )
                if cfg.approach_height
                else z_n,
                "kick_swing": kick_swing_reward(
                    inp.foot_height,
                    inp.foot_xy_speed,
                    ep_t,
                    foot=cfg.kick_foot,
                    window_s=cfg.kick_window_s,
                )
                if cfg.kick_swing
                else z_n,
                "roulade_progress": progress if cfg.roulade_progress else z_n,
                "roulade_land": land if cfg.roulade_land else z_n,
            }
        )

        scale = self.dt if cfg.scale_by_dt else 1.0
        total = np.zeros(n, dtype=np.float32)
        weighted: dict[str, np.ndarray] = {}
        for name in TERM_NAMES:
            w = float(getattr(cfg, name))
            if w == 0.0:
                term = np.zeros(n, dtype=np.float32)
            else:
                term = (raw[name].astype(np.float32) * w * scale).astype(np.float32)
                term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
            weighted[name] = term
            total += term
            self.episode_sums[name] += term
        return total.astype(np.float32), weighted
