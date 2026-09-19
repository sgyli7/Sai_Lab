"""Independent A/B walking evaluation on Godot/Jolt (not the training reward).

`game_seq` is shaped through `PlayBrain.tick` with the **same** default
`TwistLimits()` for policy A and policy B. Sidecar `sim2sim.twist_limits` are
ignored here so a looser/tighter command envelope cannot bias the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.fall import fallen
from sim2sim.godot_proc import sim2sim_root
from sim2sim.obs import DEFAULT_HOME, build_obs, command_13
from sim2sim.play import capture_home_poses
from sim2sim.play_input import PlayBrain, TwistLimits
from sim2sim.policy import PolicyBundle
from sim2sim.runner import load_robot_cfg

# Shared by A and B. Do not read per-policy sidecar limits for game_seq.
EVAL_TWIST_LIMITS = TwistLimits()

LEFT_KNEE = 3
MAX_WORKERS = 8
NUDGE_SPEED = 1.0
NUDGE_PERIOD_S = 2.0
SETTLE_S = 1.0
MA_WINDOW_S = 1.0


@dataclass(frozen=True)
class Condition:
    name: str
    vel: tuple[float, float, float] | None
    primary: str
    kind: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("idle", (0.0, 0.0, 0.0), "yaw_drift_deg", "idle"),
    Condition("walk_015", (0.15, 0.0, 0.0), "vel_err_1s_rmse", "move"),
    Condition("walk_025", (0.25, 0.0, 0.0), "vel_err_1s_rmse", "move"),
    Condition("run_040", (0.40, 0.0, 0.0), "vel_err_1s_rmse", "move"),
    Condition("back_020", (-0.2, 0.0, 0.0), "vel_err_1s_rmse", "move"),
    Condition("strafe_l", (0.0, 0.2, 0.0), "vel_err_1s_rmse", "move"),
    Condition("strafe_r", (0.0, -0.2, 0.0), "vel_err_1s_rmse", "move"),
    Condition("turn_l", (0.0, 0.0, 0.8), "yaw_err_1s_rmse", "turn"),
    Condition("turn_r", (0.0, 0.0, -0.8), "yaw_err_1s_rmse", "turn"),
    Condition("walk_turn", (0.2, 0.0, 0.5), "vel_err_1s_rmse", "move"),
    Condition("walk_push", (0.25, 0.0, 0.0), "fell", "push"),
    Condition("game_seq", None, "vel_err_1s_rmse", "game"),
)

_TABLE_METRICS = (
    "vel_err_1s_rmse",
    "yaw_err_1s_rmse",
    "mean_vel_err",
    "mean_yaw_rate_err",
    "lin_vel_rmse",
    "yaw_rate_rmse",
    "yaw_drift_deg",
    "distance",
    "fell",
    "survival_s",
    "mean_abs_daction",
    "cadence_hz_left_knee",
    "mean_trunk_z",
)

_VEL_ERR_METRICS = frozenset({"vel_err_1s_rmse", "mean_vel_err", "lin_vel_rmse"})
_YAW_ERR_METRICS = frozenset({"yaw_err_1s_rmse", "mean_yaw_rate_err", "yaw_rate_rmse"})

_LOWER_BETTER = {
    "vel_err_1s_rmse",
    "yaw_err_1s_rmse",
    "mean_vel_err",
    "mean_yaw_rate_err",
    "lin_vel_rmse",
    "yaw_rate_rmse",
    "yaw_drift_deg",
    "fell",
    "mean_abs_daction",
}
_HIGHER_BETTER = {"survival_s", "distance"}


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(a, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(b, dtype=np.float64).reshape(4)
    q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / n


def rotate_poses_yaw(poses: list[dict], yaw: float) -> list[dict]:
    """Rotate each body `pos` about world z and left-multiply `quat` (wxyz) by the yaw quat."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    qz = yaw_quat_wxyz(yaw)
    out: list[dict] = []
    for raw in poses:
        item = dict(raw)
        pos = np.asarray(raw["pos"], dtype=np.float64).reshape(3).copy()
        item["pos"] = [c * pos[0] - s * pos[1], s * pos[0] + c * pos[1], float(pos[2])]
        item["quat"] = _quat_mul_wxyz(qz, np.asarray(raw["quat"], dtype=np.float64)).tolist()
        if "linvel" in raw:
            v = np.asarray(raw["linvel"], dtype=np.float64).reshape(3).copy()
            item["linvel"] = [c * v[0] - s * v[1], s * v[0] + c * v[1], float(v[2])]
        if "angvel" in raw:
            w = np.asarray(raw["angvel"], dtype=np.float64).reshape(3).copy()
            item["angvel"] = [c * w[0] - s * w[1], s * w[0] + c * w[1], float(w[2])]
        out.append(item)
    return out


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _fft_cadence(q: np.ndarray, dt: float) -> float:
    if len(q) < 16:
        return 0.0
    x = q - q.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), dt)
    spec[0] = 0
    i = int(np.argmax(spec))
    return float(freqs[i])


def _settle_slice(n: int, dt: float) -> slice:
    skip = int(round(SETTLE_S / float(dt))) if dt > 0 else 0
    if skip >= n:
        return slice(0, n)
    return slice(skip, n)


def _full_window_ma(x: np.ndarray, win: int) -> np.ndarray:
    """Causal moving average; returns only samples with a full window, or the mean if shorter."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    win = max(1, int(win))
    if n == 0:
        return x
    if n < win:
        return np.mean(x, axis=0, keepdims=True)
    if x.ndim == 1:
        c = np.cumsum(x)
        prev = np.concatenate([[0.0], c[: n - win]])
        return (c[win - 1 :] - prev) / win
    c = np.cumsum(x, axis=0)
    prev = np.concatenate([np.zeros((1, x.shape[1]), dtype=np.float64), c[: n - win]], axis=0)
    return (c[win - 1 :] - prev) / win


def _empty_metrics(*, fell: bool, survival_s: float) -> dict[str, Any]:
    return {
        "lin_vel_rmse": float("nan"),
        "yaw_rate_rmse": float("nan"),
        "mean_vel_err": float("nan"),
        "mean_yaw_rate_err": float("nan"),
        "vel_err_1s_rmse": float("nan"),
        "yaw_err_1s_rmse": float("nan"),
        "mean_vx": float("nan"),
        "mean_vy": float("nan"),
        "mean_wz": float("nan"),
        "yaw_drift_deg": 0.0,
        "distance": 0.0,
        "fell": bool(fell),
        "survival_s": float(survival_s),
        "mean_abs_daction": 0.0,
        "cadence_hz_left_knee": 0.0,
        "mean_trunk_z": float("nan"),
        "steps": 0,
    }


def episode_metrics(
    *,
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    linvel: np.ndarray,
    angvel: np.ndarray,
    q: np.ndarray,
    action: np.ndarray,
    cmd_xyw: np.ndarray,
    dt: float,
    fell: bool,
    survival_s: float,
) -> dict[str, Any]:
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    linvel = np.asarray(linvel, dtype=np.float64)
    angvel = np.asarray(angvel, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    cmd_xyw = np.asarray(cmd_xyw, dtype=np.float64)
    n = int(len(pos))
    if n == 0:
        return _empty_metrics(fell=fell, survival_s=survival_s)
    yaws = np.array([yaw_from_quat_wxyz(qq) for qq in quat], dtype=np.float64)
    c, s = np.cos(yaws), np.sin(yaws)
    vx_b = c * linvel[:, 0] + s * linvel[:, 1]
    vy_b = -s * linvel[:, 0] + c * linvel[:, 1]
    wz = angvel[:, 2]
    lin_vel_rmse = float(np.sqrt(np.mean((vx_b - cmd_xyw[:, 0]) ** 2 + (vy_b - cmd_xyw[:, 1]) ** 2)))
    yaw_rate_rmse = float(np.sqrt(np.mean((wz - cmd_xyw[:, 2]) ** 2)))
    sl = _settle_slice(n, dt)
    mean_dvx = float(np.mean(vx_b[sl] - cmd_xyw[sl, 0]))
    mean_dvy = float(np.mean(vy_b[sl] - cmd_xyw[sl, 1]))
    mean_vel_err = float(np.hypot(mean_dvx, mean_dvy))
    mean_yaw_rate_err = float(abs(np.mean(wz[sl] - cmd_xyw[sl, 2])))
    mean_vx = float(np.mean(vx_b[sl]))
    mean_vy = float(np.mean(vy_b[sl]))
    mean_wz = float(np.mean(wz[sl]))
    win = max(1, int(round(MA_WINDOW_S / float(dt)))) if dt > 0 else 1
    vx_ma = _full_window_ma(vx_b, win)
    vy_ma = _full_window_ma(vy_b, win)
    wz_ma = _full_window_ma(wz, win)
    if n < win:
        cmd_ma = cmd_xyw.mean(axis=0)
        vel_err_1s_rmse = float(np.hypot(vx_ma.reshape(-1)[0] - cmd_ma[0], vy_ma.reshape(-1)[0] - cmd_ma[1]))
        yaw_err_1s_rmse = float(abs(wz_ma.reshape(-1)[0] - cmd_ma[2]))
    else:
        cmd_t = cmd_xyw[win - 1 :]
        vel_err_1s_rmse = float(
            np.sqrt(np.mean((vx_ma - cmd_t[:, 0]) ** 2 + (vy_ma - cmd_t[:, 1]) ** 2))
        )
        yaw_err_1s_rmse = float(np.sqrt(np.mean((wz_ma - cmd_t[:, 2]) ** 2)))
    yaw_unwrapped = np.unwrap(yaws)
    yaw_drift_deg = float(np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0]))
    distance = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    if n >= 2:
        mean_abs_daction = float(np.mean(np.abs(np.diff(action, axis=0))))
    else:
        mean_abs_daction = 0.0
    cadence = _fft_cadence(q[:, LEFT_KNEE], dt) if q.shape[1] > LEFT_KNEE else 0.0
    return {
        "lin_vel_rmse": lin_vel_rmse,
        "yaw_rate_rmse": yaw_rate_rmse,
        "mean_vel_err": mean_vel_err,
        "mean_yaw_rate_err": mean_yaw_rate_err,
        "vel_err_1s_rmse": vel_err_1s_rmse,
        "yaw_err_1s_rmse": yaw_err_1s_rmse,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "mean_wz": mean_wz,
        "yaw_drift_deg": yaw_drift_deg,
        "distance": distance,
        "fell": bool(fell),
        "survival_s": float(survival_s),
        "mean_abs_daction": mean_abs_daction,
        "cadence_hz_left_knee": cadence,
        "mean_trunk_z": float(np.mean(pos[:, 2])),
        "steps": n,
    }


def held_for_game(t: float) -> set[str]:
    """W 3 s → W+A 2 s → release 1 s → S 2 s → Q 2 s (then hold last)."""
    if t < 3.0:
        return {"fwd"}
    if t < 5.0:
        return {"fwd", "left"}
    if t < 6.0:
        return set()
    if t < 8.0:
        return {"back"}
    return {"strafe_l"}


def _mix_seed(*parts: int) -> int:
    x = 0xC0FFEE
    for p in parts:
        x = (x * 1000003 + int(p)) & 0x7FFFFFFF
    return int(x)


def _primary_value(ep: dict, metric: str) -> float:
    if metric == "fell":
        return 1.0 if ep["fell"] else 0.0
    v = float(ep[metric])
    if metric == "yaw_drift_deg":
        return abs(v)
    return v


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def bootstrap_mean_ci(
    diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    if diffs.size == 0:
        return float("nan"), float("nan")
    if diffs.size == 1:
        v = float(diffs[0])
        return v, v
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _fmt_ms(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    return f"{mean:.4f}±{std:.4f}"


class _GodotWorker:
    def __init__(self, cfg: dict, path_a: Path, path_b: Path, label_a: str, label_b: str) -> None:
        spec = Path(cfg["godot_spec"])
        dt = float(cfg.get("timestep", 0.005))
        self.backend = GodotBackend(
            spec,
            timestep=dt,
            headless=True,
            base_body=cfg.get("base_body", "trunk_base"),
            current_limit_a=cfg.get("current_limit_a", 1.75),
            recv_timeout=60.0,
        )
        self.policies = {label_a: PolicyBundle(path_a)}
        if Path(path_b).resolve() == Path(path_a).resolve():
            self.policies[label_b] = self.policies[label_a]
        else:
            self.policies[label_b] = PolicyBundle(path_b)
        home_len = int(np.asarray(cfg.get("home", DEFAULT_HOME)).size)
        for pol in self.policies.values():
            pol.check_dims(home_len)

    def close(self) -> None:
        self.backend.close()


def _run_episode(
    worker: _GodotWorker,
    *,
    label: str,
    cond: Condition,
    seed: int,
    cond_idx: int,
    seconds: float,
    home: np.ndarray,
    home_poses: list[dict],
    scale: float,
    decimation: int,
    dt: float,
) -> dict[str, Any]:
    dt_ctrl = decimation * dt
    n_steps = int(round(seconds / dt_ctrl))
    yaw = float(np.random.default_rng(_mix_seed(cond_idx, seed, 0)).uniform(-math.pi, math.pi))
    poses = rotate_poses_yaw(home_poses, yaw)
    policy = worker.policies[label]
    policy.reset_context()
    st = worker.backend.reset(ctrl=home, bodies=poses)
    last_action = np.zeros(int(home.size), dtype=np.float32)
    brain: PlayBrain | None = None
    if cond.kind == "game":
        brain = PlayBrain(
            has_walking=True,
            has_standing=True,
            has_sitstand=False,
            has_pick=False,
            has_kick_left=False,
            has_kick_right=False,
            has_roulade=False,
            lim=EVAL_TWIST_LIMITS,
        )
    nudge_rng = np.random.default_rng(_mix_seed(cond_idx, seed, 1))
    nudge_at = set()
    if cond.kind == "push":
        t_nudge = NUDGE_PERIOD_S
        while t_nudge <= seconds + 1e-9:
            nudge_at.add(int(round(t_nudge / dt_ctrl)))
            t_nudge += NUDGE_PERIOD_S

    t_log: list[float] = []
    pos_log: list[np.ndarray] = []
    quat_log: list[np.ndarray] = []
    lin_log: list[np.ndarray] = []
    ang_log: list[np.ndarray] = []
    q_log: list[np.ndarray] = []
    act_log: list[np.ndarray] = []
    cmd_log: list[np.ndarray] = []
    did_fall = False
    survival = seconds

    for i in range(n_steps):
        t_cmd = i * dt_ctrl
        if brain is not None:
            out = brain.tick(held_for_game(t_cmd), [], dt_ctrl)
            cmd = np.asarray(out.command, dtype=np.float32)
        else:
            assert cond.vel is not None
            cmd = command_13(np.asarray(cond.vel, dtype=np.float32))
        obs = build_obs(st, last_action, cmd, home=home)
        action = policy.infer(obs)
        last_action = action.astype(np.float32, copy=True)
        ctrl = home + last_action * scale
        if i in nudge_at:
            theta = float(nudge_rng.uniform(0.0, 2.0 * math.pi))
            worker.backend.nudge(
                np.array([math.cos(theta) * NUDGE_SPEED, math.sin(theta) * NUDGE_SPEED, 0.0], dtype=np.float64)
            )
        st = worker.backend.step(ctrl, n_substeps=decimation, report="lite")
        t_log.append(float(st.t))
        pos_log.append(np.asarray(st.base_pos, dtype=np.float64).copy())
        quat_log.append(np.asarray(st.base_quat_wxyz, dtype=np.float64).copy())
        lin_log.append(np.asarray(st.base_linvel, dtype=np.float64).copy())
        ang_log.append(np.asarray(st.base_angvel_local, dtype=np.float64).copy())
        q_log.append(np.asarray(st.q, dtype=np.float64).copy())
        act_log.append(np.asarray(action, dtype=np.float64).copy())
        cmd_log.append(np.asarray(cmd[:3], dtype=np.float64).copy())
        if fallen(st.base_quat_wxyz, st.base_pos):
            did_fall = True
            survival = (i + 1) * dt_ctrl
            break

    mets = episode_metrics(
        t=np.asarray(t_log),
        pos=np.asarray(pos_log).reshape(len(pos_log), 3) if pos_log else np.zeros((0, 3)),
        quat=np.asarray(quat_log).reshape(len(quat_log), 4) if quat_log else np.zeros((0, 4)),
        linvel=np.asarray(lin_log).reshape(len(lin_log), 3) if lin_log else np.zeros((0, 3)),
        angvel=np.asarray(ang_log).reshape(len(ang_log), 3) if ang_log else np.zeros((0, 3)),
        q=np.asarray(q_log) if q_log else np.zeros((0, int(home.size))),
        action=np.asarray(act_log) if act_log else np.zeros((0, int(home.size))),
        cmd_xyw=np.asarray(cmd_log) if cmd_log else np.zeros((0, 3)),
        dt=dt_ctrl,
        fell=did_fall,
        survival_s=survival,
    )
    row = {
        "condition": cond.name,
        "seed": int(seed),
        "label": label,
        "yaw0": yaw,
        **mets,
    }
    return row


def _b_better(metric: str, a: float, b: float) -> bool | None:
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if metric == "fell":
        if abs(b - a) <= 1e-12:
            return None
        return b < a
    scale = max(abs(a), abs(b), 1e-6)
    if metric == "yaw_drift_deg":
        eps = max(0.25, 0.02 * scale)
    elif metric in _VEL_ERR_METRICS:
        eps = max(0.01, 0.05 * scale)
    elif metric in _YAW_ERR_METRICS:
        eps = max(0.02, 0.05 * scale)
    elif metric == "mean_abs_daction":
        eps = max(1e-4, 0.02 * scale)
    else:
        eps = max(1e-3, 0.02 * scale)
    if abs(b - a) <= eps:
        return None
    if metric in _LOWER_BETTER:
        return b < a
    if metric in _HIGHER_BETTER:
        return b > a
    return b < a


def _summarize(episodes: list[dict], label_a: str, label_b: str, seeds: list[int]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict]] = {}
    for ep in episodes:
        by.setdefault((ep["condition"], ep["label"]), []).append(ep)
    per_cond: dict[str, Any] = {}
    n_better = 0
    n_worse = 0
    n_tie = 0
    for cond in CONDITIONS:
        a_map = {e["seed"]: e for e in by.get((cond.name, label_a), [])}
        b_map = {e["seed"]: e for e in by.get((cond.name, label_b), [])}
        seed_keys = [s for s in seeds if s in a_map and s in b_map]
        a_eps = [a_map[s] for s in seed_keys]
        b_eps = [b_map[s] for s in seed_keys]
        metrics_block: dict[str, Any] = {}
        for metric in (*_TABLE_METRICS, "mean_vx", "mean_vy", "mean_wz"):
            av = [_primary_value(e, metric) if metric in ("fell", "yaw_drift_deg") else float(e[metric]) for e in a_eps]
            bv = [_primary_value(e, metric) if metric in ("fell", "yaw_drift_deg") else float(e[metric]) for e in b_eps]
            # yaw_drift in the table is signed mean; winner uses abs via _primary_value only for primary.
            if metric == "yaw_drift_deg":
                av = [float(e["yaw_drift_deg"]) for e in a_eps]
                bv = [float(e["yaw_drift_deg"]) for e in b_eps]
            if metric == "fell":
                av = [1.0 if e["fell"] else 0.0 for e in a_eps]
                bv = [1.0 if e["fell"] else 0.0 for e in b_eps]
            ma, sa = _mean_std(av)
            mb, sb = _mean_std(bv)
            paired = list(zip(a_eps, b_eps, strict=True))
            wins = 0
            for ea, eb in paired:
                pa = _primary_value(ea, metric)
                pb = _primary_value(eb, metric)
                better = _b_better(metric, pa, pb)
                if better is True:
                    wins += 1
            metrics_block[metric] = {
                "a_mean": ma,
                "a_std": sa,
                "b_mean": mb,
                "b_std": sb,
                "delta": mb - ma,
                "wins_b": wins,
                "n": len(paired),
            }
        prim = cond.primary
        a_p = np.array([_primary_value(e, prim) for e in a_eps], dtype=np.float64)
        b_p = np.array([_primary_value(e, prim) for e in b_eps], dtype=np.float64)
        diffs = b_p - a_p
        ci_lo, ci_hi = bootstrap_mean_ci(diffs, seed=_mix_seed(list(c.name for c in CONDITIONS).index(cond.name), 99))
        mean_d = float(np.mean(diffs)) if diffs.size else float("nan")
        winner = _b_better(prim, float(np.mean(a_p)) if a_p.size else math.nan, float(np.mean(b_p)) if b_p.size else math.nan)
        if winner is True:
            n_better += 1
            tag = "B"
        elif winner is False:
            n_worse += 1
            tag = "A"
        else:
            n_tie += 1
            tag = "tie"
        per_cond[cond.name] = {
            "primary": prim,
            "kind": cond.kind,
            "metrics": metrics_block,
            "paired_delta_mean": mean_d,
            "paired_delta_ci95": [ci_lo, ci_hi],
            "winner": tag,
        }
    push = per_cond.get("walk_push", {})
    fall_a = push.get("metrics", {}).get("fell", {}).get("a_mean", 0.0)
    fall_b = push.get("metrics", {}).get("fell", {}).get("b_mean", 0.0)
    fall_ok = float(fall_b) <= float(fall_a) + 1e-12
    n_cond = len(CONDITIONS)
    if n_better > n_cond / 2.0 and fall_ok:
        verdict = "improved"
    elif n_worse > n_cond / 2.0:
        verdict = "regressed"
    else:
        verdict = "mixed"
    return {
        "per_condition": per_cond,
        "n_better": n_better,
        "n_worse": n_worse,
        "n_tie": n_tie,
        "fall_rate_a": fall_a,
        "fall_rate_b": fall_b,
        "fall_ok": fall_ok,
        "verdict": verdict,
    }


def write_report(
    *,
    out_dir: Path,
    title: str,
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    summary: dict[str, Any],
    seconds: float,
    seeds: list[int],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- A (`{label_a}`): `{path_a}`",
        f"- B (`{label_b}`): `{path_b}`",
        f"- seeds: {seeds}  seconds: {seconds}",
        "- `game_seq` uses the **same** default `TwistLimits()` for A and B "
        "(not each ONNX sidecar). Otherwise a different command ramp would confound the comparison.",
        "- Idle / constant-twist episodes query the walking ONNX with a fixed command "
        "(no stand-policy switch except inside `PlayBrain` for `game_seq`).",
        "- Tracking primary is `vel_err_1s_rmse` (1 s moving-average body vx,vy vs per-step cmd); "
        "yaw conditions use `yaw_err_1s_rmse`. Instantaneous `lin_vel_rmse` stays secondary "
        "(gait oscillation dominates it).",
        "- Winner / VERDICT ignore tiny gaps (vel 0.01 m/s or 5%, yaw-rate 0.02 rad/s or 5%, "
        "yaw drift 0.25°, exact fall rate) so Jolt episode noise is not counted as an improvement.",
        "",
        "## Per condition",
        "",
        "| condition | metric | A mean±std | B mean±std | Δ(B−A) | wins(B better)/n |",
        "|---|---|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        block = summary["per_condition"][cond.name]
        for metric in _TABLE_METRICS:
            m = block["metrics"][metric]
            n = int(m["n"])
            lines.append(
                f"| {cond.name} | {metric} | {_fmt_ms(m['a_mean'], m['a_std'])} | "
                f"{_fmt_ms(m['b_mean'], m['b_std'])} | {m['delta']:+.4f} | {m['wins_b']}/{n} |"
            )
        prim = block["primary"]
        ci = block["paired_delta_ci95"]
        lines.append(
            f"| {cond.name} | **primary={prim}** winner={block['winner']} |  |  | "
            f"{block['paired_delta_mean']:+.4f} (95% CI {ci[0]:+.4f},{ci[1]:+.4f}) |  |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- B better (primary) on **{summary['n_better']}** / {len(CONDITIONS)} conditions; "
        f"A better on {summary['n_worse']}; ties {summary['n_tie']}.",
        f"- walk_push fall rate A={summary['fall_rate_a']:.3f} B={summary['fall_rate_b']:.3f} "
        f"(not worse: {summary['fall_ok']}).",
        f"- Rule: majority of conditions better **and** fall rate not worse → improved.",
        "",
        f"VERDICT: {summary['verdict']}",
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "report.md").write_text(text)
    return text


def _print_compact(summary: dict[str, Any], label_a: str, label_b: str) -> None:
    print(f"{'condition':<12} {'metric':<18} {label_a:>14} {label_b:>14} {'Δ(B−A)':>10} {'wins':>8}")
    for cond in CONDITIONS:
        block = summary["per_condition"][cond.name]
        m = block["metrics"][cond.primary]
        print(
            f"{cond.name:<12} {cond.primary:<18} "
            f"{_fmt_ms(m['a_mean'], m['a_std']):>14} {_fmt_ms(m['b_mean'], m['b_std']):>14} "
            f"{m['delta']:+10.4f} {m['wins_b']}/{m['n']}"
        )
    print(f"VERDICT: {summary['verdict']}")
    print(f"{label_a} baseline (post-{SETTLE_S:.0f}s settle mean body vel / smoother errors):")
    print(
        f"  {'condition':<12} {'mean_vx':>8} {'mean_vy':>8} {'mean_wz':>8} "
        f"{'mean_vel_err':>13} {'mean_yaw_err':>13} {'vel_1s':>8} {'yaw_1s':>8}"
    )
    for cond in CONDITIONS:
        mets = summary["per_condition"][cond.name]["metrics"]
        print(
            f"  {cond.name:<12} "
            f"{mets.get('mean_vx', {}).get('a_mean', float('nan')):+8.4f} "
            f"{mets.get('mean_vy', {}).get('a_mean', float('nan')):+8.4f} "
            f"{mets.get('mean_wz', {}).get('a_mean', float('nan')):+8.4f} "
            f"{mets.get('mean_vel_err', {}).get('a_mean', float('nan')):13.4f} "
            f"{mets.get('mean_yaw_rate_err', {}).get('a_mean', float('nan')):13.4f} "
            f"{mets.get('vel_err_1s_rmse', {}).get('a_mean', float('nan')):8.4f} "
            f"{mets.get('yaw_err_1s_rmse', {}).get('a_mean', float('nan')):8.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim2sim-eval-walk")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--label-a", default="alpha")
    p.add_argument("--label-b", default="walk_godot")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=sim2sim_root() / "results/walk_godot_eval")
    p.add_argument("--robot", type=Path, default=sim2sim_root() / "robots/microduck.json")
    args = p.parse_args(argv)

    n_workers = max(1, min(int(args.workers), MAX_WORKERS))
    seeds = list(range(int(args.seeds)))
    cfg = load_robot_cfg(args.robot)
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    scale = float(cfg.get("action_scale", 1.0))
    decimation = int(cfg.get("decimation", 4))
    dt = float(cfg.get("timestep", 0.005))
    print("capturing home poses (MuJoCo)…", flush=True)
    home_poses = capture_home_poses(cfg)

    jobs: list[tuple[str, Condition, int, int]] = []
    for ci, cond in enumerate(CONDITIONS):
        for seed in seeds:
            jobs.append((args.label_a, cond, seed, ci))
            jobs.append((args.label_b, cond, seed, ci))
    n_workers = max(1, min(n_workers, len(jobs)))
    print(f"starting {n_workers} Godot worker(s), {len(jobs)} episodes", flush=True)
    workers = [
        _GodotWorker(cfg, args.a, args.b, args.label_a, args.label_b) for _ in range(n_workers)
    ]
    pool: queue.Queue[_GodotWorker] = queue.Queue()
    for w in workers:
        pool.put(w)

    def _job(item: tuple[str, Condition, int, int]) -> dict[str, Any]:
        label, cond, seed, ci = item
        w = pool.get()
        try:
            return _run_episode(
                w,
                label=label,
                cond=cond,
                seed=seed,
                cond_idx=ci,
                seconds=float(args.seconds),
                home=home,
                home_poses=home_poses,
                scale=scale,
                decimation=decimation,
                dt=dt,
            )
        finally:
            pool.put(w)

    episodes: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_job, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                j = futs[fut]
                row = fut.result()
                episodes.append(row)
                done += 1
                print(
                    f"eval {done}/{len(jobs)} {j[1].name} seed={j[2]} label={j[0]} "
                    f"fell={row['fell']} surv={row['survival_s']:.2f}",
                    flush=True,
                )
    finally:
        for w in workers:
            try:
                w.close()
            except Exception:
                pass

    summary = _summarize(episodes, args.label_a, args.label_b, seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "a": str(args.a),
        "b": str(args.b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "seeds": seeds,
        "seconds": float(args.seconds),
        "twist_limits_note": (
            "game_seq uses default TwistLimits() for both policies; sidecar limits are ignored."
        ),
        "eval_twist_limits": asdict(EVAL_TWIST_LIMITS),
        "episodes": episodes,
        "summary": summary,
        "verdict": summary["verdict"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    report = write_report(
        out_dir=out_dir,
        title="Walk A/B eval (Godot/Jolt)",
        path_a=args.a,
        path_b=args.b,
        label_a=args.label_a,
        label_b=args.label_b,
        summary=summary,
        seconds=float(args.seconds),
        seeds=seeds,
    )
    _print_compact(summary, args.label_a, args.label_b)
    print(f"wrote {out_dir / 'report.md'}", flush=True)
    del report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
