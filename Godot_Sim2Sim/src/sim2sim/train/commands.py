"""13-D command samplers matching sim2sim-play ``PlayBrain.command_13``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sim2sim.obs import command_13

VALID_MODES = ("twist", "zeros", "sit_flag", "pick_phase")


@dataclass
class CommandConfig:
    mode: str = "twist"
    resample_s: tuple[float, float] = (3.0, 8.0)
    vx: tuple[float, float] = (-0.4, 0.4)
    vy: tuple[float, float] = (-0.3, 0.3)
    wz: tuple[float, float] = (-1.0, 1.0)
    standing_frac: float = 0.25
    turn_in_place_frac: float = 0.15
    turn_wz: tuple[float, float] = (0.4, 1.0)
    sit_prob: float = 0.5
    pick_period: float = 4.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CommandConfig:
        if not raw:
            return cls()
        kwargs: dict[str, Any] = {}
        if "mode" in raw:
            mode = str(raw["mode"])
            if mode not in VALID_MODES:
                raise ValueError(f"commands.mode must be one of {VALID_MODES}, got {mode!r}")
            kwargs["mode"] = mode
        if "resample_s" in raw:
            lo, hi = raw["resample_s"]
            kwargs["resample_s"] = (float(lo), float(hi))
        for key in ("vx", "vy", "wz", "turn_wz"):
            if key in raw:
                lo, hi = raw[key]
                kwargs[key] = (float(lo), float(hi))
        if "standing_frac" in raw:
            kwargs["standing_frac"] = float(raw["standing_frac"])
        if "turn_in_place_frac" in raw:
            kwargs["turn_in_place_frac"] = float(raw["turn_in_place_frac"])
        if "sit_prob" in raw:
            kwargs["sit_prob"] = float(raw["sit_prob"])
        if "pick_period" in raw:
            kwargs["pick_period"] = float(raw["pick_period"])
        return cls(**kwargs)


class CommandSampler:
    """Per-env commands. Deterministic given ``rng``.

    Modes match play:
      * ``twist`` — piecewise-constant vx/vy/wz (walking / roller)
      * ``zeros`` — standing, kick, roulade, roller_crouch idle
      * ``sit_flag`` — cmd[0] ∈ {0=stand, 1=sit}, dwell ``resample_s``
      * ``pick_phase`` — cmd[0:2] = (cos 2πφ, sin 2πφ), φ += dt/period
    """

    def __init__(self, cfg: CommandConfig | Mapping[str, Any], num_envs: int, rng: np.random.Generator) -> None:
        self.cfg = cfg if isinstance(cfg, CommandConfig) else CommandConfig.from_dict(cfg)
        self.num_envs = int(num_envs)
        self.rng = rng
        self.cmd = np.zeros((self.num_envs, 13), dtype=np.float32)
        self.ttl = np.zeros(self.num_envs, dtype=np.float64)
        self.phase = np.zeros(self.num_envs, dtype=np.float64)
        self.reset(np.arange(self.num_envs))

    def reset(self, env_ids: np.ndarray | list[int]) -> None:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        mode = self.cfg.mode
        for i in ids:
            if mode == "pick_phase":
                self.phase[i] = float(self.rng.uniform(0.0, 1.0))
                self._write_pick(i)
                self.ttl[i] = 1e9
            else:
                self.cmd[i] = self._sample_one()
                self.ttl[i] = self._sample_ttl() if mode != "zeros" else 1e9

    def step(self, dt: float) -> np.ndarray:
        dt = float(dt)
        if self.cfg.mode == "pick_phase":
            period = max(float(self.cfg.pick_period), 1e-6)
            self.phase = (self.phase + dt / period) % 1.0
            ang = 2.0 * np.pi * self.phase
            self.cmd[:, 0] = np.cos(ang).astype(np.float32)
            self.cmd[:, 1] = np.sin(ang).astype(np.float32)
            self.cmd[:, 2:] = 0.0
            return self.cmd
        if self.cfg.mode == "zeros":
            return self.cmd
        self.ttl -= dt
        due = np.nonzero(self.ttl <= 0.0)[0]
        if due.size:
            self.reset(due)
        return self.cmd

    def _write_pick(self, i: int) -> None:
        ang = 2.0 * np.pi * float(self.phase[i])
        self.cmd[i] = 0.0
        self.cmd[i, 0] = np.cos(ang)
        self.cmd[i, 1] = np.sin(ang)

    def _sample_ttl(self) -> float:
        lo, hi = self.cfg.resample_s
        return float(self.rng.uniform(lo, hi))

    def _sample_one(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.mode == "zeros":
            return command_13(np.zeros(3, dtype=np.float32))
        if cfg.mode == "sit_flag":
            flag = 1.0 if float(self.rng.random()) < cfg.sit_prob else 0.0
            return command_13(np.array([flag, 0.0, 0.0], dtype=np.float32))
        vx = float(self.rng.uniform(*cfg.vx))
        vy = float(self.rng.uniform(*cfg.vy))
        wz = float(self.rng.uniform(*cfg.wz))
        standing = float(self.rng.random()) < cfg.standing_frac
        turn = float(self.rng.random()) < cfg.turn_in_place_frac
        if standing:
            vx = vy = wz = 0.0
        if turn:
            vx = 0.0
            vy = 0.0
            mag = float(self.rng.uniform(*cfg.turn_wz))
            sign = -1.0 if float(self.rng.random()) < 0.5 else 1.0
            wz = sign * mag
        return command_13(np.array([vx, vy, wz], dtype=np.float32))
