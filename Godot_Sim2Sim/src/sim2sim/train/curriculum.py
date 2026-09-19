"""Iteration-indexed linear curriculum (microduck_rl velocity schedule, scaled).

YAML entries are ``[start_value, end_value, start_iter, end_iter]``. Interpolation
is linear in the iteration index and clamped outside that window.

Source: ``microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py``
(``standing_envs``, ``action_rate_weight``, ``head_pose_bias_weight``). Those
curricula are step-staged (``step = iter * 24``) out to iter 1500–2000. This
module linearizes them and, for Godot fine-tunes, typically reaches the final
values by iter 1000 of a 3000-iter budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sim2sim.train.commands import CommandConfig
from sim2sim.train.rewards import RewardConfig

TermSpec = tuple[float, float, int, int]


def interpolate(it: int, start: float, end: float, start_it: int, end_it: int) -> float:
    """Linear interpolation, clamped to ``[start, end]`` outside the iter window."""
    it = int(it)
    start_it = int(start_it)
    end_it = int(end_it)
    if end_it <= start_it:
        return float(end) if it >= start_it else float(start)
    if it <= start_it:
        return float(start)
    if it >= end_it:
        return float(end)
    t = (it - start_it) / (end_it - start_it)
    return float(start + t * (end - start))


def _parse_spec(raw: Any, name: str) -> TermSpec:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"curriculum.{name} must be [start, end, start_iter, end_iter], got {raw!r}")
    start, end, start_it, end_it = raw
    return (float(start), float(end), int(start_it), int(end_it))


@dataclass
class Curriculum:
    """Named linear terms. Unknown keys are ignored at apply time."""

    terms: dict[str, TermSpec]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Curriculum:
        if not raw:
            return cls({})
        terms: dict[str, TermSpec] = {}
        for name, spec in raw.items():
            if spec is None:
                continue
            terms[str(name)] = _parse_spec(spec, str(name))
        return cls(terms)

    def values_at(self, it: int) -> dict[str, float]:
        return {name: interpolate(it, *spec) for name, spec in self.terms.items()}


def apply_curriculum(
    reward_cfg: RewardConfig,
    command_cfg: CommandConfig,
    values: Mapping[str, float],
) -> None:
    """Write interpolated values onto the live reward / command configs."""
    if "standing_frac" in values:
        command_cfg.standing_frac = float(values["standing_frac"])
    for name, value in values.items():
        if name == "standing_frac":
            continue
        if hasattr(reward_cfg, name):
            setattr(reward_cfg, name, float(value))
