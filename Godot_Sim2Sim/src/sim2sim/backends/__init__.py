"""Physics backend interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class SimState:
    t: float
    q: np.ndarray
    qd: np.ndarray
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    base_linvel: np.ndarray  # Trunk inertial COM velocity in world coordinates.
    base_angvel_local: np.ndarray
    extra: dict = field(default_factory=dict)


class PhysicsBackend(Protocol):
    name: str
    dt: float
    nu: int

    def reset(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        keyframe: str | None = None,
        ctrl: np.ndarray | None = None,
    ) -> SimState: ...

    def step(self, ctrl: np.ndarray, n_substeps: int = 1) -> SimState: ...

    def close(self) -> None: ...
