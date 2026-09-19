"""ONNX policy wrapper. I/O names match duck-control / infer_policy (obs → actions)."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from sim2sim.play_input import TwistLimits


class PolicyShapeError(ValueError):
    """ONNX graph shape missing/dynamic, or obs/action rank disagrees with home."""


class PolicyNumericError(ValueError):
    """obs or action contained NaN/inf."""


def expected_obs_dim(home_len: int) -> int:
    """len(build_obs(...)): gyro3 + grav3 + qrel + qd + last_action + cmd13."""
    return 3 + 3 + 2 * int(home_len) + int(home_len) + 13


def _static_last_dim(shape: Any, what: str) -> int:
    if not shape:
        raise PolicyShapeError(f"{what} shape is empty/unknown: {shape!r}")
    last = shape[-1]
    if last is None or isinstance(last, str):
        raise PolicyShapeError(f"{what} last dim is dynamic/unknown: {shape!r}")
    try:
        n = int(last)
    except (TypeError, ValueError) as e:
        raise PolicyShapeError(f"{what} last dim is dynamic/unknown: {shape!r}") from e
    if n <= 0:
        raise PolicyShapeError(f"{what} last dim is dynamic/unknown: {shape!r}")
    return n


def _sidecar_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(onnx_path.stem + ".manifest.json")


def _nested_get(d: dict[str, Any], dotted: str) -> Any:
    if dotted in d:
        return d[dotted]
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _twist_limits_from(raw: Any) -> TwistLimits:
    if not isinstance(raw, dict):
        return TwistLimits()
    allowed = {f.name for f in fields(TwistLimits)}
    kwargs: dict[str, float] = {}
    for k, v in raw.items():
        if k in allowed and v is not None:
            kwargs[k] = float(v)
    return TwistLimits(**kwargs)


def _load_sidecar(onnx_path: Path) -> dict[str, Any] | None:
    side = _sidecar_path(onnx_path)
    if not side.is_file():
        return None
    data = json.loads(side.read_text())
    if not isinstance(data, dict):
        raise PolicyShapeError(f"sidecar is not a JSON object: {side}")
    return data


class PolicyBundle:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        ishape = self.sess.get_inputs()[0].shape
        oshape = self.sess.get_outputs()[0].shape
        self.obs_dim = _static_last_dim(ishape, "obs")
        self.act_dim = _static_last_dim(oshape, "action")
        meta = dict(self.sess.get_modelmeta().custom_metadata_map or {})
        from sim2sim.policy_time import time_input_seconds,has_heading_input
        self.time_input_s = time_input_seconds(meta)
        self.heading_input = has_heading_input(meta)
        from sim2sim.policy_memory import has_yaw_memory,YawDriftMemory
        self.yaw_memory_input = has_yaw_memory(meta)
        from sim2sim.policy_state import state_input
        self.state_input=state_input(meta)
        from sim2sim.policy_task_state import task_input,BrakeTaskState
        self.task_input=task_input(meta)
        self.task_state=BrakeTaskState() if self.task_input else None
        self._yaw_memory = YawDriftMemory() if self.yaw_memory_input else None
        self.manifest = _load_sidecar(self.path)
        src: dict[str, Any] = dict(meta)
        if self.manifest is not None:
            src.update(self.manifest)
        self.action_scale = _as_float(src.get("action_scale"), 1.0)
        sim = self.manifest if self.manifest is not None else {}
        self.twist_limits = _twist_limits_from(_nested_get(sim, "sim2sim.twist_limits"))
        stand = _nested_get(sim, "sim2sim.use_stand_policy")
        self.has_standing_partner = True if stand is None else bool(stand)

    def check_dims(self, home_len: int) -> None:
        want_act = int(home_len)
        want_obs = expected_obs_dim(home_len)+(7 if self.task_input else 0)
        errs: list[str] = []
        if self.act_dim != want_act:
            errs.append(f"act_dim={self.act_dim} want={want_act}")
        if self.obs_dim != want_obs:
            errs.append(f"obs_dim={self.obs_dim} want={want_obs}")
        if errs:
            raise PolicyShapeError(f"{self.path}: " + ", ".join(errs))

    def reset_context(self) -> None:
        if self._yaw_memory is not None:self._yaw_memory.reset()
        if self.task_state is not None:self.task_state.reset()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1] != self.obs_dim:
            raise PolicyShapeError(f"obs last dim {x.shape[-1]} != {self.obs_dim}")
        if not np.isfinite(x).all():
            raise PolicyNumericError(f"obs has NaN/inf ({self.path})")
        if self._yaw_memory is not None:x=self._yaw_memory.observe(x.reshape(-1))
        y = self.sess.run([self.output_name], {self.input_name: x.reshape(1, -1)})[0]
        out = np.asarray(y, dtype=np.float32).reshape(-1)
        if not np.isfinite(out).all():
            raise PolicyNumericError(f"action has NaN/inf ({self.path})")
        return out


OnnxPolicy = PolicyBundle
