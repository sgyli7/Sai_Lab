"""Schema-2 single-policy sidecar for Walk_Godot.onnx."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim2sim.godot_proc import sim2sim_root

TASK_ID = "Sim2Sim-GodotJolt-Velocity-Walk"
REPO = "MicroDuck/sim2sim"

DEFAULT_TWIST_LIMITS: dict[str, float] = {
    "vmax_x": 0.4,
    "vmin_x": -0.3,
    "vmax_y": 0.2,
    "vmin_y": -0.2,
    "vmax_ang": 1.0,
}

COMMAND_PROSE = {
    "encoding": "constant",
    "idle": [0, 0, 0],
    "twist": "body-frame vx, vy (m/s) and yaw rate wz (rad/s)",
    "head": "unused (zeros) for walking",
    "body": "unused (zeros) for walking",
}


def git_snapshot(repo: Path | None = None) -> dict[str, Any]:
    root = Path(repo) if repo is not None else sim2sim_root()

    def _git(*args: str) -> str:
        r = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return (r.stdout or "").strip()

    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    onnx_path: Path | str,
    source: str,
    training: dict,
    eval: dict | None,
    twist_limits: dict,
    use_stand_policy: bool,
    description: str,
    name: str = "walk_godot",
    kind: str = "perpetual",
    slot: str = "walk",
    command: dict | None = None,
) -> dict:
    del onnx_path  # name/slot are fixed for this gait; path is caller's out file
    snap = git_snapshot()
    tr = {
        "task_id": TASK_ID,
        "repo": REPO,
        "commit": snap["commit"],
        "branch": snap["branch"],
        "dirty": snap["dirty"],
        "run": "",
        "checkpoint": None,
        "exported": _now_iso(),
        "init_from": str(source),
    }
    tr.update(dict(training or {}))
    limits = dict(DEFAULT_TWIST_LIMITS)
    limits.update({k: float(v) for k, v in dict(twist_limits or {}).items() if v is not None})
    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
        "name": str(name),
        "kind": str(kind),
        "slot": str(slot),
        "entry_pose": "standing",
        "action_scale": 1.0,
        "description": description,
        "command": dict(command or COMMAND_PROSE),
        "training": tr,
        "eval": eval,
        "sim2sim": {
            "twist_limits": {
                "vmax_x": float(limits["vmax_x"]),
                "vmin_x": float(limits["vmin_x"]),
                "vmax_y": float(limits["vmax_y"]),
                "vmin_y": float(limits["vmin_y"]),
                "vmax_ang": float(limits["vmax_ang"]),
            },
            "use_stand_policy": bool(use_stand_policy),
            "fall": {"tilt_deg": 70, "min_z": 0.055},
            "control": {"dt": 0.005, "decimation": 4},
        },
    }


def write_manifest(path: Path | str, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
