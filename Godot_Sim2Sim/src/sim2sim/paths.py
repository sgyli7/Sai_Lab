"""Filesystem layout for a standalone clone (or nested under pollen-robotics/microduck)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ROOT: Path | None = None


def sim2sim_root() -> Path:
    """Repo root that contains robots/ and godot/."""
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    guesses: list[Path] = []
    env = os.environ.get("SIM2SIM_ROOT")
    if env:
        guesses.append(Path(env))
    try:
        guesses.append(Path.cwd())
    except OSError:
        pass
    guesses.extend(Path(__file__).resolve().parents)
    guesses.append(Path.home() / "Projects/MicroDuck-Godot-Simi2Sim")
    guesses.append(Path.home() / "Projects/MicroDuck/sim2sim")
    for p in guesses:
        try:
            if (p / "robots" / "microduck.json").is_file() and (p / "godot").is_dir():
                _ROOT = p.resolve()
                apply_path_defaults(_ROOT)
                return _ROOT
        except OSError:
            continue
    raise FileNotFoundError("cannot find sim2sim (need robots/microduck.json and godot/)")


def apply_path_defaults(root: Path | None = None) -> None:
    if root is None:
        root = sim2sim_root()
    os.environ.setdefault("SIM2SIM_ROOT", str(root))
    if not os.environ.get("MICRODUCK_RL"):
        for c in (
            Path.home() / "Projects/microduck_rl",
            root.parent / "microduck_rl",
        ):
            try:
                c = c.resolve()
            except OSError:
                continue
            if (c / "src/mjlab_microduck").is_dir():
                os.environ["MICRODUCK_RL"] = str(c)
                break
    if not os.environ.get("MICRODUCK_POLICIES"):
        for c in (
            root / "policies",
            root.parent / "policies",
            Path.home() / "Projects/MicroDuck/policies",
        ):
            if c.is_dir():
                os.environ["MICRODUCK_POLICIES"] = str(c.resolve())
                break


def policies_dir() -> Path:
    apply_path_defaults()
    env = os.environ.get("MICRODUCK_POLICIES")
    if env:
        return Path(env)
    return sim2sim_root() / "policies"


def microduck_rl() -> Path:
    apply_path_defaults()
    env = os.environ.get("MICRODUCK_RL")
    if env:
        return Path(env)
    return Path.home() / "Projects/microduck_rl"


def expand_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    apply_path_defaults()

    def _one(v: Any) -> Any:
        if isinstance(v, str):
            return os.path.expanduser(os.path.expandvars(v))
        if isinstance(v, dict):
            return {k: _one(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_one(x) for x in v]
        return v

    return _one(cfg)


def load_robot_json(path: Path) -> dict[str, Any]:
    return expand_cfg(json.loads(Path(path).read_text()))
