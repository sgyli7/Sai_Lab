"""Check Sainiverse walkable contacts against this game's prepared Godot runtime."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


GAME = Path(__file__).resolve().parents[1]
SIBLING = GAME.parent.parent / "Sai_Art"
LEGACY = Path("/home/ethan/Projects/RobotDesign/delivery/Sai_Design")


def main() -> int:
    override = os.environ.get("SAINIVERSE_RELEASE_ROOT")
    release = Path(override) if override else SIBLING if (SIBLING / "run.py").is_file() else LEGACY
    probe = release / "tests/check_walkable_contacts.py"
    if not probe.is_file():
        print(f"Sainiverse collision probe missing: {probe}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(probe), "--game", str(GAME), *sys.argv[1:]], cwd=release)


if __name__ == "__main__":
    raise SystemExit(main())
