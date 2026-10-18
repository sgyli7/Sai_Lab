#!/usr/bin/env python3
"""Copy the openpi transformers patch into this task's virtualenv only.

Official openpi overwrites a few transformers modules. The destination must be
the isolated environment under ``.scratch/pi05_g1/venv``. Existing files are
unlinked before writing so a hardlinked uv cache entry is not modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO_ROOT
    / ".scratch"
    / "pi05_g1"
    / "openpi"
    / "src"
    / "openpi"
    / "models_pytorch"
    / "transformers_replace"
)
VENV = REPO_ROOT / ".scratch" / "pi05_g1" / "venv"
MARKER = "use_adarms"


def site_packages(venv: Path) -> Path:
    matches = sorted(venv.glob("lib/python3.11/site-packages"))
    if len(matches) != 1:
        raise SystemExit(f"expected one Python 3.11 site-packages under {venv}, found {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=VENV)
    args = parser.parse_args()
    venv = args.venv.resolve()
    allowed = VENV.resolve()
    if venv != allowed:
        raise SystemExit(f"refusing to patch outside {allowed}")
    destination_root = site_packages(venv) / "transformers"
    if not destination_root.is_dir():
        raise SystemExit(f"transformers is not installed in {destination_root}")
    copied: list[str] = []
    for source in sorted(path for path in SOURCE.rglob("*") if path.is_file()):
        relative = source.relative_to(SOURCE)
        destination = destination_root / relative
        if not destination.resolve().is_relative_to(destination_root.resolve()):
            raise SystemExit(f"refusing to write outside transformers: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.write_bytes(payload)
        copied.append(relative.as_posix())
    marker = destination_root / "models" / "gemma" / "modeling_gemma.py"
    text = marker.read_text()
    if MARKER not in text:
        raise SystemExit(f"patch marker {MARKER!r} missing from {marker}")
    if marker.stat().st_nlink != 1:
        raise SystemExit(f"{marker} is still hardlinked (nlink={marker.stat().st_nlink})")
    print(f"patched {len(copied)} files into {destination_root}")
    for relative in copied:
        print(relative)
    return 0


if __name__ == "__main__":
    sys.exit(main())
