#!/usr/bin/env python3
"""Install the single Robot_Godot_Sim2Sim desktop entry."""
from __future__ import annotations

import ast
from datetime import datetime
import json
import os
from pathlib import Path
from shlex import quote
import shutil
import subprocess
import tempfile
APP_ID = "robot-godot-sim2sim.desktop"
LEGACY_IDS = (
    "robot-godot-worlds.desktop",
    "microduck-science-station.desktop",
    "robot-godot-workshop.desktop",
    "robot-godot-polar.desktop",
    "microduck-atelier.desktop",
    "microduck-showcase.desktop",
)
STABLE_LAUNCHER = "Robot_Godot_Sim2Sim"
LEGACY_LAUNCHERS = ("microduck-science-station",)


def exec_quote(value: Path) -> str:
    """Quote a path for a Desktop Entry Exec field."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return '"' + text.replace("%", "%%") + '"'


def main():
    root = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "robot-godot-sim2sim"
    apps = data / "applications"
    bin_dir = Path.home() / ".local/bin"
    apps.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = root / "scripts/run-worlds-desktop.sh"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SystemExit(f"Executable launcher missing: {launcher}")
    # Installation may use Python to import/validate third-party robot assets;
    # the installed interactive launcher itself starts Godot directly.
    subprocess.run([str(root / "run-workshop.sh"), "--prepare-only"], check=True)
    stable_launcher = bin_dir / STABLE_LAUNCHER
    with tempfile.TemporaryDirectory(prefix=".worlds-launcher-", dir=bin_dir) as scratch:
        temporary = Path(scratch) / STABLE_LAUNCHER
        temporary.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"exec {quote(str(launcher))} \"$@\"\n"
        )
        temporary.chmod(0o755)
        temporary.replace(stable_launcher)
    entry = "\n".join([
        "[Desktop Entry]", "Version=1.0", "Type=Application", "Name=Robot_Godot_Sim2Sim",
        "Comment=从统一主界面选择 01、02、03 场景与机器人",
        f"Exec={exec_quote(stable_launcher)}", f"Path={root}",
        f"Icon={root / 'godot/atelier/icon.svg'}", "Terminal=false",
        "Categories=Game;Simulation;", "StartupNotify=false", "StartupWMClass=Robot Godot Worlds",
        "Keywords=MicroDuck;Roller;Sai;Leviathan;Godot;Sim2Sim;机器人;", "",
    ])
    with tempfile.TemporaryDirectory(prefix=".worlds-install-", dir=apps) as scratch:
        temporary = Path(scratch) / APP_ID
        temporary.write_text(entry)
        if shutil.which("desktop-file-validate"):
            subprocess.run(["desktop-file-validate", str(temporary)], check=True)
        destination = apps / temporary.name
        temporary.replace(destination)
    backup = state / "legacy-launchers" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for name in LEGACY_IDS:
        old = apps / name
        if old.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(backup / name))
    for name in LEGACY_LAUNCHERS:
        old = bin_dir / name
        if old.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(backup / name))
    if shutil.which("gsettings"):
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.shell", "favorite-apps"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            favorites = ast.literal_eval(result.stdout.strip().removeprefix("@as "))
            if any(name in favorites for name in LEGACY_IDS):
                migrated = list(dict.fromkeys(APP_ID if name in LEGACY_IDS else name for name in favorites))
                backup.mkdir(parents=True, exist_ok=True)
                (backup / "favorite-apps.json").write_text(json.dumps(favorites, indent=2) + "\n")
                subprocess.run(
                    ["gsettings", "set", "org.gnome.shell", "favorite-apps", repr(migrated)],
                    check=True,
                )
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=True)
    print(f"Installed: {destination}")
    print(f"Launches: {stable_launcher} -> {launcher}")
    if backup.exists():
        print(f"Legacy shortcut backup: {backup}")


if __name__ == "__main__":
    main()
