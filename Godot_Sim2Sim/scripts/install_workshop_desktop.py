#!/usr/bin/env python3
"""Install one workshop app and migrate the two legacy desktop shortcuts."""
from __future__ import annotations

import ast
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

APP_ID = "robot-godot-workshop.desktop"
LEGACY_IDS = ("microduck-atelier.desktop", "microduck-showcase.desktop")


def exec_quote(value: Path) -> str:
    # Desktop Entry Exec quoting is distinct from shell quoting.
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return '"' + text.replace("%", "%%") + '"'


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "robot-godot-workshop"
    apps = data / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    launcher = root / "scripts/run-workshop-desktop.sh"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SystemExit(f"Executable launcher missing: {launcher}")
    entry = "\n".join([
        "[Desktop Entry]", "Version=1.0", "Type=Application", "Name=小小维修站",
        "Comment=MicroDuck · 轮滑版 · Sai 001，三机器人统一游戏入口",
        f"Exec={exec_quote(launcher)}", f"Path={root}", f"Icon={root / 'godot/atelier/icon.svg'}",
        "Terminal=false", "Categories=Game;Simulation;", "StartupNotify=false",
        "StartupWMClass=Robot Godot Workshop",
        "Keywords=MicroDuck;Sai;SO101;Robot;Godot;维修站;机器人;", "",
    ])
    with tempfile.TemporaryDirectory(prefix=".workshop-install-", dir=apps) as scratch:
        temporary = Path(scratch) / APP_ID
        temporary.write_text(entry)
        if shutil.which("desktop-file-validate"):
            subprocess.run(["desktop-file-validate", str(temporary)], check=True)
        temporary.replace(apps / APP_ID)
    backup = state / "legacy-launchers" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for name in LEGACY_IDS:
        old = apps / name
        if old.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(backup / name))
    if shutil.which("gsettings"):
        result = subprocess.run(["gsettings", "get", "org.gnome.shell", "favorite-apps"], capture_output=True, text=True)
        if result.returncode == 0:
            favorites = ast.literal_eval(result.stdout.strip().removeprefix("@as "))
            if any(name in favorites for name in LEGACY_IDS):
                migrated = list(dict.fromkeys(APP_ID if name in LEGACY_IDS else name for name in favorites))
                backup.mkdir(parents=True, exist_ok=True)
                (backup / "favorite-apps.json").write_text(json.dumps(favorites, indent=2) + "\n")
                subprocess.run(["gsettings", "set", "org.gnome.shell", "favorite-apps", repr(migrated)], check=True)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=True)
    print(f"Installed: {apps / APP_ID}")
    print(f"Launches: {root / 'run-workshop.sh'}")
    if backup.exists():
        print(f"Legacy shortcut backup: {backup}")


if __name__ == "__main__":
    main()
