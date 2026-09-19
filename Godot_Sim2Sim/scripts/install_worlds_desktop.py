#!/usr/bin/env python3
"""Install the exploration chooser beside the existing workshop entry."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from install_workshop_desktop import exec_quote


def main():
    root = Path(__file__).resolve().parents[1]
    apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    entry = "\n".join([
        "[Desktop Entry]", "Version=1.0", "Type=Application", "Name=风口科学站",
        "Comment=选择风口科学站或小小维修站，MicroDuck · 轮足 · Sai 自由探索",
        f"Exec={exec_quote(root / 'scripts/run-worlds-desktop.sh')}", f"Path={root}",
        f"Icon={root / 'godot/science_station/icon.svg'}", "Terminal=false",
        "Categories=Game;Simulation;", "StartupNotify=false", "StartupWMClass=Robot Godot Worlds",
        "Keywords=MicroDuck;Sai;Godot;科学站;维修站;机器人;", "",
    ])
    with tempfile.TemporaryDirectory(prefix=".worlds-install-", dir=apps) as scratch:
        temporary = Path(scratch) / "robot-godot-worlds.desktop"
        temporary.write_text(entry)
        if shutil.which("desktop-file-validate"):
            subprocess.run(["desktop-file-validate", str(temporary)], check=True)
        destination = apps / temporary.name
        temporary.replace(destination)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=True)
    print(f"Installed: {destination}")


if __name__ == "__main__":
    main()
