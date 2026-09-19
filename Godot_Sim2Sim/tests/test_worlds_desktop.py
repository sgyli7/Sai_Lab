from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_installer_creates_one_product_entry_and_migrates_scene_entries(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = tmp_path / "data"
    state = tmp_path / "state"
    apps = data / "applications"
    apps.mkdir(parents=True)
    legacy_ids = (
        "robot-godot-worlds.desktop",
        "microduck-science-station.desktop",
        "robot-godot-workshop.desktop",
        "robot-godot-polar.desktop",
        "microduck-atelier.desktop",
        "microduck-showcase.desktop",
    )
    for name in legacy_ids:
        (apps / name).write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=旧场景入口\n"
            'Exec="/home/ethan/Projects/MicroDuck/sim2sim/scripts/missing.sh"\n'
        )
    legacy_launcher = home / ".local/bin/microduck-science-station"
    legacy_launcher.parent.mkdir(parents=True)
    legacy_launcher.write_text("#!/bin/sh\nexit 1\n")
    env = os.environ | {
        "HOME": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_STATE_HOME": str(state),
    }

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/install_worlds_desktop.py")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    entry = apps / "robot-godot-sim2sim.desktop"
    stable_launcher = home / ".local/bin/Robot_Godot_Sim2Sim"
    assert entry.is_file()
    assert stable_launcher.is_file()
    assert not stable_launcher.is_symlink()
    assert os.access(stable_launcher, os.X_OK)
    assert str(ROOT / "scripts/run-worlds-desktop.sh") in stable_launcher.read_text()
    assert f'Exec="{stable_launcher}"' in entry.read_text()
    assert "Name=Robot_Godot_Sim2Sim" in entry.read_text()
    assert "01、02、03" in entry.read_text()
    assert f"Icon={ROOT / 'godot/atelier/icon.svg'}" in entry.read_text()
    assert all(not (apps / name).exists() for name in legacy_ids)
    assert not legacy_launcher.exists()
    backups = list((state / "robot-godot-sim2sim/legacy-launchers").glob("*/*.desktop"))
    assert {path.name for path in backups} == set(legacy_ids)
    assert len(list((state / "robot-godot-sim2sim/legacy-launchers").glob("*/microduck-science-station"))) == 1
