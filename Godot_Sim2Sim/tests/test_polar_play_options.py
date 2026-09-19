from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "integrations/leviathan/godot/hub/hub.gd"
RUNNER = ROOT / "scripts/run_leviathan.py"
SPAWNS = ROOT / "integrations/leviathan/godot/polar_range/spawns.json"


def test_polar_scene_exposes_three_named_spawn_locations() -> None:
    catalog = json.loads(SPAWNS.read_text())
    assert list(catalog) == ["under", "platform", "cockpit"]
    assert [catalog[key]["label"] for key in catalog] == ["车底", "平台上", "驾驶舱室内"]
    assert catalog["under"]["frame"] == "world"
    assert catalog["platform"]["frame"] == "front"
    assert catalog["cockpit"]["frame"] == "front"
    assert all(len(item["position"]) == 3 for item in catalog.values())


def test_polar_hud_allows_all_robot_and_spawn_choices() -> None:
    source = HUB.read_text()
    assert '["microduck","roller","sai","leviathan"]' in source
    assert 'options.has("leviathan") and kind!="sai":return' not in source
    assert "func select_polar_spawn(" in source
    assert 'FileAccess.get_file_as_string("res://polar_range/spawns.json")' in source
    assert "KEY_1" in source and "KEY_2" in source and "KEY_3" in source


def test_polar_runner_accepts_robot_and_spawn_options() -> None:
    source = RUNNER.read_text()
    assert 'p.add_argument("--robot", choices=("microduck", "roller", "sai")' in source
    assert 'p.add_argument("--spawn", choices=("under", "platform", "cockpit")' in source
    assert 'robot=a.robot' in source
    assert 'polar_spawn=a.spawn' in source
