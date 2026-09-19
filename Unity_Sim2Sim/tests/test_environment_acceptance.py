import importlib.util
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate-environment-acceptance.py"
TERRAIN_IDS = [
    "flat_plaza",
    "upstream_pyramid_stairs",
    "upstream_random_grid",
    "upstream_pyramid_slope",
    "upstream_roller_slope",
    "rock_steps",
    "stairs_bridge",
]
CAMERA_MODES = ["Side", "Rear", "Top", "Showcase", "FreeFly"]


def _load_validator():
    spec = importlib.util.spec_from_file_location("environment_acceptance", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_report(tmp_path: Path) -> dict:
    terrain = {}
    camera = {}
    for name in TERRAIN_IDS:
        path = tmp_path / f"terrain-{name}.png"
        path.write_bytes(b"png")
        terrain[name] = str(path)
    for name in CAMERA_MODES:
        path = tmp_path / f"camera-{name}.png"
        path.write_bytes(b"png")
        camera[name] = str(path)
    video = tmp_path / "tour.mp4"
    video.write_bytes(b"video")
    player = tmp_path / "AgenticRobotGame.exe"
    player.write_bytes(b"exe")
    log = tmp_path / "player.log"
    log.write_text("healthy", encoding="utf-8")
    return {
        "schemaVersion": 2,
        "captureCompleted": True,
        "requiresVisualReview": True,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "playerPath": str(player),
        "playerSha256": hashlib.sha256(player.read_bytes()).hexdigest(),
        "playerLog": str(log),
        "playerLogErrorCount": 0,
        "video": str(video),
        "terrainCheckpoints": terrain,
        "cameraCheckpoints": camera,
        "events": [{"name": "cycle terrain", "sentAtSeconds": 1.0}],
    }


def test_validator_accepts_complete_fresh_environment_evidence(tmp_path: Path) -> None:
    validator = _load_validator()
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_valid_report(tmp_path)), encoding="utf-8")

    result = validator.validate_report(report)

    assert result["valid"] is True
    assert result["terrainCount"] == 7
    assert result["cameraCount"] == 5


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda data: data.update(captureCompleted=False), "captureCompleted"),
        (lambda data: data["terrainCheckpoints"].pop("rock_steps"), "rock_steps"),
        (lambda data: data["cameraCheckpoints"].pop("FreeFly"), "FreeFly"),
        (lambda data: data.update(playerLogErrorCount=1), "playerLogErrorCount"),
        (lambda data: data.update(playerSha256="a" * 64), "playerSha256"),
    ],
)
def test_validator_rejects_incomplete_or_failed_evidence(
    tmp_path: Path,
    mutation,
    expected: str,
) -> None:
    validator = _load_validator()
    data = _valid_report(tmp_path)
    mutation(data)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        validator.validate_report(report)


def test_visual_acceptance_tour_names_every_required_capture() -> None:
    source = (ROOT / "scripts" / "run-visual-acceptance.ps1").read_text(encoding="utf-8")
    for terrain_id in TERRAIN_IDS:
        assert terrain_id in source
    for camera_mode in CAMERA_MODES:
        assert camera_mode in source
    assert "requiresVisualReview = $true" in source
    assert "captureCompleted = $true" in source
    assert "return to flat_plaza for handoff" in source
    assert "restore camera Side preset for handoff" in source
    assert "latest-report.json" in source
    assert '$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)' in source
