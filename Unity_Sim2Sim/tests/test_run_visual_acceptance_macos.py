import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run-visual-acceptance-macos.py"


def _load_macos():
    spec = importlib.util.spec_from_file_location("run_visual_acceptance_macos", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


macos = _load_macos()


def test_checkpoint_frame_numbers_match_windows_formula() -> None:
    assert macos.checkpoint_frame_number(0.75, 8, 192) == 7
    assert macos.checkpoint_frame_number(5.85, 8, 192) == 47
    assert macos.checkpoint_frame_number(21.25, 8, 192) == 171


def test_player_command_uses_macos_binary_without_windows_paths() -> None:
    binary = Path("/tmp/AgenticRobotGame.app/Contents/MacOS/AgenticRobotGame")
    command = macos.player_command(
        binary,
        Path("/tmp/player.log"),
        Path("/tmp/tour-report.json"),
        Path("/tmp/frames"),
    )
    rendered = " ".join(command)
    assert command[0] == str(binary)
    assert "Contents/MacOS/AgenticRobotGame" in command[0]
    assert ".exe" not in rendered
    assert "\\" not in rendered
    assert "-microduckTourReport" in command
    assert "-screen-width" in command


def test_copy_checkpoints_and_interaction_pass_with_fake_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = tmp_path / "AgenticRobotGame.app"
    binary = app / "Contents" / "MacOS" / "AgenticRobotGame"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"player-binary")
    output = tmp_path / "out"

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 4242
            self._returncode = 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        @property
        def returncode(self) -> int:
            return self._returncode

    def fake_popen(command, **kwargs):
        session_frames = Path(command[command.index("-microduckTourFrames") + 1])
        tour_report = Path(command[command.index("-microduckTourReport") + 1])
        log_path = Path(command[command.index("-logFile") + 1])
        session_frames.mkdir(parents=True, exist_ok=True)
        for index in range(1, 193):
            (session_frames / f"frame-{index:04d}.png").write_bytes(b"png-frame")
        log_path.write_text("player started\n", encoding="utf-8")
        tour_report.write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "captureCompleted": True,
                    "requiresVisualReview": True,
                    "capturedAt": datetime.now(timezone.utc).isoformat(),
                    "events": [
                        {
                            "name": "select standing policy 2 on flat_plaza",
                            "kind": "press",
                            "virtualKey": 50,
                            "sentAtSeconds": 0.2,
                        }
                    ],
                    "interactionChecks": {
                        "walkingForwardMeters": 0.12,
                        "walkingUprightDot": 0.9,
                        "resetDistanceMeters": 0.01,
                        "hotSwapTicksMonotonic": True,
                        "rollerModelRebuilt": True,
                        "leggedRestored": True,
                        "tensorsFinite": True,
                        "faults": [],
                        "passed": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    def fake_run(command, **kwargs):
        video = Path(command[-1])
        video.write_bytes(b"fake-mp4")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_validate(report_path: Path, output_path: Path) -> dict:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["playerPath"].endswith("Contents/MacOS/AgenticRobotGame")
        assert payload["frameCount"] == 192
        assert payload["framesPerSecond"] == 8
        assert payload["durationSeconds"] == 24.0
        assert payload["playerLogErrorCount"] == 0
        assert payload["playerSha256"] == hashlib.sha256(b"player-binary").hexdigest()
        output_path.write_text(json.dumps({"valid": True}) + "\n", encoding="utf-8")
        return {"valid": True}

    monkeypatch.setattr(macos, "validate_environment_report", fake_validate)

    report = macos.run_visual_acceptance(
        player=app,
        output_directory=output,
        frames_per_second=8,
        popen=fake_popen,
        run_command=fake_run,
        which=lambda name: "/opt/homebrew/bin/ffmpeg" if name == "ffmpeg" else None,
    )

    assert (output / "latest-report.json").is_file()
    assert (output / "latest-validation.json").is_file()
    session = Path(report["sessionDirectory"])
    assert (session / "terrain-flat_plaza.png").is_file()
    assert (session / "camera-FreeFly.png").is_file()
    assert (session / "terrain-stairs_bridge.png").is_file()
    assert report["validation"]["valid"] is True


def test_load_tour_report_rejects_failed_interaction_checks(tmp_path: Path) -> None:
    path = tmp_path / "tour-report.json"
    path.write_text(
        json.dumps(
            {
                "interactionChecks": {
                    "passed": False,
                    "faults": ["no motion"],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="interactionChecks.passed"):
        macos.load_tour_report(path)


def test_scripts_package_import_does_not_require_windows_powershell() -> None:
    source = (ROOT / "scripts" / "run-visual-acceptance-macos.py").read_text(
        encoding="utf-8"
    )
    assert "powershell" not in source.lower()
    assert "user32" not in source.lower()
    for terrain in (
        "flat_plaza",
        "upstream_pyramid_stairs",
        "upstream_random_grid",
        "upstream_pyramid_slope",
        "upstream_roller_slope",
        "rock_steps",
        "stairs_bridge",
    ):
        assert terrain in source
    for camera in ("Side", "Rear", "Top", "Showcase", "FreeFly"):
        assert camera in source
