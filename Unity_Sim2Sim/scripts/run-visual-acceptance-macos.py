#!/usr/bin/env python3
"""macOS Player visual-acceptance driver. Windows SendInput is replaced by an
in-player tour; ffmpeg and evidence validation stay aligned with the ps1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER = ROOT / "Builds" / "macOS" / "AgenticRobotGame.app"
DEFAULT_OUTPUT = ROOT / "artifacts" / "mvp" / "environment-acceptance"
VALIDATOR = ROOT / "scripts" / "validate-environment-acceptance.py"
DURATION_SECONDS = 24.0
PLAYER_TIMEOUT_SECONDS = 120.0
SERIOUS_LOG_PATTERN = (
    r"NullReferenceException|MissingReferenceException|DllNotFoundException|"
    r"EntryPointNotFoundException|Crash!!!|MICRODUCK_PLAYER_SMOKE_FAIL|"
    r"Assertion failed"
)
CHECKPOINT_SPECS: tuple[dict[str, Any], ...] = (
    {"category": "terrain", "id": "flat_plaza", "at": 0.75},
    {"category": "camera", "id": "Side", "at": 0.75},
    {"category": "camera", "id": "Rear", "at": 1.55},
    {"category": "camera", "id": "Top", "at": 2.55},
    {"category": "camera", "id": "Showcase", "at": 3.55},
    {"category": "camera", "id": "FreeFly", "at": 5.85},
    {"category": "terrain", "id": "upstream_pyramid_stairs", "at": 8.35},
    {"category": "terrain", "id": "upstream_random_grid", "at": 11.15},
    {"category": "terrain", "id": "upstream_pyramid_slope", "at": 13.35},
    {"category": "terrain", "id": "upstream_roller_slope", "at": 15.95},
    {"category": "terrain", "id": "rock_steps", "at": 18.65},
    {"category": "terrain", "id": "stairs_bridge", "at": 21.25},
)


def resolve_player_binary(player: Path) -> Path:
    player = player.expanduser().resolve()
    if player.suffix == ".app" and player.is_dir():
        binary = player / "Contents" / "MacOS" / "AgenticRobotGame"
    else:
        binary = player
    if not binary.is_file():
        raise FileNotFoundError(f"macOS Player not found: {binary}")
    return binary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_log_errors(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(SERIOUS_LOG_PATTERN, text))


def frame_count_for(duration_seconds: float, frames_per_second: int) -> int:
    return int(math.ceil(duration_seconds * frames_per_second))


def checkpoint_frame_number(
    at_seconds: float,
    frames_per_second: int,
    frame_count: int,
) -> int:
    return min(
        frame_count,
        max(1, int(at_seconds * frames_per_second) + 1),
    )


def copy_checkpoints(
    session_directory: Path,
    frame_directory: Path,
    frames_per_second: int,
    frame_count: int,
) -> tuple[dict[str, str], dict[str, str]]:
    terrain: dict[str, str] = {}
    camera: dict[str, str] = {}
    for spec in CHECKPOINT_SPECS:
        frame_number = checkpoint_frame_number(
            float(spec["at"]),
            frames_per_second,
            frame_count,
        )
        source = frame_directory / f"frame-{frame_number:04d}.png"
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Missing checkpoint frame {source.name} for {spec['category']}-{spec['id']}"
            )
        destination = session_directory / f"{spec['category']}-{spec['id']}.png"
        shutil.copyfile(source, destination)
        if spec["category"] == "terrain":
            terrain[str(spec["id"])] = str(destination.resolve())
        else:
            camera[str(spec["id"])] = str(destination.resolve())
    return terrain, camera


def ffmpeg_command(
    ffmpeg: str,
    frame_directory: Path,
    frames_per_second: int,
    video_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(frames_per_second),
        "-i",
        str(frame_directory / "frame-%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]


def player_command(
    binary: Path,
    log_path: Path,
    tour_report: Path,
    frames_directory: Path,
    width: int = 1280,
    height: int = 720,
) -> list[str]:
    return [
        str(binary),
        "-screen-width",
        str(width),
        "-screen-height",
        str(height),
        "-screen-fullscreen",
        "0",
        "-logFile",
        str(log_path),
        "-microduckTourReport",
        str(tour_report),
        "-microduckTourFrames",
        str(frames_directory),
    ]


def load_tour_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"tour-report.json was not produced: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("interactionChecks")
    if not isinstance(checks, dict):
        raise ValueError("tour-report.json is missing interactionChecks")
    if checks.get("passed") is not True:
        raise ValueError(
            "interactionChecks.passed is not true: "
            + json.dumps(checks, ensure_ascii=False)
        )
    return payload


def write_report(
    report_path: Path,
    *,
    player_path: Path,
    player_log: Path,
    video_path: Path,
    frames_per_second: int,
    frame_count: int,
    duration_seconds: float,
    events: Sequence[Mapping[str, Any]],
    terrain: Mapping[str, str],
    camera: Mapping[str, str],
    process_id: int,
) -> dict[str, Any]:
    report = {
        "schemaVersion": 2,
        "captureCompleted": True,
        "requiresVisualReview": True,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "playerProcessId": process_id,
        "playerPath": str(player_path.resolve()),
        "playerSha256": sha256_file(player_path),
        "playerLog": str(player_log.resolve()),
        "playerLogErrorCount": count_log_errors(player_log),
        "framesPerSecond": frames_per_second,
        "frameCount": frame_count,
        "durationSeconds": duration_seconds,
        "events": list(events),
        "video": str(video_path.resolve()),
        "terrainCheckpoints": dict(terrain),
        "cameraCheckpoints": dict(camera),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_environment_report(report_path: Path, output: Path) -> dict[str, Any]:
    python = ROOT / ".venv" / "bin" / "python"
    executable = str(python) if python.is_file() else sys.executable
    completed = subprocess.run(
        [executable, str(VALIDATOR), "--input", str(report_path), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Environment acceptance evidence validation failed: "
            + (completed.stdout or completed.stderr)
        )
    return json.loads(output.read_text(encoding="utf-8"))


def wait_for_player(
    process: subprocess.Popen[str],
    timeout_seconds: float,
    *,
    kill_on_timeout: bool = True,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    deadline = clock() + timeout_seconds
    while process.poll() is None:
        if clock() >= deadline:
            if kill_on_timeout:
                process.kill()
                process.wait(timeout=10)
            raise TimeoutError(
                f"macOS Player tour did not finish within {timeout_seconds:.0f}s"
            )
        sleeper(0.2)
    return int(process.returncode or 0)


def run_visual_acceptance(
    *,
    player: Path,
    output_directory: Path,
    frames_per_second: int = 8,
    keep_player_open: bool = False,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    output_directory = output_directory.expanduser().resolve()
    binary = resolve_player_binary(player)
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg was not found on PATH")

    output_directory.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_directory = (output_directory / session).resolve()
    frame_directory = (session_directory / "frames").resolve()
    session_directory.mkdir(parents=True, exist_ok=True)
    frame_directory.mkdir(parents=True, exist_ok=True)
    log_path = (session_directory / "player.log").resolve()
    tour_report_path = (session_directory / "tour-report.json").resolve()
    video_path = (session_directory / "microduck-interactive-acceptance.mp4").resolve()
    command = player_command(binary, log_path, tour_report_path, frame_directory)
    process = popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    returncode = wait_for_player(
        process,
        PLAYER_TIMEOUT_SECONDS,
        kill_on_timeout=not keep_player_open,
    )
    if returncode != 0:
        raise RuntimeError(
            f"macOS Player tour exited with {returncode}; see {log_path}"
        )

    tour_payload = load_tour_report(tour_report_path)
    frame_count = frame_count_for(DURATION_SECONDS, frames_per_second)
    encoded = run_command(
        ffmpeg_command(ffmpeg, frame_directory, frames_per_second, video_path),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if encoded.returncode != 0 or not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError(
            "ffmpeg did not produce the visual acceptance recording: "
            + (encoded.stderr or encoded.stdout or "")
        )

    terrain, camera = copy_checkpoints(
        session_directory,
        frame_directory,
        frames_per_second,
        frame_count,
    )
    report_path = session_directory / "report.json"
    report = write_report(
        report_path,
        player_path=binary,
        player_log=log_path,
        video_path=video_path,
        frames_per_second=frames_per_second,
        frame_count=frame_count,
        duration_seconds=DURATION_SECONDS,
        events=tour_payload.get("events") or [],
        terrain=terrain,
        camera=camera,
        process_id=int(process.pid or 0),
    )
    validation_path = session_directory / "validation.json"
    validation = validate_environment_report(report_path, validation_path)
    shutil.copyfile(report_path, output_directory / "latest-report.json")
    shutil.copyfile(validation_path, output_directory / "latest-validation.json")
    report["tourReport"] = str(tour_report_path.resolve())
    report["validation"] = validation
    report["sessionDirectory"] = str(session_directory.resolve())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player",
        type=Path,
        default=DEFAULT_PLAYER,
        help="Path to AgenticRobotGame.app or its MacOS binary",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--frames-per-second", type=int, default=8)
    parser.add_argument("--keep-player-open", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_visual_acceptance(
            player=args.player,
            output_directory=args.output_directory,
            frames_per_second=args.frames_per_second,
            keep_player_open=args.keep_player_open,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, ensure_ascii=False))
        return 1

    print(json.dumps({"passed": True, "report": report["sessionDirectory"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
