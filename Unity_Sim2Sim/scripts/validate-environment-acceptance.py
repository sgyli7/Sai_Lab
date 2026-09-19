#!/usr/bin/env python3
"""Fail closed unless a fresh, complete MicroDuck Player tour was captured."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TERRAINS = (
    "flat_plaza",
    "upstream_pyramid_stairs",
    "upstream_random_grid",
    "upstream_pyramid_slope",
    "upstream_roller_slope",
    "rock_steps",
    "stairs_bridge",
)
EXPECTED_CAMERAS = ("Side", "Rear", "Top", "Showcase", "FreeFly")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_nonempty_file(value: object, field: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must name a file")
    path = Path(value)
    _require(path.is_file(), f"{field} does not exist: {path}")
    _require(path.stat().st_size > 0, f"{field} is empty: {path}")
    return path


def validate_report(report_path: Path) -> dict[str, object]:
    report_path = Path(report_path)
    _require(report_path.is_file(), f"report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8-sig") as handle:
        report = json.load(handle)

    _require(report.get("schemaVersion") == 2, "schemaVersion must be 2")
    _require(report.get("captureCompleted") is True, "captureCompleted must be true")
    _require(report.get("requiresVisualReview") is True, "requiresVisualReview must be true")
    _require(report.get("playerLogErrorCount") == 0, "playerLogErrorCount must be zero")

    captured_at = report.get("capturedAt")
    _require(isinstance(captured_at, str), "capturedAt must be an ISO timestamp")
    try:
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("capturedAt must be an ISO timestamp") from error
    _require(captured.tzinfo is not None, "capturedAt must include a timezone")
    age_seconds = (datetime.now(timezone.utc) - captured.astimezone(timezone.utc)).total_seconds()
    _require(-300 <= age_seconds <= 6 * 60 * 60, "capturedAt is not fresh")

    player = _require_nonempty_file(report.get("playerPath"), "playerPath")
    player_log = _require_nonempty_file(report.get("playerLog"), "playerLog")
    video = _require_nonempty_file(report.get("video"), "video")
    digest = report.get("playerSha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdefABCDEF" for character in digest),
        "playerSha256 must be a 64-character hexadecimal digest",
    )
    actual_digest = hashlib.sha256(player.read_bytes()).hexdigest()
    _require(digest.lower() == actual_digest, "playerSha256 does not match playerPath")

    terrain = report.get("terrainCheckpoints")
    camera = report.get("cameraCheckpoints")
    _require(isinstance(terrain, dict), "terrainCheckpoints must be an object")
    _require(isinstance(camera, dict), "cameraCheckpoints must be an object")
    for terrain_id in EXPECTED_TERRAINS:
        _require(terrain_id in terrain, f"terrainCheckpoints is missing {terrain_id}")
        _require_nonempty_file(terrain[terrain_id], f"terrainCheckpoints.{terrain_id}")
    for camera_mode in EXPECTED_CAMERAS:
        _require(camera_mode in camera, f"cameraCheckpoints is missing {camera_mode}")
        _require_nonempty_file(camera[camera_mode], f"cameraCheckpoints.{camera_mode}")

    _require(isinstance(report.get("events"), list) and report["events"], "events must not be empty")
    return {
        "valid": True,
        "report": str(report_path.resolve()),
        "player": str(player.resolve()),
        "playerLog": str(player_log.resolve()),
        "video": str(video.resolve()),
        "terrainCount": len(EXPECTED_TERRAINS),
        "cameraCount": len(EXPECTED_CAMERAS),
        "capturedAt": captured.isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = validate_report(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {"valid": False, "error": str(error)}
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 1

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
