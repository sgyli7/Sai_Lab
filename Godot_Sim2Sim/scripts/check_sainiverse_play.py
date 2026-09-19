"""Rendered Sainiverse driving and onboard-robot regression on this workstation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run-sainiverse-v0.1.sh"


def run(mode: str, seconds: int, output: Path, *, drive_probe: bool = False, switch_probe: bool = False) -> None:
    env = os.environ.copy()
    if drive_probe:
        env["SAINIVERSE_DRIVE_PROBE"] = "1"
    if switch_probe:
        env["SAINIVERSE_SWITCH_PROBE"] = "microduck,roller,sai001,sai002,vehicle"
    subprocess.run(
        [str(LAUNCHER), "--mode", mode, "--terrain", "polar", "--seconds", str(seconds), "--output", str(output)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def frame_p95_ms(output: Path, *, after_s: float) -> float:
    report = json.loads((output / "run_visual.json").read_text())
    assert report["rendering_method"] == "forward_plus" and report["resolution"] == [1920, 1080]
    frames = sorted(float(row["frame_ms"]) for row in report["frames"] if float(row["simulation_s"]) >= after_s)
    assert len(frames) >= 100, "Not enough rendered frames for a performance verdict"
    return frames[int(0.95 * (len(frames) - 1))]


def check_drive(output: Path) -> dict:
    run("manual", 23, output, drive_probe=True)
    report = json.loads((output / "run.json").read_text())
    samples = report["samples"]
    distance = float(samples[-1]["hull_positions"][0][0]) - float(samples[0]["hull_positions"][0][0])
    lateral = float(samples[-1]["hull_positions"][0][1]) - float(samples[0]["hull_positions"][0][1])
    heading = float(samples[-1]["hull_heading_rad"][0]) - float(samples[0]["hull_heading_rad"][0])
    yaw_request = max(abs(float(row.get("path_yaw_rate_request_rad_s", 0))) for row in samples)
    p95 = frame_p95_ms(output, after_s=3)
    assert not report["failed"] and distance >= 20 and report["peak_speed_kmh"] >= 10
    assert yaw_request >= 0.02 and abs(heading) >= 0.02 and abs(lateral) >= 0.5 and p95 <= 1000 / 60
    return {"mode": "drive", "distance_m": distance, "lateral_m": lateral, "heading_rad": heading, "peak_kmh": report["peak_speed_kmh"], "yaw_request_rad_s": yaw_request, "p95_frame_ms": p95}


def check_microduck(output: Path) -> dict:
    run("cabin_patrol", 16, output)
    patrol = json.loads((output / "robot_patrol.json").read_text())
    rows = patrol["samples"]
    distance = float(rows[-1]["local_source_m"][0]) - float(rows[0]["local_source_m"][0])
    p95 = frame_p95_ms(output, after_s=10.5)
    assert not patrol["error"] and patrol["first_fall"] is None
    assert distance >= 0.5 and max(row["carrier_contacts"] for row in rows) >= 1 and p95 <= 1000 / 60
    return {"mode": "microduck", "distance_m": distance, "p95_frame_ms": p95}


def check_roller(output: Path) -> dict:
    run("deck_patrol", 16, output)
    patrol = json.loads((output / "robot_patrol.json").read_text())
    rows = patrol["samples"]
    distance = float(rows[-1]["local_source_m"][0]) - float(rows[0]["local_source_m"][0])
    p95 = frame_p95_ms(output, after_s=10.5)
    assert not patrol["error"] and patrol["first_fall"] is None
    assert distance >= 1 and max(row["carrier_contacts"] for row in rows) >= 1 and p95 <= 1000 / 60
    return {"mode": "roller", "distance_m": distance, "p95_frame_ms": p95}


def check_sai(output: Path, *, robot_id: str = "Sai_Agent_001") -> dict:
    run("sai_board_002" if robot_id == "Sai_Agent_002" else "sai_board", 16, output)
    boarding = json.loads((output / "sai_boarding.json").read_text())
    rows = boarding["samples"]
    p95 = frame_p95_ms(output, after_s=10.5)
    assert boarding["robot_id"] == robot_id
    assert not boarding["failure"] and min(row["upright"] for row in rows) >= 0.95
    assert max(row["wheels_supported"] for row in rows) == 4 and p95 <= 1000 / 60
    return {"mode": robot_id, "samples": len(rows), "p95_frame_ms": p95}


def check_sai002(output: Path) -> dict:
    return check_sai(output, robot_id="Sai_Agent_002")


def check_switch(output: Path) -> dict:
    run("manual", 44, output, switch_probe=True)
    report = json.loads((output / "robot_switches.json").read_text())
    rows = report["history"]
    assert report["active_robot"] == "vehicle"
    assert [row["robot"] for row in rows] == ["microduck", "roller", "sai001", "sai002", "vehicle"]
    assert [row["physics_hz"] for row in rows] == [200, 200, 1000, 1000, 60]
    minimum_distances = [0.2, 1.0, 0.2, 0.2]
    for row, minimum in zip(rows[:4], minimum_distances):
        assert row["distance_m"] >= minimum, (row["robot"], row["distance_m"])
        assert row.get("first_fall") is None and not row.get("failure"), row
    p95 = frame_p95_ms(output, after_s=11)
    assert p95 <= 1000 / 60
    return {"mode": "switch", "distance_m": [round(row["distance_m"], 3) for row in rows[:4]], "p95_frame_ms": p95}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("drive", "microduck", "roller", "sai", "sai002", "switch", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or Path(tempfile.mkdtemp(prefix="sainiverse-play-"))
    target.mkdir(parents=True, exist_ok=True)
    checks = {"drive": check_drive, "microduck": check_microduck, "roller": check_roller, "sai": check_sai, "sai002": check_sai002, "switch": check_switch}
    selected = checks if args.mode == "all" else {args.mode: checks[args.mode]}
    results = [check(target / name) for name, check in selected.items()]
    print(json.dumps({"results": results, "output": str(target)}, indent=2))
