from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "flat-v1.onnx": "1f9df39d6a034de77764df065cb7208e29143f569c882e84d7bded3caf5656f0",
    "stairs-dev40.onnx": "0eca6930e7ccd3201023ce9dd0ce4b9cb0476a9490020da89802cac598b2e38d",
    "experimental/ascent60.onnx": "501ad1f8e9b6131bc83b9e35012e6f1ad7bcaf7e5869352eefb636440f1e44f4",
    "experimental/descent60.onnx": "821b2cededa8e204a850384b990828d8ac749a07580b6e4ddc563fd437089d57",
}
FLAT_MOTION_SHA256 = "094adb4484b3d812dfb8a056491beda34a23fd0fa6f463b7784854ebc354d53d"


def test_frozen_sai_models_match_their_manifests_and_82x16_contract() -> None:
    bundle = ROOT / "src/sim2sim/assets/sai/upstream"
    for relative, expected in MODELS.items():
        model = bundle / relative
        manifest = json.loads(model.with_suffix(".json").read_text())
        assert hashlib.sha256(model.read_bytes()).hexdigest() == expected
        assert manifest["onnx_sha256"] == expected
        assert manifest["training"]["observation_size"] == 82
        assert manifest["training"]["action_size"] == 16
        assert manifest["training"]["control_hz"] == 50

    model = ROOT / "src/sim2sim/assets/sai/flat-motion-v1.onnx"
    manifest = json.loads(model.with_suffix(".json").read_text())
    assert hashlib.sha256(model.read_bytes()).hexdigest() == FLAT_MOTION_SHA256
    assert manifest["onnx_sha256"] == FLAT_MOTION_SHA256
    assert manifest["observation_size"] == 82
    assert manifest["action_size"] == 16
    assert manifest["control_hz"] == 50


def test_native_launcher_passes_runtime_options_directly_to_godot(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "sai_policy").mkdir(parents=True)
    (runtime / "project.godot").write_text("")
    (runtime / "sai_policy/flat-motion-v1.onnx").write_bytes(b"fixture")
    fake_godot = tmp_path / "godot"
    captured = tmp_path / "args.txt"
    fake_godot.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    fake_godot.chmod(0o755)
    plan = tmp_path / "plan.json"
    plan.write_text('{"seconds": 1}')
    output = tmp_path / "output"
    env = os.environ | {
        "GODOT": str(fake_godot),
        "CAPTURE_ARGS": str(captured),
        "SAI_NATIVE_RUNTIME": str(runtime),
        "SAI_NATIVE_OUTPUT": str(output),
    }

    subprocess.run(
        [str(ROOT / "run-native.sh"), "--headless", "--fast-check", "--scene", "science_station",
         "--robot", "sai", "--plan", str(plan), "--output", str(output)],
        cwd=ROOT,
        env=env,
        check=True,
    )

    arguments = captured.read_text().splitlines()
    assert "--headless" in arguments
    assert "--fixed-fps" in arguments
    assert "--sai-controller=native" in arguments
    assert "--scene=science_station" in arguments
    assert "--robot=sai" in arguments
    assert f"--plan={plan}" in arguments
    assert f"--output={output}" in arguments


def test_native_inference_is_the_default_for_every_interactive_entry() -> None:
    launcher = (ROOT / "scripts/run-worlds-desktop.sh").read_text()
    assert '"$WORLD_ROOT/run-native.sh"' in launcher
    assert "run-workshop.sh" not in launcher

    workshop = (ROOT / "src/sim2sim/workshop.py").read_text()
    assert 'choices=("native", "python"), default="native"' in workshop

    hub = (ROOT / "godot/hub/hub.gd").read_text()
    assert '"sai_controller":"native"' in hub
