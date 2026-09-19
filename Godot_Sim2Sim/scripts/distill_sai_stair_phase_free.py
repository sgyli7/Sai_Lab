#!/usr/bin/env python3
"""Distill a successful stair expert into the phase-free 104x16 contract."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

import sai_loaded_mujoco as loaded
from sim2sim.sai_controller import MotionController
from sim2sim.sai_stair_v2 import ACTION_SIZE, OBSERVATION_SIZE


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 256), nn.ELU(),
            nn.Linear(256, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU(),
            nn.Linear(128, ACTION_SIZE))

    def forward(self, obs):
        return self.mlp(obs)


@contextmanager
def teacher(profile: Path):
    original = loaded.CompliantController
    class Teacher(MotionController):
        def __init__(self, root, parameters=None):
            super().__init__(root, stair_profile=profile)
    loaded.CompliantController = Teacher
    try:
        yield
    finally:
        loaded.CompliantController = original


def collect(profile: Path, output: Path, cases: list[str], seeds: list[int]):
    x_train, y_train, x_val, y_val, episodes = [], [], [], [], []
    validation_seed = seeds[-1]
    with teacher(profile):
        for case in cases:
            for seed in seeds:
                destination = output / "episodes" / f"{case}-{seed}"
                report = loaded.run(case, seed, .1, None, destination, clamped=False)
                trace = json.loads((destination / "trace.json").read_text())
                rows = [row for row in trace if row["command"] > 0 and row["stage"] == "stairs"]
                x = np.asarray([row["distill_observation"] for row in rows], np.float32)
                y = np.asarray([row["distill_action"] for row in rows], np.float32)
                target = (x_val, y_val) if seed == validation_seed else (x_train, y_train)
                target[0].append(x);target[1].append(y)
                episodes.append({"case": case, "seed": seed, "samples": len(rows),
                    "completed": report["metrics"]["completed"],
                    "cargo_accel_rms": report["metrics"]["cargo_accel_rms"]})
    return tuple(np.concatenate(items) for items in (x_train, y_train, x_val, y_val)), episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["up40", "up60"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1601, 1602, 1603, 1604])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=160922)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    (x_train, y_train, x_val, y_val), episodes = collect(
        args.teacher_profile.resolve(), args.output, args.cases, args.seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Actor().to(device);optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    x = torch.from_numpy(x_train).to(device);y = torch.from_numpy(y_train).to(device)
    xv = torch.from_numpy(x_val).to(device);yv = torch.from_numpy(y_val).to(device)
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=device);losses = []
        model.train()
        for ids in order.split(1024):
            estimate = model(x[ids]);loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.inference_mode():
                val = model(xv);mae = float((val - yv).abs().mean());mse = float((val - yv).square().mean())
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_mse": mse,
                            "validation_mae": mae})
    actor = args.output / "final.onnx";model.eval()
    torch.onnx.export(model, torch.zeros((1, OBSERVATION_SIZE), device=device), str(actor),
        input_names=["obs"], output_names=["actions"], opset_version=18, dynamo=True)
    exported = onnx.load(str(actor), load_external_data=True)
    onnx.save_model(exported, str(actor), save_as_external_data=False)
    actor.with_name(actor.name + ".data").unlink(missing_ok=True)
    digest = hashlib.sha256(actor.read_bytes()).hexdigest()
    profile = {"schema_version": 2, "id": "sai-stairs-phase-free-distilled-v1",
        "actor": actor.name, "onnx_sha256": digest, "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE, "contract": "sai-phase-free-stairs-v2",
        "algorithm": "state-conditioned behavior distillation",
        "training": {"teacher_profile": str(args.teacher_profile), "cases": args.cases,
            "seeds": args.seeds, "validation_seed": args.seeds[-1], "epochs": args.epochs,
            "samples": {"train": len(x_train), "validation": len(x_val)}, "history": history,
            "episodes": episodes},
        "control": {"speed": .16, "wheel_residual_scale": 6., "yaw_correction_limit": .4}}
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"onnx_sha256": digest, "train_samples": len(x_train),
        "validation_samples": len(x_val), "final": history[-1]}), flush=True)


if __name__ == "__main__":
    main()
