#!/usr/bin/env python3
"""Correct phase-free stair-policy covariate shift with closed-loop DAgger."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch

import sai_loaded_mujoco as loaded
from distill_sai_stair_phase_free import Actor
from sim2sim.sai_controller import MotionController
from sim2sim.sai_stair_v2 import ACTION_SIZE, OBSERVATION_SIZE


def episode_arrays(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads((directory / "trace.json").read_text())
    rows = [row for row in rows if row["command"] > 0 and row["stage"] == "stairs"]
    return (np.asarray([row["distill_observation"] for row in rows], np.float32),
            np.asarray([row["distill_action"] for row in rows], np.float32))


def fit(model: Actor, datasets, *, epochs: int, seed: int, device: str) -> dict:
    x = torch.from_numpy(np.concatenate([item[0] for item in datasets])).to(device)
    y = torch.from_numpy(np.concatenate([item[1] for item in datasets])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed)
    final = {}
    for epoch in range(epochs):
        losses = []
        for ids in torch.randperm(len(x), generator=generator, device=device).split(1024):
            estimate = model(x[ids]); loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        final = {"epoch": epoch, "train_mse": float(np.mean(losses)), "samples": len(x)}
    return final


def export(model: Actor, directory: Path, metadata: dict, device: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    actor = directory / "final.onnx"; model.eval()
    torch.onnx.export(model, torch.zeros((1, OBSERVATION_SIZE), device=device), str(actor),
        input_names=["obs"], output_names=["actions"], opset_version=18, dynamo=True)
    graph = onnx.load(str(actor), load_external_data=True)
    onnx.save_model(graph, str(actor), save_as_external_data=False)
    actor.with_name(actor.name + ".data").unlink(missing_ok=True)
    digest = hashlib.sha256(actor.read_bytes()).hexdigest()
    profile = {"schema_version": 2, "id": f"sai-stairs-phase-free-dagger-r{metadata['round']}",
        "actor": actor.name, "onnx_sha256": digest, "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE, "contract": "sai-phase-free-stairs-v2",
        "algorithm": "closed-loop dataset aggregation", "training": metadata,
        "control": {"speed": .16, "wheel_residual_scale": 6., "yaw_correction_limit": .4}}
    profile_path = directory / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    return profile_path


@contextmanager
def dagger_rollout(student_profile: Path, teacher_profile: Path, expert_blend: float):
    original = loaded.CompliantController

    class DaggerController:
        def __init__(self, root, parameters=None):
            self.student = MotionController(root, stair_profile=student_profile)
            self.teacher = MotionController(root, stair_profile=teacher_profile)
            self.spec = self.student.spec

        def command(self, state):
            student = self.student.command(state)
            expert = self.teacher.command(state)
            student["distill_target_leg"] = expert["target_leg"]
            if expert_blend > 0.:
                student_target = np.asarray(student["target_leg"], dtype=float)
                expert_target = np.asarray(expert["target_leg"], dtype=float)
                target = (1. - expert_blend) * student_target + expert_blend * expert_target
                student["target_leg"] = target.tolist()
                student["wheel_speed"] = target[3::4].tolist()
            return student

    loaded.CompliantController = DaggerController
    try:
        yield
    finally:
        loaded.CompliantController = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-data", type=Path, required=True)
    parser.add_argument("--teacher-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--expert-blends", type=float, nargs="+", default=[.75, .5, .25, .1])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702])
    parser.add_argument("--cases", nargs="+", default=["up40", "up60"])
    parser.add_argument("--seed", type=int, default=160923)
    args = parser.parse_args()
    if len(args.expert_blends) < args.rounds:
        raise SystemExit("--expert-blends needs at least --rounds entries")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Actor().to(device)
    datasets = [episode_arrays(path) for path in sorted((args.initial_data / "episodes").iterdir())]
    rounds = []
    for round_index in range(args.rounds + 1):
        stats = fit(model, datasets, epochs=args.epochs, seed=args.seed + round_index, device=device)
        metadata = {"round": round_index, "teacher_profile": str(args.teacher_profile.resolve()),
            "initial_data": str(args.initial_data.resolve()), "datasets": len(datasets),
            "cases": args.cases, "seeds": args.seeds, **stats, "rollouts": []}
        profile = export(model, args.output / f"round-{round_index}", metadata, device)
        if round_index == args.rounds:
            rounds.append(metadata); break
        expert_blend = args.expert_blends[round_index]
        metadata["expert_blend"] = expert_blend
        with dagger_rollout(profile.resolve(), args.teacher_profile.resolve(), expert_blend):
            for case in args.cases:
                for seed in args.seeds:
                    destination = args.output / "episodes" / f"r{round_index}-{case}-{seed}"
                    report = loaded.run(case, seed, .1, None, destination, clamped=False)
                    arrays = episode_arrays(destination); datasets.append(arrays)
                    metadata["rollouts"].append({"case": case, "seed": seed,
                        "samples": len(arrays[0]), "completed": report["metrics"]["completed"],
                        "distance": report["metrics"]["distance"],
                        "cargo_accel_rms": report["metrics"]["cargo_accel_rms"]})
        (profile.parent / "profile.json").write_text(json.dumps({**json.loads(profile.read_text()),
            "training": metadata}, indent=2) + "\n")
        rounds.append(metadata)
        print(json.dumps({"round": round_index, "fit": stats,
            "rollouts": metadata["rollouts"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"rounds": rounds}, indent=2) + "\n")
    print(json.dumps({"final_profile": str(profile), "rounds": len(rounds),
        "final_fit": rounds[-1]}, default=str), flush=True)


if __name__ == "__main__":
    main()
