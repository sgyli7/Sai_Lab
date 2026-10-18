"""Paired diagnostic for optional removal of fully masked wrist views.

Run only through the root GPU budget supervisor. This never changes defaults,
starts training, or establishes physical task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .model_runtime import ModelRuntime
from .training import EpisodeDataset, masked_flow_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "norm-stats", "input", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", default=".scratch/pi05_g1/data/lerobot")
    parser.add_argument("--manifest", default=".scratch/pi05_g1/prepared/split_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runtime = ModelRuntime(args.checkpoint, args.norm_stats)
    torch = runtime.torch
    dataset = EpisodeDataset(args.dataset, args.manifest, "train", runtime.norm)
    rgb, state, target, mask = dataset.sample(0)
    with np.load(args.input, allow_pickle=False) as request:
        if not np.array_equal(request["rgb"], rgb) or not np.array_equal(request["state"], state):
            raise ValueError("Paired preflight input must be the actual first training sample for its flow target")
    target = torch.as_tensor(target, device=runtime.device)[None]
    loss_mask = torch.as_tensor(mask, device=runtime.device)
    flow_generator = torch.Generator(device=runtime.device).manual_seed(args.seed + 1000)
    noise = torch.randn(target.shape, device=runtime.device, generator=flow_generator)
    flow_time = torch.full((1,), .5, dtype=torch.float32, device=runtime.device)
    modes = {}
    predictions = {}
    for name, enabled in (("three_views", False), ("head_only", True)):
        runtime.set_head_only(enabled)
        runtime.model.eval()
        # Warm each shape independently, then compare the exact same seed/noise.
        runtime.infer(rgb, state, seed=args.seed, num_steps=10)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = runtime.infer(rgb, state, seed=args.seed, num_steps=10)
        inference_peak = torch.cuda.max_memory_allocated()
        observation = runtime.observation(rgb, state)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = runtime.model(observation, target, noise=noise, time=flow_time)
            flow_loss = float(masked_flow_loss(losses, loss_mask))
        torch.cuda.synchronize()
        modes[name] = {"head_only": enabled, "inference_seconds": result["inference_seconds"], "inference_peak_allocated_bytes": inference_peak, "fixed_flow_loss": flow_loss, "flow_forward_seconds": time.monotonic() - started, "flow_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        predictions[name] = result
    span = np.asarray(runtime.norm.stats["actions"]["q99"]) - np.asarray(runtime.norm.stats["actions"]["q01"])
    active = span > 1e-6
    if not active.any():
        raise ValueError("No nonconstant action dimensions for comparison")
    normalized_diff = predictions["head_only"]["normalized_actions"][:, active] - predictions["three_views"]["normalized_actions"][:, active]
    physical_diff = predictions["head_only"]["actions"] - predictions["three_views"]["actions"]
    normalized_max = float(np.max(np.abs(normalized_diff)))
    loss_delta = abs(modes["head_only"]["fixed_flow_loss"] - modes["three_views"]["fixed_flow_loss"])
    finite = all(np.isfinite(value).all() for result in predictions.values() for key, value in result.items() if key in ("actions", "normalized_actions")) and np.isfinite(loss_delta)
    passed = bool(finite and normalized_max <= .01 and loss_delta <= .005)
    report = {"status": "passed" if passed else "failed", "thresholds": {"active_normalized_action_max_abs": .01, "flow_loss_absolute_difference": .005}, "checkpoint": args.checkpoint, "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), "same_noise_seed": args.seed, "flow_noise_seed": args.seed + 1000, "flow_time": .5, "num_denoising_steps": 10, "flow_target": "audited first training sample action chunk with tail mask", "active_action_dimensions": np.flatnonzero(active).tolist(), "active_normalized_action_max_abs": normalized_max, "active_normalized_action_rms": float(np.sqrt(np.mean(np.square(normalized_diff.astype(np.float64))))), "physical_action_max_abs": float(np.max(np.abs(physical_diff))), "physical_action_max_abs_per_dimension": np.max(np.abs(physical_diff), axis=0).tolist(), "flow_loss_absolute_difference": float(loss_delta), "modes": modes, "physical_success_evaluated": False, "default_changed": False, "interpretation": "floating point equivalence diagnostic; not bitwise identity or task effectiveness"}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
