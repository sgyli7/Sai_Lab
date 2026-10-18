"""Held-out action error for diagnosis; physical task success is evaluated separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .model_runtime import ModelRuntime
from .training import EpisodeDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "manifest", "norm-stats", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--hours", type=float, default=.25)
    args = parser.parse_args()
    started = time.monotonic()
    runtime = ModelRuntime(args.checkpoint, args.norm_stats)
    dataset = EpisodeDataset(args.dataset, args.manifest, args.split, runtime.norm)
    indices = dataset.validation_indices()
    absolute = np.zeros(32, dtype=np.float64)
    square = np.zeros(32, dtype=np.float64)
    count, latency = 0, []
    for index in indices:
        if time.monotonic() - started >= args.hours * 3600:
            break
        rgb, state, target, mask = dataset.sample(index)
        prediction = runtime.infer(rgb, state, seed=index)
        truth = runtime.norm.unnormalize(target)
        error = prediction["actions"][mask] - truth[mask]
        absolute += np.abs(error).sum(axis=0)
        square += np.square(error).sum(axis=0)
        count += int(mask.sum())
        latency.append(prediction["inference_seconds"])
    if not count:
        raise RuntimeError("No complete offline inference sample within time budget")
    result = {"status": "complete" if len(latency) == len(indices) else "partial_time_budget", "split": args.split, "checkpoint": args.checkpoint, "valid_frames": count, "chunks_evaluated": len(latency), "chunks_planned": len(indices), "mae_per_dimension": (absolute / count).tolist(), "rmse_per_dimension": np.sqrt(square / count).tolist(), "inference_seconds_mean": float(np.mean(latency)), "inference_seconds_p95": float(np.quantile(latency, .95)), "physical_success_evaluated": False}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
