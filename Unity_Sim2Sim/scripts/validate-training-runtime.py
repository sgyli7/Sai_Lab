"""Load the exact PPO checkpoint and TensorBoard stream in the upstream runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    training = json.loads(arguments.training_report.read_text(encoding="utf-8"))
    checkpoint = Path(training["checkpoint"]).resolve()
    if sha256(checkpoint) != training.get("checkpointSha256"):
        raise ValueError("Checkpoint bytes no longer match the training report")

    # A training checkpoint is still external binary input even after hashing.
    # The upstream state is tensors/primitives only, so never enable pickle code
    # execution while validating it.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required_keys = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
    }
    if not isinstance(payload, dict) or not required_keys <= payload.keys():
        raise ValueError("PPO checkpoint is loadable but lacks required training state")
    if payload["iter"] != 4:
        raise ValueError(f"Expected checkpoint iteration 4, got {payload['iter']!r}")

    event_evidence = []
    for item in training.get("eventFiles", []):
        event_path = Path(item["path"]).resolve()
        if sha256(event_path) != item.get("sha256"):
            raise ValueError(f"TensorBoard event bytes changed: {event_path}")
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        max_step = max(
            (event.step for tag in scalar_tags for event in accumulator.Scalars(tag)),
            default=-1,
        )
        if max_step < 4:
            raise ValueError(f"TensorBoard evidence ends before iteration 4: {event_path}")
        event_evidence.append(
            {
                "path": str(event_path),
                "sha256": item["sha256"],
                "scalarTagCount": len(scalar_tags),
                "maximumStep": max_step,
            }
        )

    if not event_evidence:
        raise ValueError("No TensorBoard event evidence was loaded")

    result = {
        "schemaVersion": 1,
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpointSha256": training["checkpointSha256"],
        "iteration": payload["iter"],
        "actorTensorCount": len(payload["actor_state_dict"]),
        "criticTensorCount": len(payload["critic_state_dict"]),
        "eventFiles": event_evidence,
        "torchVersion": torch.__version__,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
