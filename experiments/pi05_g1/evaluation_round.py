"""Serial frozen evaluation round; caller owns the single GPU budget and process group.

Use --output .scratch/pi05_g1/evaluation to reuse its existing verified baseline
cases. The selected checkpoint is frozen before any test/evaluation is read.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".scratch/pi05_g1/venv/bin/python"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def select_checkpoint(training_dir, manifest, norm_stats):
    training_dir = Path(training_dir).resolve()
    pointer_path = training_dir / "best_checkpoint.json"
    pointer = read_json(pointer_path)
    selected = (training_dir / pointer["path"]).resolve()
    if not selected.is_relative_to(training_dir) or selected == training_dir:
        raise ValueError("Selected checkpoint must be a child of the training directory")
    metadata_path = selected / "metadata.json"
    metadata = read_json(metadata_path)
    identity = {"manifest_sha256": sha256(manifest), "norm_sha256": sha256(norm_stats)}
    if metadata.get("identity") != identity:
        raise ValueError("Selected checkpoint manifest or train normalization identity differs")
    step, loss = metadata.get("step"), metadata.get("validation_flow_loss")
    if type(step) is not int or step < 1 or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError("Selected checkpoint requires a completed update and finite full-validation loss")
    if pointer.get("validation_flow_loss") != loss:
        raise ValueError("Best pointer and checkpoint validation loss disagree")
    if metadata.get("head_only", False):
        raise ValueError("This round requires the frozen three-view path; head-only equivalence was rejected")
    weights = selected / "model.safetensors"
    return {"training_dir": str(training_dir), "pointer_path": str(pointer_path), "pointer_sha256": sha256(pointer_path), "checkpoint": str(selected), "weights_sha256": sha256(weights), "metadata_sha256": sha256(metadata_path), "step": step, "validation_flow_loss": loss, "identity": identity, "selection_rule": "minimum completed validation flow loss within this declared training attempt; test never selects weights"}


def validate_inputs(args):
    selected = select_checkpoint(args.training_dir, args.manifest, args.norm_stats)
    cases_data = read_json(args.cases)
    cases = cases_data["cases"]
    ids = [case["case_id"] for case in cases]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("Exactly 20 unique frozen physical cases are required")
    manifest = read_json(args.manifest)
    test = manifest["test"]
    if len(test) != 21 or len(set(test)) != 21 or any(type(episode) is not int for episode in test):
        raise ValueError("Expected the 21 predeclared complete test episodes")
    if set(test) & (set(manifest["train"]) | set(manifest["validation"])):
        raise ValueError("Test episodes overlap training or validation")
    gates = read_json(args.replay_result)
    if gates.get("source_interface_passed") is not True or gates.get("replay_passed") is not True:
        raise ValueError("Verified source-interface and physical replay gates are required")
    if not (Path(args.base_checkpoint) / "model.safetensors").is_file():
        raise FileNotFoundError("Official converted base checkpoint is missing")
    receipt = {"schema": "pi05_g1_evaluation_round_v1", "status": "prepared", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "selected": selected, "base_checkpoint": str(Path(args.base_checkpoint).resolve()), "base_weights_sha256": sha256(Path(args.base_checkpoint) / "model.safetensors"), "frozen": {"cases_sha256": sha256(args.cases), "template_sha256": sha256(args.template), "replay_result_sha256": sha256(args.replay_result), **selected["identity"]}, "case_ids": ids, "test_episode_ids": test, "test_checkpoint_selection": False, "gpu_budget_owner": "external root supervisor; no nested budget or worker", "children": []}
    return receipt


def persist(path, receipt):
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    temporary.replace(path)


def child_commands(args, selected):
    output = Path(args.output).resolve()
    common = ["--template", str(Path(args.template).resolve()), "--cases", str(Path(args.cases).resolve()), "--replay-result", str(Path(args.replay_result).resolve()), "--norm-stats", str(Path(args.norm_stats).resolve())]
    commands = []
    for policy, checkpoint in (("baseline", Path(args.base_checkpoint).resolve()), ("finetuned", Path(selected["checkpoint"]))):
        commands.append(("physical_" + policy, [str(VENV_PYTHON), "-m", "experiments.pi05_g1.run_evaluation", *common, "--checkpoint", str(checkpoint), "--policy", policy, "--output", str(output / policy)]))
    for policy, checkpoint in (("baseline", Path(args.base_checkpoint).resolve()), ("finetuned", Path(selected["checkpoint"]))):
        commands.append(("offline_test_" + policy, [str(VENV_PYTHON), "-m", "experiments.pi05_g1.evaluate_offline", "--dataset", str(Path(args.dataset).resolve()), "--manifest", str(Path(args.manifest).resolve()), "--norm-stats", str(Path(args.norm_stats).resolve()), "--checkpoint", str(checkpoint), "--split", "test", "--hours", "0.25", "--output", str(output / ("offline_" + policy + ".json"))]))
    commands.append(("physical_summary", [str(VENV_PYTHON), "-m", "experiments.pi05_g1.summarize_evaluation", "--cases", str(Path(args.cases).resolve()), "--results", str(output / "baseline"), str(output / "finetuned"), "--gates", str(Path(args.replay_result).resolve()), "--output", str(output / "summary.json")]))
    return commands


def run_round(args, *, runner=None):
    runner = runner or subprocess.run
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "round_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Round receipt already exists; preserve and audit it before any rerun")
    receipt = {"schema": "pi05_g1_evaluation_round_v1", "status": "preparing", "requested_input_paths": {key: str(Path(value).resolve()) for key, value in vars(args).items()}, "children": []}
    persist(receipt_path, receipt)
    try:
        receipt = validate_inputs(args)
        persist(receipt_path, receipt)
        frozen_paths = {"cases_sha256": args.cases, "template_sha256": args.template, "replay_result_sha256": args.replay_result, "manifest_sha256": args.manifest, "norm_sha256": args.norm_stats}
        for name, command in child_commands(args, receipt["selected"]):
            for key, path in frozen_paths.items():
                if sha256(path) != receipt["frozen"][key]:
                    raise RuntimeError(f"Frozen round input changed: {key}")
            if sha256(Path(receipt["selected"]["checkpoint"]) / "metadata.json") != receipt["selected"]["metadata_sha256"]:
                raise RuntimeError("Selected checkpoint metadata changed")
            # Pointer changes indicate a concurrent writer or altered selection;
            # do not silently evaluate a newly selected model halfway through.
            if sha256(receipt["selected"]["pointer_path"]) != receipt["selected"]["pointer_sha256"]:
                raise RuntimeError("Frozen best checkpoint pointer changed during this round")
            record = {"name": name, "command": command, "status": "running", "started_unix": time.time()}
            receipt["status"] = "running"
            receipt["children"].append(record)
            persist(receipt_path, receipt)
            print("ROUND_CHILD_START", name, flush=True)
            # Inherit stdout, stderr, environment and process group. SIGTERM has
            # its normal disposition; root's owned unit handles all descendants.
            result = runner(command, cwd=ROOT)
            record.update(exit_code=result.returncode, elapsed_seconds=time.time() - record["started_unix"], status="completed" if result.returncode == 0 else "infrastructure_failed")
            persist(receipt_path, receipt)
            if result.returncode:
                raise RuntimeError(f"Evaluation infrastructure failed in {name}: exit {result.returncode}")
            if name.startswith("offline_test_"):
                policy = name.removeprefix("offline_test_")
                diagnostic = read_json(output / ("offline_" + policy + ".json"))
                if diagnostic.get("status") != "complete" or diagnostic.get("split") != "test":
                    raise RuntimeError(f"Held-out diagnostic incomplete in {name}; no test-based checkpoint selection")
                record["evidence_sha256"] = sha256(output / ("offline_" + policy + ".json"))
            print("ROUND_CHILD_COMPLETE", name, flush=True)
        summary = read_json(output / "summary.json")
        receipt["summary_sha256"] = sha256(output / "summary.json")
        receipt["summary_verdicts"] = {key: summary.get(key) for key in ("complete", "evidence_complete", "training_verdict", "task_verdict")}
        if not summary.get("complete") or not summary.get("evidence_complete") or not all(summary.get("gates", {}).get(key) is True for key in ("source_interface_passed", "replay_passed")):
            raise RuntimeError("Physical summary contains incomplete evidence or failed gates; see preserved summary")
        receipt.update(status="completed", ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        persist(receipt_path, receipt)
        print("ROUND_COMPLETE", json.dumps(receipt["summary_verdicts"]), flush=True)
        return receipt
    except Exception as exc:
        receipt.update(status="infrastructure_failed", error=f"{type(exc).__name__}: {exc}")
        persist(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-dir", "template", "cases", "replay-result", "norm-stats", "dataset", "manifest", "base-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    run_round(parser.parse_args())


if __name__ == "__main__":
    main()
