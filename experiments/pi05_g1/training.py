"""Bounded full-parameter pi05 G1 training, without openpi data_loader dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .model_runtime import ModelRuntime, HORIZON, PROMPT


class MasterAdamW:
    """FP32 master weights/moments with BF16 model storage and computation."""
    def __init__(self, parameters, **kwargs):
        import torch
        self.model_parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.master_parameters = [torch.nn.Parameter(parameter.detach().to(dtype=torch.float32, copy=True), requires_grad=True) for parameter in self.model_parameters]
        kwargs.setdefault("foreach", False)  # Avoid another full parameter-sized GPU temporary list.
        self.optimizer = torch.optim.AdamW(self.master_parameters, **kwargs)

    def zero_grad(self, set_to_none=True):
        for parameter in self.model_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        import torch
        for parameter, master in zip(self.model_parameters, self.master_parameters, strict=True):
            master.grad = None if parameter.grad is None else parameter.grad.detach().float()
        result = self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(self.model_parameters, self.master_parameters, strict=True):
                parameter.copy_(master)
        return result

    def state_dict(self):
        return {"schema": "fp32_master_adamw_v1", "master_parameters": [parameter.detach() for parameter in self.master_parameters], "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state, progress=None):
        import torch
        notify = progress or (lambda phase: None)
        notify("before_master_restore")
        self.zero_grad(set_to_none=True)
        if state.get("schema") != "fp32_master_adamw_v1":
            raise ValueError("Resume requires FP32 master optimizer checkpoint")
        if len(state["master_parameters"]) != len(self.master_parameters):
            raise ValueError("Master parameter count changed")
        with torch.no_grad():
            for index, (model, master, saved) in enumerate(zip(self.model_parameters, self.master_parameters, state["master_parameters"], strict=True)):
                if saved.shape != master.shape or saved.dtype != torch.float32:
                    raise ValueError("Master parameter shape or dtype changed")
                master.copy_(saved)
                model.copy_(master)
                if (index + 1) % 64 == 0:
                    notify(f"master_parameters_restored_{index + 1}")
        notify("after_master_restore")
        # Standard PyTorch restores per-parameter tensors onto the FP32 masters
        # and restores param-group hyperparameters/step counters. Drop any old
        # moments first, so reuse cannot temporarily keep two full GPU state sets.
        self.optimizer.state.clear()
        notify("before_moment_restore")
        self.optimizer.load_state_dict(state["optimizer"])
        notify("after_moment_restore")
        # AdamW intentionally leaves non-capturable step counters on CPU. Without
        # this tiny clone they retain the complete mmap-backed checkpoint storage
        # after `del saved`, even though masters/moments have moved to CUDA.
        for values in self.optimizer.state.values():
            counter = values.get("step")
            if torch.is_tensor(counter) and counter.device.type == "cpu":
                values["step"] = counter.detach().clone()
        notify("after_cpu_step_detached")


class EpisodeDataset:
    """Episode-safe action chunks and timestamp-matched egocentric video."""
    def __init__(self, root, manifest, split, normalizer):
        import pyarrow.parquet as pq
        from .contract import encode_row
        self.root = Path(root)
        if not (self.root / "meta/info.json").is_file() and (self.root / "lerobot/meta/info.json").is_file():
            self.root = self.root / "lerobot"
        info = json.loads((self.root / "meta/info.json").read_text())
        self.info = info
        manifest_data = json.loads(Path(manifest).read_text())
        ids = manifest_data[split]
        frozen_files = {int(item["episode_id"]): item for item in manifest_data.get("files", [])}
        if not frozen_files:
            raise ValueError("Manifest requires audited file hashes; run prepare_training_data first")
        def verify_hash(path, expected):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Dataset changed after audit: {path}")
        if not ids:
            raise ValueError(f"{split} split has no complete episodes")
        candidates = [k for k, v in info["features"].items() if v.get("dtype") == "video" and ("ego" in k.lower() or "head" in k.lower())]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one ego/head video feature, found {candidates}")
        self.video_key = candidates[0]
        self.episodes = []
        self.indices = []
        for episode in ids:
            fmt = dict(episode_index=int(episode), episode_chunk=int(episode) // info.get("chunks_size", 1000), video_key=self.video_key)
            data_path = self.root / info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet").format(**fmt)
            frozen = frozen_files[int(episode)]
            verify_hash(data_path, frozen["parquet_sha256"])
            rows = pq.read_table(data_path).to_pylist()
            if not rows:
                raise ValueError(f"Empty episode: {episode}")
            if any(int(row.get("episode_index", episode)) != int(episode) for row in rows):
                raise ValueError(f"Mixed episode file: {data_path}")
            times = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Nonmonotonic timestamps: {episode}")
            encoded = [encode_row(row) for row in rows]
            states = np.asarray([x["state"] for x in encoded], dtype=np.float32)
            actions = np.asarray([x["actions"] for x in encoded], dtype=np.float32)
            if states.shape != (len(rows), 32) or actions.shape != (len(rows), 32):
                raise ValueError("Contract must encode 32 dimensional state and actions")
            video = self.root / info["video_path"].format(**fmt)
            if not video.is_file():
                raise FileNotFoundError(video)
            verify_hash(video, frozen["video_sha256"][self.video_key])
            item = dict(id=int(episode), states=states, actions=normalizer.normalize(actions, "actions"), timestamps=times, video=video)
            self.episodes.append(item)
            self.indices.extend((len(self.episodes) - 1, i) for i in range(len(rows)))
        self.normalizer = normalizer

    def sample(self, index):
        episode_index, frame = self.indices[index]
        episode = self.episodes[episode_index]
        stop = min(frame + HORIZON, len(episode["actions"]))
        count = stop - frame
        actions = np.repeat(episode["actions"][stop - 1:stop], HORIZON, axis=0)
        actions[:count] = episode["actions"][frame:stop]
        mask = np.arange(HORIZON) < count
        rgb = decode_video_frame(episode["video"], episode["timestamps"][frame])
        return rgb, episode["states"][frame], actions, mask

    def validation_indices(self):
        # Fixed non-overlapping chunks cover each validation episode exactly once.
        return [i for i, (_, frame) in enumerate(self.indices) if frame % HORIZON == 0]


def decode_video_frame(path, timestamp):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if abs(fps - 50.) > .01:
            raise ValueError(f"Frozen source expects 50Hz video, found {fps}: {path}")
        tolerance = .5 / fps + 1e-5
        offset = max(0, int(timestamp / float(stream.time_base)))
        container.seek(offset, stream=stream, backward=True, any_frame=False)
        previous = None
        for frame in container.decode(stream):
            frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else float(frame.time)
            if frame_time >= timestamp:
                chosen = frame if previous is None or abs(frame_time - timestamp) <= abs(previous[0] - timestamp) else previous[1]
                chosen_time = float(chosen.pts * stream.time_base) if chosen.pts is not None else float(chosen.time)
                if abs(chosen_time - timestamp) > tolerance:
                    raise ValueError(f"Video timestamp gap exceeds half a frame: {timestamp} vs {chosen_time} in {path}")
                return chosen.to_ndarray(format="rgb24")
            previous = (frame_time, frame)
        if previous is not None and abs(previous[0] - timestamp) <= tolerance:
            return previous[1].to_ndarray(format="rgb24")
    raise ValueError(f"No timestamp-aligned frame at {timestamp} in {path}")


def masked_flow_loss(losses, valid_mask):
    mask = valid_mask.to(dtype=losses.dtype)[None, :, None].expand_as(losses)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def validate(runtime, dataset, *, deadline=float("inf")):
    torch = runtime.torch
    runtime.model.eval()
    total, count = 0., 0
    # Repeatable flow noise/time and no image augmentation. Keep training RNG unchanged.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in dataset.validation_indices():
                if time.monotonic() >= deadline:
                    runtime.model.train()
                    return None  # A partial validation cannot select a checkpoint.
                rgb, state, actions, mask = dataset.sample(index)
                observation = runtime.observation(rgb, state)
                target = torch.as_tensor(actions, device=runtime.device)[None]
                losses = runtime.model(observation, target)
                tensor_mask = torch.as_tensor(mask, device=runtime.device)
                loss = masked_flow_loss(losses, tensor_mask)
                weight = int(mask.sum()) * 32
                total += float(loss) * weight
                count += weight
    runtime.model.train()
    return total / count


def save_checkpoint(runtime, optimizer, output, state):
    import torch
    from safetensors.torch import save_model
    save_started = time.monotonic()
    output = Path(output)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=True)
    # No full CPU/GPU state clones: safetensors consumes model references and
    # torch.save streams existing master/moment storages. Release dead grads first.
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    save_model(runtime.model, str(temporary / "model.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), **state}, temporary / "training_state.pt")
    state["elapsed_seconds"] += time.monotonic() - save_started
    (temporary / "metadata.json").write_text(json.dumps(state, indent=2))
    os.replace(temporary, output)


def retain_best_and_last(output, latest):
    """Retain two resumable payloads; preserve every checkpoint's metadata evidence."""
    output = Path(output)
    pointer = output / "best_checkpoint.json"
    best = json.loads(pointer.read_text())["path"] if pointer.is_file() else None
    retained = {latest, best}
    pruned = []
    for path in output.iterdir():
        if not path.is_dir() or path.name in retained or not (path.name.startswith("step_") or path.name.startswith("final_")):
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        if "identity" not in metadata or "step" not in metadata:
            continue
        removed = []
        for filename in ("model.safetensors", "training_state.pt"):
            payload = path / filename
            if payload.is_file() and not payload.is_symlink():
                removed.append({"name": filename, "bytes": payload.stat().st_size})
                payload.unlink()
        if removed:
            metadata["payload_retention"] = {"status": "pruned", "reason": "retain_best_and_last", "removed": removed}
            metadata_path.write_text(json.dumps(metadata, indent=2))
            pruned.append(path.name)
    (output / "last_checkpoint.json").write_text(json.dumps({"path": latest}))
    return pruned


def memory_snapshot(torch, checkpoint=None):
    """Read process/system counters; never allocate a tensor or synchronize CUDA."""
    snapshot = {}
    for filename, wanted in (("/proc/self/status", {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap"}), ("/proc/meminfo", {"MemAvailable", "MemFree", "SwapFree"})):
        for line in Path(filename).read_text().splitlines():
            key, _, value = line.partition(":")
            if key in wanted:
                snapshot[key + "_bytes"] = int(value.strip().split()[0]) * 1024
    initialized = torch.cuda.is_initialized()
    snapshot["cuda_initialized"] = initialized
    if initialized:
        snapshot.update(cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if checkpoint:
        filename = str(Path(checkpoint).resolve())
        snapshot["source_checkpoint_mapping_count"] = sum(filename in line for line in Path("/proc/self/maps").read_text().splitlines())
    return snapshot


def resume_admission_guard(snapshot):
    """GB10-specific screening from the measured ~70GB fresh-step peak.

    This is a conservative local resource admission check, not a generic pi05
    requirement or proof that a driver allocation cannot fail.
    """
    limits = {"minimum_mem_available_bytes": 32 * 1024**3, "maximum_cuda_allocated_bytes": 55 * 1024**3, "source_checkpoint_mapping_count": 0}
    reasons = []
    if snapshot.get("source_checkpoint_mapping_count") != 0:
        reasons.append("source_checkpoint_mapping_still_present_or_unknown")
    if snapshot.get("MemAvailable_bytes", -1) < limits["minimum_mem_available_bytes"]:
        reasons.append("system_available_memory_below_32_GiB")
    if "cuda_allocated_bytes" not in snapshot or snapshot["cuda_allocated_bytes"] > limits["maximum_cuda_allocated_bytes"]:
        reasons.append("cuda_allocation_unknown_or_above_55_GiB")
    return {"passed": not reasons, "reasons": reasons, "limits": limits, "scope": "local GB10 screening using measured fresh-step peak of approximately 70GB"}


def train(args):
    import torch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + min(args.hours, 4.) * 3600
    step, best, elapsed_previous = 0, float("inf"), 0.
    log = (output / "progress.jsonl").open("a", buffering=1)
    def event(**fields):
        fields.update(elapsed_seconds=elapsed_previous + time.monotonic() - started)
        line = json.dumps(fields, allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)
    checkpoint_source = Path(args.resume) / "training_state.pt" if args.resume else None
    def phase(name):
        event(event="restore_memory_phase", phase=name, **memory_snapshot(torch, checkpoint_source))
    phase("before_model_load")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = ModelRuntime(args.resume or args.base_checkpoint, args.norm_stats, training=True, head_only=args.head_only)
    phase("after_model_load")
    train_data = EpisodeDataset(args.dataset, args.manifest, "train", runtime.norm)
    val_data = EpisodeDataset(args.dataset, args.manifest, "validation", runtime.norm)
    phase("before_master_initialization")
    optimizer = MasterAdamW(runtime.model.parameters(), lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    phase("after_master_initialization")
    identity = {"manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), "norm_sha256": hashlib.sha256(Path(args.norm_stats).read_bytes()).hexdigest()}
    if args.resume:
        phase("before_mmap_load")
        saved = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
        phase("after_mmap_load")
        if saved["identity"] != identity:
            raise ValueError("Resume manifest or train normalization changed")
        optimizer.load_state_dict(saved["optimizer"], progress=phase)
        step, best, elapsed_previous = saved["step"], saved["best_validation_loss"] if saved["best_validation_loss"] is not None else float("inf"), saved["elapsed_seconds"]
        completed_metadata = json.loads((Path(args.resume) / "metadata.json").read_text())
        elapsed_previous = max(elapsed_previous, completed_metadata["elapsed_seconds"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved  # CPU optimizer checkpoint tensors otherwise stay live for the whole run.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        phase("after_source_released_and_cache_cleared")
        admission_snapshot = memory_snapshot(torch, checkpoint_source)
        admission = resume_admission_guard(admission_snapshot)
        event(event="resume_admission", **admission, **admission_snapshot)
        if not admission["passed"]:
            raise RuntimeError("Resume resource admission failed before training forward: " + ", ".join(admission["reasons"]))
        deadline -= elapsed_previous
    event(event="training_started", step=step, identity=identity, train_frames=len(train_data.indices), validation_chunks=len(val_data.validation_indices()), head_only=runtime.head_only, head_only_reason=runtime.metadata["head_only_reason"], architecture_exclusions=runtime.architecture_exclusions, parameter_dtypes=runtime.dtype_summary, optimizer_master_dtype="float32", float32_stability_parameter_names=runtime.float32_parameter_names, config={"batch_size": 1, "gradient_accumulation": 8, "lr": 1e-5, "max_steps": min(args.max_steps, 2000), "max_hours": min(args.hours, 4.)})
    last_validation = None
    last_saved_name, last_saved_step = None, None
    validation_deadline = deadline - 120.  # Reserve at least two minutes for complete checkpoint writes.
    training_deadline = min(validation_deadline, max(started, deadline - min(300., max(0., (deadline - started) * .1))))
    while step < min(args.max_steps, 2000) and time.monotonic() < training_deadline:
        optimizer.zero_grad(set_to_none=True)
        total_loss, completed = 0., 0
        torch.cuda.synchronize()
        tick = time.monotonic()
        for _ in range(8):
            if time.monotonic() >= training_deadline:
                break
            rgb, state, actions, mask = train_data.sample(random.randrange(len(train_data.indices)))
            observation = runtime.observation(rgb, state)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = runtime.model(observation, torch.as_tensor(actions, device=runtime.device)[None])
                loss = masked_flow_loss(losses, torch.as_tensor(mask, device=runtime.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite flow loss")
            (loss / 8).backward()
            total_loss += float(loss.detach())
            completed += 1
        if completed != 8:
            optimizer.zero_grad(set_to_none=True)
            break
        gradient_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1., error_if_nonfinite=True)
        # Tiny deterministic parameter probes distinguish a completed optimizer
        # call from zero updates due to BF16 quantization.
        probes = {}
        for name, parameter in runtime.model.named_parameters():
            selected = name in ("action_in_proj.weight", "action_out_proj.weight") or (
                name.endswith("layers.0.self_attn.q_proj.weight")
            )
            if selected:
                probes[name] = (parameter, parameter.detach().reshape(-1)[:128].float().clone())
        optimizer.step()
        update_probes = {}
        for name, (parameter, before) in probes.items():
            after = parameter.detach().reshape(-1)[:128].float()
            update_probes[name] = {"sampled_elements": before.numel(), "changed_elements": int((after != before).sum()), "max_abs_delta": float((after - before).abs().max())}
        step += 1
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        event(event="optimizer_step", step=step, train_flow_loss=total_loss / 8, parameter_update_probes=update_probes, gradient_norm=float(gradient_norm), step_seconds=time.monotonic() - tick, allocated_after_zero_grad_bytes=torch.cuda.memory_allocated(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if step % 100 == 0:
            value = validate(runtime, val_data, deadline=validation_deadline)
            last_validation = step if value is not None else None
            event(event="validation", step=step, flow_loss=value, complete=value is not None)
            improved = value is not None and value < best
            if improved:
                best = value
            state = {"step": step, "best_validation_loss": best if np.isfinite(best) else None, "elapsed_seconds": elapsed_previous + time.monotonic() - started, "identity": identity, "validation_flow_loss": value, "head_only": runtime.head_only, "head_only_reason": runtime.metadata["head_only_reason"], "architecture_exclusions": runtime.architecture_exclusions, "parameter_dtypes": runtime.dtype_summary, "float32_stability_parameter_names": runtime.float32_parameter_names, "optimizer_master_dtype": "float32"}
            save_checkpoint(runtime, optimizer, output / f"step_{step:06d}", state)
            last_saved_name, last_saved_step = f"step_{step:06d}", step
            if improved:
                (output / "best_checkpoint.json").write_text(json.dumps({"path": f"step_{step:06d}", "validation_flow_loss": best}))
            event(event="checkpoint_retention", retained_latest=f"step_{step:06d}", pruned=retain_best_and_last(output, f"step_{step:06d}"))
    value = validate(runtime, val_data, deadline=validation_deadline) if step != last_validation else None
    improved = value is not None and value < best
    if improved:
        best = value
    state = {"step": step, "best_validation_loss": best if np.isfinite(best) else None, "elapsed_seconds": elapsed_previous + time.monotonic() - started, "identity": identity, "validation_flow_loss": value, "head_only": runtime.head_only, "head_only_reason": runtime.metadata["head_only_reason"], "architecture_exclusions": runtime.architecture_exclusions, "parameter_dtypes": runtime.dtype_summary, "float32_stability_parameter_names": runtime.float32_parameter_names, "optimizer_master_dtype": "float32"}
    final_name = last_saved_name if last_saved_step == step and not improved else f"final_{step:06d}_{int(time.time())}"
    if final_name != last_saved_name:
        save_checkpoint(runtime, optimizer, output / final_name, state)
    if improved:
        (output / "best_checkpoint.json").write_text(json.dumps({"path": final_name, "validation_flow_loss": best}))
    event(event="checkpoint_retention", retained_latest=final_name, pruned=retain_best_and_last(output, final_name))
    event(event="training_stopped", step=step, reason="max_steps" if step >= min(args.max_steps, 2000) else "time_budget", checkpoint=final_name, validation_flow_loss=value, best_validation_loss=state["best_validation_loss"], selected_checkpoint_exists=(output / "best_checkpoint.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "manifest", "norm-stats", "base-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--head-only", action="store_true", help="Opt-in masked wrist removal after paired equivalence check")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--hours", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps < 1 or args.hours <= 0:
        parser.error("positive max-steps and hours required")
    train(args)


if __name__ == "__main__":
    main()
