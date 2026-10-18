"""Shared, CUDA-only official pi05 preprocessing and inference.

Checkpoints must be the official pi05_base converted by openpi's conversion
script, or a full checkpoint emitted by training.py. No random initialization
fallback is permitted.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import sys
import json
import hashlib
import inspect
import time
from pathlib import Path

import numpy as np

from .contract import PROMPT, ACTION_HORIZON, normalize, denormalize

HORIZON = ACTION_HORIZON
DIM = 32


def exclude_unused_expert_logits_head(model):
    """Remove only the root-approved language logits head outside action forward."""
    head = model.paligemma_with_expert.gemma_expert.lm_head
    if head is None or tuple(head.weight.shape) != (257152, 1024):
        raise RuntimeError("Unexpected expert logits head; exclusion must be re-reviewed")
    count = head.weight.numel()
    source_modules = [inspect.getmodule(type(model.paligemma_with_expert)), inspect.getmodule(type(model).__mro__[1])]
    # Runtime's small evaluation subclass inherits PI0Pytorch; conversion uses it directly.
    from openpi.models_pytorch import pi0_pytorch
    source_modules[-1] = pi0_pytorch
    sources = []
    for module in source_modules:
        path = Path(inspect.getfile(module)).resolve()
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    model.paligemma_with_expert.gemma_expert.lm_head = None
    return [{"excluded_parameter": "paligemma_with_expert.gemma_expert.lm_head.weight", "parameter_count": count, "reason": "unused language logits head outside action forward; no pretrained values", "source_evidence": {"gemma_pytorch.py": "constructor lines 57-58; action forward lines 114,126 use gemma_expert.model", "pi0_pytorch.py": "lines 370,462 compute actions with action_out_proj"}, "source_files": sources}]


class QuantileNormalizer:
    def __init__(self, path):
        self.path = Path(path)
        self.stats = json.loads(self.path.read_text())
        self.stats = self.stats.get("norm_stats", self.stats)
        for key in ("state", "actions"):
            for field in ("q01", "q99"):
                value = np.asarray(self.stats[key][field], dtype=np.float32)
                if value.shape != (DIM,) or not np.isfinite(value).all():
                    raise ValueError(f"{key}.{field} must contain 32 finite values")

    def normalize(self, value, key):
        out = normalize(value, self.stats[key])
        if key == "state":
            out[..., -1] = 0
        return out

    def unnormalize(self, value, key="actions"):
        return denormalize(value, self.stats[key])


def resize_pad_rgb(rgb):
    from PIL import Image
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 [height,width,3] in RGB order")
    height, width = rgb.shape[:2]
    scale = min(224 / height, 224 / width)
    image = Image.fromarray(rgb).resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 127.5 - 1


def preprocess_g1_observation(observation, *, train=False, head_only=False):
    """Skip masked cameras while preserving the existing augmentation RNG stream."""
    import torch
    from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
    keys = ("base_0_rgb",) if head_only else ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    result = preprocess_observation_pytorch(observation, train=train, image_keys=keys)
    if train and head_only:
        # Original missing wrists each draw brightness/contrast/saturation, after
        # the head view and before PI0 samples flow noise/time. Batch size is one.
        for _ in range(6):
            torch.rand(1, device=observation.state.device)
    return result


class ModelRuntime:
    def __init__(self, checkpoint, norm_stats, *, training=False, head_only=False):
        import torch
        from safetensors.torch import load_model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend is required; CPU fallback is disabled")
        self.torch = torch
        self.device = torch.device("cuda")
        self.norm = QuantileNormalizer(norm_stats)
        self.head_only = bool(head_only)
        self.checkpoint = Path(checkpoint)
        weights = self.checkpoint / "model.safetensors" if self.checkpoint.is_dir() else self.checkpoint
        if not weights.is_file():
            raise FileNotFoundError(f"Converted pi05_base/full checkpoint missing: {weights}")
        config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, dtype="bfloat16", pytorch_compile_mode=None)

        class EvaluationAwarePI05(PI0Pytorch):
            def _preprocess_observation(self, observation, *, train=True):
                processed = preprocess_g1_observation(observation, train=train and self.training, head_only=getattr(self, "_g1_head_only", False))
                return (list(processed.images.values()), list(processed.image_masks.values()), processed.tokenized_prompt, processed.tokenized_prompt_mask, processed.state)

        self.model = EvaluationAwarePI05(config)
        self.model._g1_head_only = self.head_only
        self.architecture_exclusions = exclude_unused_expert_logits_head(self.model)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        # Establish destination dtypes before loading so resumed FP32 stability
        # parameters are not silently rounded through BF16.
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        load_model(self.model, str(weights), strict=True, device=str(self.device))
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.dtype_summary = {}
        self.float32_parameter_names = []
        for name, parameter in self.model.named_parameters():
            key = str(parameter.dtype)
            entry = self.dtype_summary.setdefault(key, {"parameter_count": 0, "parameter_bytes": 0})
            entry["parameter_count"] += parameter.numel()
            entry["parameter_bytes"] += parameter.numel() * parameter.element_size()
            if parameter.dtype == torch.float32:
                self.float32_parameter_names.append(name)
        self.metadata = {"checkpoint": str(weights), "architecture_exclusions": self.architecture_exclusions, "parameter_dtypes": self.dtype_summary, "float32_stability_parameter_names": self.float32_parameter_names, "upstream_module_modified": True, "head_only": self.head_only, "head_only_reason": "optional removal of fully masked wrist views; six training RNG draws retained"}
        print(json.dumps({"event": "model_runtime_initialized", **self.metadata}), file=sys.stderr, flush=True)
        self.tokenizer = PaligemmaTokenizer(config.max_token_len)
        self.model.train(training)
        if training:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            self.model.gradient_checkpointing_enable()

    def set_head_only(self, enabled):
        self.head_only = bool(enabled)
        self.model._g1_head_only = self.head_only
        self.metadata["head_only"] = self.head_only

    def observation(self, rgb, state, prompt=PROMPT):
        from openpi.models.model import Observation
        torch = self.torch
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (DIM,) or not np.isfinite(state).all():
            raise ValueError("state must contain 32 finite values in contract order")
        normalized = self.norm.normalize(state, "state")
        tokens, token_mask = self.tokenizer.tokenize(prompt, normalized)
        # SigLIP embed_image consumes NCHW. Official preprocess preserves the
        # caller layout, so normalized float images must be channels-first here.
        image = np.transpose(resize_pad_rgb(rgb), (2, 0, 1)).copy()
        tensor = lambda x: torch.as_tensor(np.asarray(x), device=self.device).unsqueeze(0)
        images = {"base_0_rgb": tensor(image)}
        masks = {"base_0_rgb": tensor(True)}
        if not getattr(self, "head_only", False):
            for key in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
                images[key] = tensor(np.zeros_like(image))
                masks[key] = tensor(False)
        return Observation(
            images=images,
            image_masks=masks,
            state=tensor(normalized), tokenized_prompt=tensor(tokens.astype(np.int64)), tokenized_prompt_mask=tensor(token_mask),
        )

    def infer(self, rgb, state, prompt=PROMPT, seed=0, num_steps=10):
        torch = self.torch
        self.model.eval()
        observation = self.observation(rgb, state, prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, HORIZON, DIM), device=self.device, generator=generator)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model.sample_actions(self.device, observation, noise=noise, num_steps=num_steps)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        actions = self.norm.unnormalize(prediction[0].float().cpu().numpy())
        if actions.shape != (HORIZON, DIM) or not np.isfinite(actions).all():
            raise RuntimeError("Model returned invalid actions")
        return {"actions": actions, "normalized_actions": prediction[0].float().cpu().numpy(), "inference_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--head-only", action="store_true", help="Opt-in removal of fully masked wrist views")
    parser.add_argument("--server", action="store_true", help="JSONL stdin/stdout server; diagnostics go to stderr")
    parser.add_argument("--input", help="NPZ containing rgb uint8 and raw state float32[32]")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = ModelRuntime(args.checkpoint, args.norm_stats, head_only=args.head_only)
    if args.server:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                shape = tuple(request["rgb_shape"])
                if len(shape) != 3 or shape[2] != 3 or np.prod(shape) > 4096 * 4096 * 3:
                    raise ValueError("Invalid rgb_shape")
                rgb = np.frombuffer(base64.b64decode(request["rgb_uint8_b64"], validate=True), dtype=np.uint8).reshape(shape)
                with contextlib.redirect_stdout(sys.stderr):
                    result = runtime.infer(rgb, request["state"], request.get("prompt", PROMPT), request.get("seed", 0))
                reply = {"episode_id": request.get("episode_id"), "control_tick": request.get("control_tick"), "actions": result["actions"].tolist(), "inference_seconds": result["inference_seconds"]}
            except Exception as exc:
                reply = {"episode_id": request.get("episode_id"), "control_tick": request.get("control_tick"), "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(reply, allow_nan=False), flush=True)
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required without --server")
    with np.load(args.input, allow_pickle=False) as request:
        result = runtime.infer(request["rgb"], request["state"], seed=args.seed)
    np.savez(args.output, **result)
    print(json.dumps({"status": "inference_complete", "inference_seconds": result["inference_seconds"], "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
