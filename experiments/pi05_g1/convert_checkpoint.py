"""Run pinned upstream JAX->PyTorch conversion with complete parameter coverage.

--checkpoint is the base directory containing params/, not params/ itself.
No weight transformation algorithm is copied or changed here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time


def load_upstream(source_path):
    """Use upstream module; omit only its unused training.config CLI dependency if missing."""
    source_path = Path(source_path).resolve()
    source = source_path.read_text()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("pi05_upstream_conversion", source_path)
    module = importlib.util.module_from_spec(spec)
    removed = []
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        # Retry unchanged transformation functions, with the CLI-only config import omitted.
        tree = ast.parse(source, filename=str(source_path))
        filtered = []
        for node in tree.body:
            if isinstance(node, ast.Import) and any(alias.name == "openpi.training.config" for alias in node.names):
                if len(node.names) != 1:
                    raise RuntimeError("Unexpected mixed import in upstream converter") from exc
                removed.append("import openpi.training.config as _config")
            else:
                filtered.append(node)
        if not removed:
            raise
        tree.body = filtered
        module = importlib.util.module_from_spec(spec)
        exec(compile(ast.fix_missing_locations(tree), str(source_path), "exec"), module.__dict__)
    return module, {"source_path": str(source_path), "source_sha256": digest, "omitted_cli_imports": removed}


def checked_parameter_coverage(model, state_dict):
    """Missing tied aliases require the identical Parameter object to be loaded elsewhere."""
    names = dict(model.named_parameters(remove_duplicate=False))
    loaded_ids = {id(parameter) for name, parameter in names.items() if name in state_dict}
    uncovered = [name for name, parameter in names.items() if parameter.requires_grad and id(parameter) not in loaded_ids]
    state_names = set(model.state_dict())
    unexpected = sorted(set(state_dict) - state_names)
    if uncovered or unexpected:
        raise RuntimeError(f"Incomplete upstream conversion: uncovered_trainable={uncovered}; unexpected={unexpected}")
    aliases = [name for name, parameter in names.items() if name not in state_dict and id(parameter) in loaded_ids]
    return {"trainable_parameter_names": len(names), "covered_unique_parameters": len(loaded_ids), "allowed_shared_aliases": aliases, "unexpected_keys": unexpected, "uncovered_trainable_parameters": uncovered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream-source", default=".scratch/pi05_g1/openpi/examples/convert_jax_model_to_pytorch.py")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not (checkpoint / "params").is_dir():
        parser.error("--checkpoint must contain params/ (pass pi05_base, not pi05_base/params)")
    if "pi05" not in str(checkpoint):
        parser.error("Official converter selects pi05 norm branches using checkpoint path; pi05 must appear in its path")
    if output.exists():
        parser.error("Output already exists; preserve existing conversion evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Conversion is CPU-only; avoid JAX consuming the experiment GPU allocation.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    upstream, provenance = load_upstream(args.upstream_source)
    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, dtype="bfloat16", pytorch_compile_mode=None)
    from .model_runtime import exclude_unused_expert_logits_head
    original_init = PI0Pytorch.__init__
    original_load = PI0Pytorch.load_state_dict
    exclusions = []
    def action_model_init(model, *positional, **keyword):
        original_init(model, *positional, **keyword)
        exclusions.extend(exclude_unused_expert_logits_head(model))
    PI0Pytorch.__init__ = action_model_init
    coverage = {}
    def strict_load(model, state_dict, *positional, **keyword):
        coverage.update(checked_parameter_coverage(model, state_dict))
        result = original_load(model, state_dict, *positional, **keyword)
        aliases = set(coverage["allowed_shared_aliases"])
        missing = set(result.missing_keys) - aliases
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Uncovered model state: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
        return result
    PI0Pytorch.load_state_dict = strict_load
    start = time.monotonic()
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    provenance.update(checkpoint=str(checkpoint), model="pi05_base", pi05=True, action_dim=32, action_horizon=50, precision="bfloat16", status="started")
    try:
        upstream.convert_pi0_checkpoint(str(checkpoint), "bfloat16", str(temporary), config)
        if not coverage or not (temporary / "model.safetensors").is_file():
            raise RuntimeError("Converter did not verify parameter coverage and write weights")
        provenance.update(status="complete", architecture_exclusions=exclusions, parameter_coverage=coverage, elapsed_seconds=time.monotonic() - start)
        (temporary / "conversion_provenance.json").write_text(json.dumps(provenance, indent=2))
        os.replace(temporary, output)
    except BaseException as exc:
        provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - start)
        (temporary / "conversion_provenance.json").write_text(json.dumps(provenance, indent=2))
        raise
    finally:
        PI0Pytorch.__init__ = original_init
        PI0Pytorch.load_state_dict = original_load
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
