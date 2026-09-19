"""Recover an rsl_rl 5.0.1 actor from an exported MLP ONNX graph."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# EmpiricalNormalization.forward: (x - mean) / (_std + eps) with default eps=1e-2.
# Exported ONNX folds that into Div(initializer = _std + eps).
RSL_NORM_EPS = 1e-2
DEFAULT_FROZEN_COUNT = 1_000_000_000
# Walking recovers at ~6e-6. One-shot skills ~1–3e-5. roller.onnx ~1.5e-4
# after clamping inverted stds on near-zero command slots.
PARITY_FAIL_ABS = 2e-4
_ACTOR_OBS_KEY = "actor"

_ACTIVATION_OPS: dict[str, str] = {
    "Elu": "elu",
    "Relu": "relu",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Selu": "selu",
    "LeakyRelu": "lrelu",
    "Gelu": "gelu",
    "Celu": "crelu",
    "Softplus": "softplus",
}

_PASSTHROUGH_OPS = frozenset({"Identity"})


class OnnxMlpParseError(ValueError):
    """The ONNX graph is not a supported Sub/Div → Gemm/Act MLP."""


@dataclass
class RecoveredActor:
    """Weights recovered from an rsl_rl MLPModel ONNX export."""

    mean: np.ndarray
    std: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]
    obs_dim: int
    act_dim: int
    hidden_dims: tuple[int, ...]
    activation: str
    metadata: dict[str, str] = field(default_factory=dict)


def parse_mlp_onnx(path: str | Path) -> RecoveredActor:
    """Walk Sub/Div → Gemm/Act → Gemm and recover actor tensors."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    tensors = _collect_tensors(graph, numpy_helper)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    try:
        return _parse_by_walk(graph, tensors, metadata)
    except OnnxMlpParseError as walk_err:
        try:
            return _parse_by_name_fallback(graph, tensors, metadata)
        except OnnxMlpParseError:
            raise walk_err from None


def build_actor_state_dict(
    rec: RecoveredActor, *, init_std: float, count: int
) -> dict[str, "torch.Tensor"]:
    """Build an rsl_rl 5.0.1 `MLPModel` actor `state_dict`.

    `rec.std` is the ONNX Div operand (i.e. `_std + eps`). Buffers are stored
    so that `EmpiricalNormalization.forward` matches that Div exactly:
    `_std = rec.std - eps`, `_var = _std ** 2`.
    """
    import torch

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    mean = np.asarray(rec.mean, dtype=np.float32).reshape(1, rec.obs_dim)
    div_std = np.asarray(rec.std, dtype=np.float32).reshape(1, rec.obs_dim)
    rsl_std = np.maximum(div_std - np.float32(RSL_NORM_EPS), np.float32(0.0))
    rsl_var = rsl_std * rsl_std
    sd: dict[str, torch.Tensor] = {
        "obs_normalizer._mean": torch.from_numpy(np.copy(mean)),
        "obs_normalizer._var": torch.from_numpy(np.copy(rsl_var)),
        "obs_normalizer._std": torch.from_numpy(np.copy(rsl_std)),
        "obs_normalizer.count": torch.tensor(int(count), dtype=torch.long),
        "distribution.std_param": torch.full((rec.act_dim,), float(init_std), dtype=torch.float32),
    }
    for i, (weight, bias) in enumerate(rec.layers):
        idx = i * 2
        w = np.asarray(weight, dtype=np.float32)
        b = np.asarray(bias, dtype=np.float32)
        sd[f"mlp.{idx}.weight"] = torch.from_numpy(np.copy(w))
        sd[f"mlp.{idx}.bias"] = torch.from_numpy(np.copy(b))
    return sd


def make_rsl_actor(rec: RecoveredActor, *, init_std: float, device: str) -> "torch.nn.Module":
    """Instantiate rsl_rl `MLPModel` as PPO does for the actor and load weights."""
    sd = build_actor_state_dict(rec, init_std=init_std, count=DEFAULT_FROZEN_COUNT)
    meta = {
        "obs_dim": rec.obs_dim,
        "act_dim": rec.act_dim,
        "hidden_dims": rec.hidden_dims,
        "activation": rec.activation,
    }
    return make_rsl_actor_from_state_dict(sd, meta, device=device, init_std=init_std)


def make_rsl_actor_from_state_dict(
    state_dict: dict[str, Any],
    meta: dict[str, Any],
    *,
    device: str,
    init_std: float = 1.0,
) -> "torch.nn.Module":
    """Construct an actor `MLPModel` and `load_state_dict(..., strict=True)`."""
    actor = _construct_mlp_actor(
        obs_dim=int(meta["obs_dim"]),
        act_dim=int(meta["act_dim"]),
        hidden_dims=tuple(int(d) for d in meta["hidden_dims"]),
        activation=str(meta.get("activation", "elu")),
        init_std=float(init_std),
        device=device,
    )
    actor.load_state_dict(state_dict, strict=True)
    return actor


def load_rsl_checkpoint_actor(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a native rsl_rl checkpoint and verify the actor `state_dict` is strict-loadable."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
        raise ValueError(f"{path} is not an rsl_rl 5.0.1 checkpoint (missing actor_state_dict)")
    sd = ckpt["actor_state_dict"]
    obs_dim, act_dim, hidden_dims = _layout_from_state_dict(sd)
    meta: dict[str, Any] = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": hidden_dims,
        "activation": "elu",
        "iter": ckpt.get("iter"),
        "infos": ckpt.get("infos"),
        "source": str(path),
    }
    make_rsl_actor_from_state_dict(sd, meta, device="cpu", init_std=1.0)
    return sd, meta


def verify_parity(
    rec_or_module: RecoveredActor | Any,
    onnx_path: str | Path,
    n: int = 10000,
    seed: int = 0,
) -> float:
    """Max |torch mean-action − onnxruntime| over N(0,1) and realistic 61-D obs."""
    import torch

    if isinstance(rec_or_module, RecoveredActor):
        rec = rec_or_module
        actor = make_rsl_actor(rec, init_std=0.25, device="cpu")
        obs_dim = rec.obs_dim
    else:
        actor = rec_or_module
        obs_dim = int(actor.obs_dim)
    actor.eval()
    device = next(actor.parameters()).device
    rng = np.random.default_rng(seed)
    batches = [rng.standard_normal((n, obs_dim), dtype=np.float32)]
    if obs_dim == 61:
        batches.append(_sample_realistic_obs61(n, rng))
    max_err = 0.0
    onnx_fn = _onnx_actions_fn(onnx_path)
    for obs in batches:
        torch_act = _actor_mean_actions(actor, obs, device)
        onnx_act = onnx_fn(obs)
        max_err = max(max_err, float(np.max(np.abs(torch_act - onnx_act))))
    return max_err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sim2sim-onnx-import")
    parser.add_argument("--onnx", type=Path, required=True, help="Source MLP ONNX (obs→actions)")
    parser.add_argument("--out", type=Path, required=True, help="Output .pt with actor_state_dict")
    parser.add_argument("--init-std", type=float, default=0.25, help="GaussianDistribution scalar std")
    parser.add_argument("--count", type=int, default=DEFAULT_FROZEN_COUNT, help="Frozen normalizer count")
    parser.add_argument("--n", type=int, default=10000, help="Parity sample count per distribution")
    args = parser.parse_args(argv)

    rec = parse_mlp_onnx(args.onnx)
    sd = build_actor_state_dict(rec, init_std=args.init_std, count=args.count)
    err = verify_parity(rec, args.onnx, n=args.n, seed=0)
    payload = {
        "actor_state_dict": sd,
        "source_onnx": str(args.onnx.resolve()),
        "parity_max_abs_err": err,
        "hidden_dims": rec.hidden_dims,
        "metadata": rec.metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(payload, args.out)
    print(f"wrote {args.out}")
    print(f"hidden_dims={rec.hidden_dims} obs_dim={rec.obs_dim} act_dim={rec.act_dim}")
    print(f"parity_max_abs_err={err:.6e}")
    if err >= PARITY_FAIL_ABS:
        print(f"PARITY FAIL: {err} >= {PARITY_FAIL_ABS}", file=sys.stderr)
        return 1
    return 0


# --- graph walk ----------------------------------------------------------------


def _collect_tensors(graph: Any, numpy_helper: Any) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                tensors[node.output[0]] = numpy_helper.to_array(attr.t)
    return tensors


def _user_inputs(graph: Any) -> list[Any]:
    init_names = {i.name for i in graph.initializer}
    return [inp for inp in graph.input if inp.name not in init_names]


def _consumers(graph: Any) -> dict[str, list[Any]]:
    mapping: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            mapping.setdefault(name, []).append(node)
    return mapping


def _attr(node: Any, name: str, default: Any = None) -> Any:
    import onnx

    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _float_attr(node: Any, name: str, default: float) -> float:
    value = _attr(node, name, default)
    if value is None:
        return default
    return float(value)


def _follow(name: str, consumers: dict[str, list[Any]]) -> Any:
    current = name
    while True:
        nodes = consumers.get(current, [])
        if not nodes:
            raise OnnxMlpParseError(f"tensor {current!r} has no consumer")
        if len(nodes) != 1:
            types = [n.op_type for n in nodes]
            raise OnnxMlpParseError(f"tensor {current!r} has {len(nodes)} consumers {types}")
        node = nodes[0]
        if node.op_type in _PASSTHROUGH_OPS:
            current = node.output[0]
            continue
        return node


def _as_vec(arr: np.ndarray, dim: int, what: str) -> np.ndarray:
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.size != dim:
        raise OnnxMlpParseError(f"{what} has {vec.size} elements, expected {dim}")
    return vec


def _parse_gemm(node: Any, tensors: dict[str, np.ndarray], current: str) -> tuple[np.ndarray, np.ndarray, str]:
    if node.op_type != "Gemm":
        raise OnnxMlpParseError(f"expected Gemm, got {node.op_type} ({node.name})")
    alpha = _float_attr(node, "alpha", 1.0)
    beta = _float_attr(node, "beta", 1.0)
    trans_a = int(_attr(node, "transA", 0) or 0)
    trans_b = int(_attr(node, "transB", 0) or 0)
    if trans_a != 0:
        raise OnnxMlpParseError(f"Gemm {node.name} has transA={trans_a}, expected 0")
    if abs(alpha - 1.0) > 1e-6 or abs(beta - 1.0) > 1e-6:
        raise OnnxMlpParseError(f"Gemm {node.name} has alpha={alpha} beta={beta}, expected 1")
    if len(node.input) < 3:
        raise OnnxMlpParseError(f"Gemm {node.name} has no bias")
    a_name, b_name, c_name = node.input[0], node.input[1], node.input[2]
    if a_name != current:
        raise OnnxMlpParseError(f"Gemm {node.name} first input is {a_name!r}, expected {current!r}")
    if b_name not in tensors or c_name not in tensors:
        raise OnnxMlpParseError(f"Gemm {node.name} weight/bias are not initializers")
    weight = np.asarray(tensors[b_name], dtype=np.float32)
    bias = np.asarray(tensors[c_name], dtype=np.float32).reshape(-1)
    # Y = A @ B.T if transB else A @ B. nn.Linear stores (out, in).
    if trans_b:
        if weight.ndim != 2:
            raise OnnxMlpParseError(f"Gemm {node.name} weight rank {weight.ndim}")
        linear_w = weight
    else:
        if weight.ndim != 2:
            raise OnnxMlpParseError(f"Gemm {node.name} weight rank {weight.ndim}")
        linear_w = weight.T
    if linear_w.shape[0] != bias.shape[0]:
        raise OnnxMlpParseError(
            f"Gemm {node.name} out features {linear_w.shape[0]} != bias {bias.shape[0]}"
        )
    return linear_w, bias, node.output[0]


def _parse_by_walk(graph: Any, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> RecoveredActor:
    inputs = _user_inputs(graph)
    if len(inputs) != 1:
        names = [i.name for i in inputs]
        raise OnnxMlpParseError(f"expected one graph input, got {names}")
    obs_name = inputs[0].name
    consumers = _consumers(graph)
    graph_outputs = {o.name for o in graph.output}

    sub = _follow(obs_name, consumers)
    if sub.op_type != "Sub":
        raise OnnxMlpParseError(f"expected Sub after input, got {sub.op_type}")
    if sub.input[0] != obs_name:
        raise OnnxMlpParseError(f"Sub is {sub.input[0]}-{sub.input[1]}, expected obs - mean")
    mean_name = sub.input[1]
    if mean_name not in tensors:
        raise OnnxMlpParseError(f"Sub mean {mean_name!r} is not an initializer/Constant")
    current = sub.output[0]

    div = _follow(current, consumers)
    if div.op_type != "Div":
        raise OnnxMlpParseError(f"expected Div after Sub, got {div.op_type}")
    if div.input[0] != current:
        raise OnnxMlpParseError("Div first input is not the Sub output")
    std_name = div.input[1]
    if std_name not in tensors:
        raise OnnxMlpParseError(f"Div std {std_name!r} is not an initializer/Constant")
    current = div.output[0]

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    activations: list[str] = []
    while True:
        node = _follow(current, consumers)
        w, b, current = _parse_gemm(node, tensors, current)
        layers.append((w, b))
        if current in graph_outputs:
            break
        nxt = _follow(current, consumers)
        if nxt.op_type == "Gemm":
            raise OnnxMlpParseError("expected activation between Gemm layers")
        if nxt.op_type not in _ACTIVATION_OPS:
            raise OnnxMlpParseError(f"unexpected op after Gemm: {nxt.op_type} ({nxt.name})")
        act_name = _ACTIVATION_OPS[nxt.op_type]
        if nxt.op_type == "Elu":
            alpha = _float_attr(nxt, "alpha", 1.0)
            if abs(alpha - 1.0) > 1e-6:
                raise OnnxMlpParseError(f"Elu alpha={alpha}, expected 1.0")
        activations.append(act_name)
        current = nxt.output[0]
        if current in graph_outputs:
            raise OnnxMlpParseError("graph output is an activation, expected final Gemm")

    if len(layers) < 2:
        raise OnnxMlpParseError(f"need at least 2 Gemm layers, got {len(layers)}")
    if not activations or len(activations) != len(layers) - 1:
        raise OnnxMlpParseError("activation count does not match hidden layers")
    if len(set(activations)) != 1:
        raise OnnxMlpParseError(f"mixed activations {activations}")

    obs_dim = int(layers[0][0].shape[1])
    act_dim = int(layers[-1][0].shape[0])
    hidden_dims = tuple(int(w.shape[0]) for w, _ in layers[:-1])
    mean = _as_vec(tensors[mean_name], obs_dim, "mean")
    std = _as_vec(tensors[std_name], obs_dim, "std")
    if np.any(std <= 0):
        raise OnnxMlpParseError("Div std has non-positive entries")
    return RecoveredActor(
        mean=np.array(mean, dtype=np.float32, copy=True),
        std=np.array(std, dtype=np.float32, copy=True),
        layers=layers,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        activation=activations[0],
        metadata=metadata,
    )


def _parse_by_name_fallback(
    graph: Any, tensors: dict[str, np.ndarray], metadata: dict[str, str]
) -> RecoveredActor:
    """Fallback: initializer names `obs_normalizer._mean` / `mlp.{k}.weight`."""
    if "obs_normalizer._mean" not in tensors:
        raise OnnxMlpParseError("fallback: missing initializer obs_normalizer._mean")
    weight_keys = sorted(
        (name for name in tensors if name.startswith("mlp.") and name.endswith(".weight")),
        key=lambda n: int(n.split(".")[1]),
    )
    if not weight_keys:
        raise OnnxMlpParseError("fallback: no mlp.*.weight initializers")
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for w_name in weight_keys:
        b_name = w_name.replace(".weight", ".bias")
        if b_name not in tensors:
            raise OnnxMlpParseError(f"fallback: missing {b_name}")
        w = np.asarray(tensors[w_name], dtype=np.float32)
        b = np.asarray(tensors[b_name], dtype=np.float32).reshape(-1)
        # PyTorch export with transB=1 stores (out, in) already.
        if w.shape[0] != b.shape[0] and w.shape[1] == b.shape[0]:
            w = w.T
        layers.append((w, b))
    div_candidates = []
    for node in graph.node:
        if node.op_type == "Div" and node.input[1] in tensors:
            div_candidates.append(node.input[1])
    if len(div_candidates) != 1:
        # last-ditch: any 1-D initializer matching obs dim besides mean
        obs_dim = int(layers[0][0].shape[1])
        div_candidates = [
            name
            for name, arr in tensors.items()
            if name != "obs_normalizer._mean" and np.asarray(arr).reshape(-1).size == obs_dim
        ]
        if len(div_candidates) != 1:
            raise OnnxMlpParseError(f"fallback: cannot uniquely identify Div std, candidates={div_candidates}")
    std_name = div_candidates[0]
    obs_dim = int(layers[0][0].shape[1])
    act_dim = int(layers[-1][0].shape[0])
    hidden_dims = tuple(int(w.shape[0]) for w, _ in layers[:-1])
    act_nodes = [n.op_type for n in graph.node if n.op_type in _ACTIVATION_OPS]
    if not act_nodes:
        raise OnnxMlpParseError("fallback: no activation nodes")
    mapped = [_ACTIVATION_OPS[op] for op in act_nodes]
    if len(set(mapped)) != 1:
        raise OnnxMlpParseError(f"fallback: mixed activations {mapped}")
    return RecoveredActor(
        mean=_as_vec(tensors["obs_normalizer._mean"], obs_dim, "mean"),
        std=_as_vec(tensors[std_name], obs_dim, "std"),
        layers=layers,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        activation=mapped[0],
        metadata=metadata,
    )


def _construct_mlp_actor(
    *,
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    init_std: float,
    device: str,
) -> "torch.nn.Module":
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs = TensorDict({_ACTOR_OBS_KEY: torch.zeros(1, obs_dim)}, batch_size=[1])
    dist_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": float(init_std),
        "std_type": "scalar",
    }
    actor = MLPModel(
        obs,
        obs_groups={_ACTOR_OBS_KEY: [_ACTOR_OBS_KEY]},
        obs_set=_ACTOR_OBS_KEY,
        output_dim=act_dim,
        hidden_dims=list(hidden_dims),
        activation=activation,
        obs_normalization=True,
        distribution_cfg=dist_cfg,
    )
    return actor.to(device)


def _layout_from_state_dict(sd: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    weight_keys = sorted(
        (k for k in sd if k.startswith("mlp.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not weight_keys:
        raise ValueError("actor_state_dict has no mlp.*.weight keys")
    weights = [sd[k] for k in weight_keys]
    obs_dim = int(weights[0].shape[1])
    act_dim = int(weights[-1].shape[0])
    hidden_dims = tuple(int(w.shape[0]) for w in weights[:-1])
    return obs_dim, act_dim, hidden_dims


def _sample_realistic_obs61(n: int, rng: np.random.Generator) -> np.ndarray:
    gyro = rng.uniform(-3.0, 3.0, size=(n, 3))
    grav = rng.normal(size=(n, 3))
    norms = np.linalg.norm(grav, axis=1, keepdims=True)
    grav = grav / np.clip(norms, 1e-8, None)
    q = rng.uniform(-1.0, 1.0, size=(n, 14))
    qd = rng.uniform(-10.0, 10.0, size=(n, 14))
    act = rng.uniform(-3.0, 3.0, size=(n, 14))
    vx = rng.uniform(-0.4, 0.4, size=(n, 1))
    vy = rng.uniform(-0.3, 0.3, size=(n, 1))
    wz = rng.uniform(-1.0, 1.0, size=(n, 1))
    rest = np.zeros((n, 10), dtype=np.float64)
    cmd = np.concatenate([vx, vy, wz, rest], axis=1)
    obs = np.concatenate([gyro, grav, q, qd, act, cmd], axis=1)
    return obs.astype(np.float32)


def _actor_mean_actions(actor: Any, obs: np.ndarray, device: Any) -> np.ndarray:
    import torch
    from tensordict import TensorDict

    x = torch.as_tensor(obs, device=device)
    td = TensorDict({_ACTOR_OBS_KEY: x}, batch_size=[x.shape[0]])
    with torch.no_grad():
        out = actor(td, stochastic_output=False)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _onnx_actions_fn(onnx_path: str | Path):
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    # Exported graphs pin batch=1; relax the first dim so we can score N samples
    # in one session.run without changing weights or ops.
    for io_value in list(model.graph.input) + list(model.graph.output):
        dims = io_value.type.tensor_type.shape.dim
        if dims:
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "batch"
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run(obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        return sess.run(None, {name: obs})[0].astype(np.float32, copy=False)

    return run


if __name__ == "__main__":
    raise SystemExit(main())
