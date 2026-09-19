"""Train a real 76→15 privileged-state pickup teacher from MuJoCo demonstrations.

The original 61→14 GroundPick actor is frozen inside the exported model. A
learned residual changes the head/neck and commands the new beak actuator.
This is imitation training on *successful physical rollouts*, not yet a visual
policy or a claim of autonomous A→B transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sim2sim.train.onnx_import import parse_mlp_onnx


class PickableActor(nn.Module):
    def __init__(self, source: Path, feature_mean: np.ndarray, feature_std: np.ndarray):
        super().__init__()
        rec = parse_mlp_onnx(source)
        if rec.obs_dim != 61 or rec.act_dim != 14 or rec.activation != "elu":
            raise ValueError("Expected original 61→14 ELU GroundPick actor")
        self.register_buffer("source_mean", torch.tensor(rec.mean, dtype=torch.float32))
        self.register_buffer("source_std", torch.tensor(rec.std, dtype=torch.float32))
        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(feature_std, dtype=torch.float32))
        self.source_layers = nn.ModuleList()
        for weights, bias in rec.layers:
            layer = nn.Linear(weights.shape[1], weights.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(weights))
                layer.bias.copy_(torch.tensor(bias))
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            self.source_layers.append(layer)
        self.residual = nn.Sequential(nn.Linear(28, 96), nn.ELU(), nn.Linear(96, 96),
                                      nn.ELU(), nn.Linear(96, 15))
        # Corrections to body joints are bounded independently. The beak is
        # physical position 0..0.48 rad, trained by a separate binary loss.
        scales = np.full(14, .65, np.float32)
        scales[7] = .3
        self.register_buffer("body_scale", torch.tensor(scales))

    def source(self, obs: torch.Tensor) -> torch.Tensor:
        h = (obs[..., :61]-self.source_mean)/self.source_std
        for layer in self.source_layers[:-1]:
            h = torch.nn.functional.elu(layer(h))
        return self.source_layers[-1](h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = (obs[..., 48:]-self.feature_mean)/self.feature_std
        raw = self.residual(feat)
        body = self.source(obs) + torch.tanh(raw[..., :14])*self.body_scale
        jaw = torch.sigmoid(raw[..., 14:15])*.48
        return torch.cat((body, jaw), dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demonstrations", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--seed", type=int, default=20260919)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    data = np.load(args.demonstrations)
    obs, actions, lengths = data["obs"], data["action"], data["episode_lengths"]
    if len(lengths) < 2 or sum(lengths) != len(obs):
        raise ValueError("Need at least two complete physical demonstrations")
    rng = np.random.default_rng(args.seed)
    episodes = rng.permutation(len(lengths))
    holdout = max(1, int(np.ceil(.2*len(lengths))))
    val_episodes, train_episodes = episodes[:holdout], episodes[holdout:]
    starts = np.r_[0, np.cumsum(lengths)]
    train_idx = np.concatenate([np.arange(starts[i],starts[i+1]) for i in train_episodes])
    val_idx = np.concatenate([np.arange(starts[i],starts[i+1]) for i in val_episodes])
    mean = obs[train_idx, 48:].mean(axis=0)
    std = np.maximum(obs[train_idx, 48:].std(axis=0), .05)
    device = torch.device("cuda")
    model = PickableActor(args.source, mean, std).to(device)
    optimizer = torch.optim.AdamW(model.residual.parameters(), lr=3e-4, weight_decay=1e-5)
    xs = torch.from_numpy(obs).to(device)
    ys = torch.from_numpy(actions).to(device)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        permutation = rng.permutation(train_idx)
        train_losses = []
        for batch in np.array_split(permutation, max(1, int(np.ceil(len(permutation)/512)))):
            predicted = model(xs[batch])
            target = ys[batch]
            body_loss = torch.nn.functional.mse_loss(predicted[:, :14], target[:, :14])
            jaw_loss = torch.nn.functional.binary_cross_entropy(predicted[:,14]/.48,
                                                                 target[:,14]/.48)
            loss = body_loss + .20*jaw_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1.)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            out = model(xs[val_idx])
            val_body = float(torch.nn.functional.mse_loss(out[:,:14],ys[val_idx,:14]))
            val_jaw = float(torch.nn.functional.binary_cross_entropy(out[:,14]/.48,
                                                                      ys[val_idx,14]/.48))
            val_loss = val_body + .20*val_jaw
        row = {"epoch": epoch+1, "train_loss": float(np.mean(train_losses)),
               "val_loss": val_loss, "val_body_mse": val_body, "val_jaw_bce": val_jaw}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%10 == 0:
            print(json.dumps(row), flush=True)
    if best_state is None:
        raise RuntimeError("No trained state")
    model.load_state_dict(best_state)
    model.cpu().eval()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "GroundPick_Godot_Pickable.pt"
    torch.save({"model": best_state, "optimizer": optimizer.state_dict(),
                "history": history, "train_episodes": train_episodes.tolist(),
                "val_episodes": val_episodes.tolist(), "seed": args.seed}, checkpoint)
    onnx_path = args.out / "GroundPick_Godot_Pickable.onnx"
    dummy = torch.from_numpy(obs[:1])
    torch.onnx.export(model, dummy, onnx_path, input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    graph.metadata_props.add(key="training_kind", value="privileged_state_teacher_imitation")
    graph.metadata_props.add(key="source_sha256", value=hashlib.sha256(args.source.read_bytes()).hexdigest())
    onnx.save(graph, str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = obs[val_idx[:min(32,len(val_idx))]]
    ort_out = session.run(None, {"obs": sample})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()
    parity = float(np.max(np.abs(ort_out-torch_out)))
    if parity > 1e-4:
        raise RuntimeError(f"ONNX parity error: {parity}")
    manifest = {"schema_version": 1, "kind": "privileged_state_teacher_imitation",
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "demonstrations_sha256": hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "gpu": torch.cuda.get_device_name(0), "epochs": args.epochs,
                "train_episodes": len(train_episodes), "val_episodes": len(val_episodes),
                "best_val_loss": best_val, "onnx_parity_max_abs": parity,
                "obs_dim": 76, "action_dim": 15, "limitations":
                "Teacher has MuJoCo object pose/mass and is not a camera student or A-to-B policy."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.out / "history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in history))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
