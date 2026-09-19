"""Export an rsl_rl actor to ONNX + schema-2 sidecar (mjlab / rsl_rl _OnnxMLPModel path)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sim2sim.paths import policies_dir, sim2sim_root
from sim2sim.policy import PolicyBundle
from sim2sim.train.manifest import DEFAULT_TWIST_LIMITS, build_manifest, git_snapshot, write_manifest
from sim2sim.train.onnx_import import (
    PARITY_FAIL_ABS,
    _layout_from_state_dict,
    load_rsl_checkpoint_actor,
    make_rsl_actor_from_state_dict,
    verify_parity,
)

# Exact strings copied from alpha_walking.onnx metadata_props (opset 18 export).
_ALPHA_METADATA_FALLBACK: dict[str, str] = {
    "run_path": "None",
    "joint_names": (
        "left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,"
        "neck_pitch,head_pitch,head_yaw,head_roll,"
        "right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle"
    ),
    "joint_stiffness": "1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000",
    "joint_damping": "0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000",
    "default_joint_pos": "0.000,-0.087,-0.458,-0.005,0.453,0.349,0.349,0.000,0.000,0.000,0.087,0.458,0.005,-0.453",
    "command_names": "twist,head_pose,body_pose",
    "observation_names": (
        "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
    ),
    "action_scale": "1.0",
}

DEFAULT_DESCRIPTION = "Godot/Jolt fine-tuned perpetual walking gait (sim2sim)."


def alpha_metadata_strings() -> dict[str, str]:
    """Copy metadata_props key/value strings from alpha_walking.onnx when present."""
    import onnx

    path = policies_dir() / "alpha_walking.onnx"
    if not path.is_file():
        alt = Path("/home/ethan/Projects/MicroDuck/policies/alpha_walking.onnx")
        path = alt if alt.is_file() else path
    if not path.is_file():
        return dict(_ALPHA_METADATA_FALLBACK)
    model = onnx.load(str(path))
    got = {prop.key: prop.value for prop in model.metadata_props}
    out = dict(_ALPHA_METADATA_FALLBACK)
    out.update(got)
    return out


def attach_alpha_style_metadata(onnx_path: Path, extra: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    while model.metadata_props:
        model.metadata_props.pop()
    for key, value in alpha_metadata_strings().items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = str(value)
        model.metadata_props.append(entry)
    for key, value in extra.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = str(value)
        model.metadata_props.append(entry)
    onnx.save(model, str(onnx_path))


def sidecar_path(onnx_path: Path) -> Path:
    return Path(onnx_path).with_name(Path(onnx_path).stem + ".manifest.json")


def _parse_twist_limits(raw: str | None) -> dict[str, float]:
    limits = dict(DEFAULT_TWIST_LIMITS)
    if not raw:
        return limits
    path = Path(raw)
    data = json.loads(path.read_text()) if path.is_file() else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--twist-limits must be a JSON object or a file containing one")
    for k, v in data.items():
        if v is not None:
            limits[k] = float(v)
    return limits


def export_actor(
    *,
    state_dict: dict[str, Any],
    source: str,
    out: Path,
    checkpoint: int | None = None,
    run: str = "",
    eval_info: dict | None = None,
    twist_limits: dict | None = None,
    use_stand_policy: bool = True,
    description: str = DEFAULT_DESCRIPTION,
    n_parity: int = 10000,
    name: str = "walk_godot",
    kind: str = "perpetual",
    slot: str = "walk",
    command: dict | None = None,
) -> dict[str, Any]:
    """Build rsl_rl `_OnnxMLPModel` (normalizer folded in, deterministic mean), export, sidecar, parity."""
    import torch

    obs_dim, act_dim, hidden_dims = _layout_from_state_dict(state_dict)
    meta = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": hidden_dims,
        "activation": "elu",
    }
    actor = make_rsl_actor_from_state_dict(state_dict, meta, device="cpu")
    actor.eval()
    onnx_model = actor.as_onnx(verbose=False)
    onnx_model.to("cpu")
    onnx_model.eval()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = (torch.zeros(1, obs_dim),)
    torch.onnx.export(
        onnx_model,
        dummy,
        str(out),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamo=False,
    )
    snap = git_snapshot()
    attach_alpha_style_metadata(
        out,
        extra={"init_from": str(source), "sim2sim_commit": str(snap["commit"])},
    )
    limits = dict(DEFAULT_TWIST_LIMITS)
    limits.update(dict(twist_limits or {}))
    man = build_manifest(
        onnx_path=out,
        source=str(source),
        training={"checkpoint": checkpoint, "run": run, "init_from": str(source)},
        eval=eval_info,
        twist_limits=limits,
        use_stand_policy=use_stand_policy,
        description=description,
        name=name,
        kind=kind,
        slot=slot,
        command=command,
    )
    write_manifest(sidecar_path(out), man)
    err = verify_parity(actor, out, n=n_parity, seed=0)
    print(f"wrote {out}")
    print(f"sidecar {sidecar_path(out)}")
    print(f"parity_max_abs_err={err:.6e}")
    PolicyBundle(out).check_dims(14)
    # Export only to the requested destination. Publishing a temporary export
    # here silently replaced live policies during parity tests and experiments.
    return {"parity_max_abs_err": err, "out": str(out), "manifest": man}


def publish_to_repo_policies(onnx_path: Path | str) -> list[Path]:
    """Explicitly copy ONNX + sidecar into the default policy directory."""
    import shutil

    src = Path(onnx_path)
    dest_dir = sim2sim_root() / "policies"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for p in (src, sidecar_path(src)):
        if not p.is_file():
            continue
        dest = dest_dir / p.name
        try:
            if dest.resolve() == p.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(p, dest)
        copied.append(dest)
    return copied


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim2sim-export")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="rsl_rl 5.0.1 checkpoint (actor_state_dict)")
    src.add_argument("--actor-pt", type=Path, help="onnx_import output .pt with actor_state_dict")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Walk_Godot.onnx path (default: <repo>/policies/Walk_Godot.onnx)",
    )
    p.add_argument("--manifest-eval", type=Path, default=None, help="JSON object written to manifest.eval")
    p.add_argument("--twist-limits", default=None, help="JSON object or path; default vmax_x=0.4 ...")
    p.add_argument("--description", default=DEFAULT_DESCRIPTION)
    p.add_argument("--run", default="")
    p.add_argument("--no-stand-policy", action="store_true", help="sim2sim.use_stand_policy=false")
    p.add_argument("--n", type=int, default=10000, help="parity sample count per distribution")
    args = p.parse_args(argv)

    src_path = args.checkpoint if args.checkpoint is not None else args.actor_pt
    out = args.out if args.out is not None else sim2sim_root() / "policies" / "Walk_Godot.onnx"
    sd, meta = load_rsl_checkpoint_actor(src_path)
    eval_info = None
    if args.manifest_eval is not None:
        eval_info = json.loads(Path(args.manifest_eval).read_text())
    source = str(meta.get("source") or src_path)
    if args.actor_pt is not None:
        # Prefer the recovered ONNX path recorded by sim2sim-onnx-import.
        import torch

        payload = torch.load(str(args.actor_pt), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("source_onnx"):
            source = str(payload["source_onnx"])
    try:
        result = export_actor(
            state_dict=sd,
            source=source,
            out=out,
            checkpoint=meta.get("iter"),
            run=args.run,
            eval_info=eval_info,
            twist_limits=_parse_twist_limits(args.twist_limits),
            use_stand_policy=not args.no_stand_policy,
            description=args.description,
            n_parity=args.n,
        )
    except Exception as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1
    if result["parity_max_abs_err"] >= PARITY_FAIL_ABS:
        print(f"PARITY FAIL: {result['parity_max_abs_err']} >= {PARITY_FAIL_ABS}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
