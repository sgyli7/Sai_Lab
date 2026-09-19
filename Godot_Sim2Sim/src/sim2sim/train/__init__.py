"""Training extras: ONNX import, Godot VecEnv, rsl_rl PPO loop."""

from sim2sim.train.onnx_import import (
    RecoveredActor,
    build_actor_state_dict,
    load_rsl_checkpoint_actor,
    make_rsl_actor,
    make_rsl_actor_from_state_dict,
    parse_mlp_onnx,
    verify_parity,
)

__all__ = [
    "RecoveredActor",
    "build_actor_state_dict",
    "load_rsl_checkpoint_actor",
    "make_rsl_actor",
    "make_rsl_actor_from_state_dict",
    "parse_mlp_onnx",
    "verify_parity",
]
