#!/usr/bin/env python3
"""Closed-loop MuJoCo-Warp screen for an exported v6 ONNX actor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn.functional as F

from train_sai_stair_task_v6 import SaiStairWorld
from search_sai_high_step_expert import center_reset
from sim2sim.sai_stair_v6 import STAIR_CLEARANCE_M


JOINT_NAMES = tuple(
    f"{leg}_{axis}"
    for leg in ("front_left", "front_right", "rear_left", "rear_right")
    for axis in ("haa", "hip", "knee", "wheel")
)
MEASURED_LIMITS = torch.tensor(
    [0.45, 0.70, 1.20, float("inf")] * 4, device="cuda"
)
NON_WHEEL = torch.tensor(
    [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], device="cuda"
)


def load_actor(path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    graph = onnx.load(str(path), load_external_data=True)
    values = {item.name: torch.from_numpy(numpy_helper.to_array(item).copy()).to("cuda")
              for item in graph.graph.initializer}
    return [(values[f"mlp.{index}.weight"], values[f"mlp.{index}.bias"])
            for index in (0, 2, 4, 6)]


def infer(layers, observation: torch.Tensor) -> torch.Tensor:
    value = observation
    for index, (weight, bias) in enumerate(layers):
        value = F.linear(value, weight, bias)
        if index + 1 < len(layers):
            value = F.elu(value)
    return value.clamp(-1., 1.)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260926)
    parser.add_argument("--horizon", type=int, default=430)
    parser.add_argument("--rise-mm", type=float, default=40.)
    parser.add_argument("--center-reset", action="store_true")
    parser.add_argument("--isolated-step", action="store_true")
    parser.add_argument("--goal-edges", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    layers = load_actor(args.actor)
    env = SaiStairWorld(args.envs, args.seed, max_steps=args.horizon,
        min_rise_mm=args.rise_mm, terrain_stage="high", position_objective=True,
        payload_clamp=True, domain_randomization=False, cargo_accel_weight=.02,
        action_rate_weight=.08, lift_event_weight=.2, upright_weight=16.,
        support_weight=4., isolated_step=args.isolated_step)
    env.curriculum_count = 1
    env.curriculum_edges = args.goal_edges
    ids = torch.arange(args.envs, device="cuda")
    env.reset(ids)
    env.goal_edges.fill_(args.goal_edges)
    goal_x = (env.starts[env.lane] + (env.goal_edges - 1) * env.treads[env.lane]
              + STAIR_CLEARANCE_M)
    env.last_goal_distance.copy_((goal_x - env.qpos[:, env.qa]).clamp(min=0.))
    if args.center_reset:
        center_reset(env)
    active = torch.ones(args.envs, device="cuda", dtype=torch.bool)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    joint_failed = torch.zeros_like(active)
    joint_position_failed = torch.zeros_like(active)
    joint_speed_failed = torch.zeros_like(active)
    tilt_failed = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    min_up = torch.ones(args.envs, device="cuda")
    max_side_error = torch.zeros(args.envs, device="cuda")
    cargo_square = torch.zeros(args.envs, device="cuda")
    samples = torch.zeros(args.envs, device="cuda")
    done_step = torch.full((args.envs,), args.horizon, device="cuda", dtype=torch.long)
    terminal_min_wheel_x = torch.full((args.envs,), float("nan"), device="cuda")
    action_abs_sum = torch.zeros((args.envs, 16), device="cuda")
    peak_joint_utilization = torch.zeros((args.envs, 16), device="cuda")
    peak_joint_speed = torch.zeros((args.envs, 16), device="cuda")
    trace = []
    try:
        with torch.inference_mode():
            for step in range(args.horizon):
                observation = env.get_observations()["policy"]
                action = infer(layers, observation)
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                _, _, _, info = env.step(action)
                min_up = torch.minimum(min_up, torch.where(active, env.last_up, min_up))
                side_error = ((env.last_wheel_x[:, 0] + env.last_wheel_x[:, 2])
                              - (env.last_wheel_x[:, 1] + env.last_wheel_x[:, 3])).abs() * .5
                max_side_error = torch.maximum(
                    max_side_error, torch.where(active, side_error, max_side_error))
                cargo_square += env.last_cargo_accel_norm.square() * active
                samples += active
                action_abs_sum += action.abs() * active[:, None]
                joint_utilization = env.last_joint_position.abs() / MEASURED_LIMITS[None]
                joint_speed = env.last_joint_velocity.abs()
                peak_joint_utilization = torch.maximum(
                    peak_joint_utilization,
                    torch.where(active[:, None], joint_utilization, peak_joint_utilization),
                )
                peak_joint_speed = torch.maximum(
                    peak_joint_speed,
                    torch.where(active[:, None], joint_speed, peak_joint_speed),
                )
                if active[0]:
                    trace.append({
                        "step": step, "time_s": step * .02,
                        "body_x": float(env.last_body_x[0]), "body_z": float(env.last_body_z[0]),
                        "quaternion_wxyz": env.last_quat[0].cpu().tolist(),
                        "upright": float(env.last_up[0]),
                        "wheel_x": env.last_wheel_x[0].cpu().tolist(),
                        "wheel_z": env.last_wheel_z[0].cpu().tolist(),
                        "cargo_accel_mps2": float(env.last_cargo_accel_norm[0]),
                        "stance_weights": env.last_stance_weights[0].cpu().tolist(),
                        "support_force_N": float(env.last_support_force[0]),
                        "joint_position": env.last_joint_position[0].cpu().tolist(),
                        "joint_velocity": env.last_joint_velocity[0].cpu().tolist(),
                        "joint_target": env.last_joint_target[0].cpu().tolist(),
                        "joint_torque": env.last_joint_torque[0].cpu().tolist(),
                        "action": action[0].cpu().tolist(),
                        "observation": observation[0].cpu().tolist(),
                    })
                success |= env.last_success & active
                failed |= (env.last_fall | env.last_safety_violation) & active
                joint_failed |= env.last_unsafe_joint & active
                joint_position_failed |= ((joint_utilization[:, NON_WHEEL] >= .96).any(-1)
                                          & active)
                joint_speed_failed |= ((joint_speed[:, NON_WHEEL] > 12.).any(-1) & active)
                tilt_failed |= env.last_unsafe_tilt & active
                timed_out |= info["time_outs"] & active
                ended = env.last_done & active
                done_step = torch.where(ended, torch.full_like(done_step, step), done_step)
                terminal_min_wheel_x = torch.where(
                    ended, env.last_wheel_x.amin(-1), terminal_min_wheel_x
                )
                active &= ~env.last_done
                if not active.any():
                    break
        safe = success & ~failed & (min_up >= math.cos(math.radians(15.)))
        cargo_rms = torch.sqrt(cargo_square / samples.clamp(min=1.))
        timeout_remaining = goal_x[timed_out] - terminal_min_wheel_x[timed_out]
        mean_abs_action = action_abs_sum / samples.clamp(min=1.)[:, None]
        report = {
            "actor": str(args.actor.resolve()), "seed": args.seed,
            "terrain": "isolated-step" if args.isolated_step else "continuous-stairs",
            "rise_mm": args.rise_mm, "goal_edges": args.goal_edges, "episodes": args.envs,
            "successes": int(success.sum()), "safe15_successes": int(safe.sum()),
            "failures": int(failed.sum()), "timeouts": int(timed_out.sum()),
            "joint_limit_failures": int(joint_failed.sum()),
            "joint_position_failures": int(joint_position_failed.sum()),
            "joint_speed_failures": int(joint_speed_failed.sum()),
            "tilt_failures": int(tilt_failed.sum()),
            "min_upright": float(min_up.min()), "mean_min_upright": float(min_up.mean()),
            "max_side_progress_error_m": float(max_side_error.max()),
            "cargo_accel_rms_mean": float(cargo_rms.mean()),
            "cargo_accel_rms_max": float(cargo_rms.max()),
            "completion_step_mean": float(done_step[success].float().mean()) if success.any() else None,
            "timeout_goal_remaining_m": {
                "min": float(timeout_remaining.min()) if timeout_remaining.numel() else None,
                "mean": float(timeout_remaining.mean()) if timeout_remaining.numel() else None,
                "max": float(timeout_remaining.max()) if timeout_remaining.numel() else None,
            },
            "mean_abs_action": mean_abs_action.mean(0).cpu().tolist(),
            "peak_joint_utilization_by_joint": {
                name: float(value)
                for name, value in zip(JOINT_NAMES, peak_joint_utilization.amax(0).cpu().tolist())
            },
            "peak_joint_speed_rad_s_by_joint": {
                name: float(value)
                for name, value in zip(JOINT_NAMES, peak_joint_speed.amax(0).cpu().tolist())
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "trace-env0.json").write_text(json.dumps(trace, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return 0 if int(safe.sum()) == args.envs else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
