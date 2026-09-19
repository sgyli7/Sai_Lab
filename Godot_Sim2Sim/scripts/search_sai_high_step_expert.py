#!/usr/bin/env python3
"""Discover a bounded high-step task-space expert with parallel MuJoCo CEM.

The resulting spline is privileged training data, not a deployment controller.
It exists to give the phase-free v6 actor successful trajectories without
copying the legacy periodic stair gait.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from train_sai_stair_task_v6 import SaiStairWorld
from sim2sim.sai_stair_v6 import STAIR_CLEARANCE_M


PARAMETERS = 9


def seed_mean(knots: int, device: str) -> torch.Tensor:
    """A neutral symmetric motion seed; CEM owns the final trajectory."""
    mean = torch.zeros((knots, PARAMETERS), device=device)
    # Keep wheel authority during the maneuver and let CEM discover endpoint
    # and body motion.  This is not a per-leg lift schedule.
    mean[:, 4:6] = .25
    mean[:, 8] = .75
    mean[0:2, 6] = -.20
    mean[2:5, 6] = .20
    return mean


def center_reset(env: SaiStairWorld) -> None:
    """Remove approach pose noise for initial skill discovery."""
    ids = torch.arange(env.num_envs, device="cuda")
    lane = env.lane
    x = env.start_x.clone()
    y = env.lane_centers[lane]
    env.qpos[:, env.qa] = x
    env.qpos[:, env.qa + 1] = y
    env.qpos[:, env.qa + 3] = 1.
    env.qpos[:, env.qa + 4:env.qa + 7] = 0.
    env.qvel[:, env.va:env.va + 6] = 0.
    env.qpos[:, env.payload_qa] = x - .09
    env.qpos[:, env.payload_qa + 1] = y
    if env.payload_clamp:
        env.qpos[:, env.payload_qa + 3] = np.cos(np.pi / 4)
        env.qpos[:, env.payload_qa + 4:env.payload_qa + 6] = 0.
        env.qpos[:, env.payload_qa + 6] = np.sin(np.pi / 4)
    else:
        env.qpos[:, env.payload_qa + 3] = 1.
        env.qpos[:, env.payload_qa + 4:env.payload_qa + 7] = 0.
    env._sync_forward()
    wheel = wp.to_torch(env.wd.xpos)[:, env.wheel_bodies]
    ground = env.terrain_height(wheel[:, :, 0])
    env.height_reference.copy_(ground.mean(-1) + .2192)
    env.last_payload_velocity.copy_(env.qvel[:, env.payload_va:env.payload_va + 3])
    goal_x = (env.starts[lane] + (env.goal_edges - 1) * env.treads[lane]
              + STAIR_CLEARANCE_M)
    env.last_goal_distance.copy_((goal_x - x).clamp(min=0.))


def expand(parameters: torch.Tensor) -> torch.Tensor:
    """Map symmetric task parameters to the v6 16-dimensional action."""
    result = torch.zeros((len(parameters), 16), device=parameters.device)
    result[:, 0:2] = parameters[:, 0:1]
    result[:, 2:4] = parameters[:, 1:2]
    result[:, 4:6] = parameters[:, 2:3]
    result[:, 6:8] = parameters[:, 3:4]
    result[:, 8:10] = parameters[:, 4:5]
    result[:, 10:12] = parameters[:, 5:6]
    result[:, 12] = parameters[:, 6]
    result[:, 13] = 0.
    result[:, 14] = parameters[:, 7]
    result[:, 15] = 1.
    return result


def actions_at(splines: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    knots = splines.shape[1]
    position = phase.clamp(0., 1.) * (knots - 1)
    left = position.floor().long().clamp(0, knots - 1)
    right = (left + 1).clamp(max=knots - 1)
    blend = (position - left).unsqueeze(1)
    ids = torch.arange(len(splines), device=splines.device)
    values = splines[ids, left] * (1. - blend) + splines[ids, right] * blend
    return expand(values)


def evaluate(env: SaiStairWorld, splines: torch.Tensor, horizon: int,
             activation_s: float) -> dict[str, torch.Tensor]:
    count = len(splines)
    device = splines.device
    active = torch.ones(count, device=device, dtype=torch.bool)
    entered = torch.zeros_like(active)
    phase_steps = torch.zeros(count, device=device)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    returns = torch.zeros(count, device=device)
    cargo_square = torch.zeros(count, device=device)
    cargo_samples = torch.zeros(count, device=device)
    min_up = torch.ones(count, device=device)
    max_x = env.qpos[:, env.qa].clone()
    initial_x = max_x.clone()
    wheel_x = wp.to_torch(env.wd.xpos)[:, env.wheel_bodies, 0].clone()
    initial_front = wheel_x[:, :2].amin(-1)
    initial_rear = wheel_x[:, 2:].amin(-1)
    max_front = initial_front.clone()
    max_rear = initial_rear.clone()
    previous_action = torch.zeros((count, 16), device=device)
    action_variation = torch.zeros(count, device=device)
    edge = env.starts[env.lane]
    activation_x = edge - .30
    for _ in range(horizon):
        current_x = env.qpos[:, env.qa]
        entered |= current_x >= activation_x
        phase_steps += entered & active
        phase = phase_steps / max(activation_s * 50., 1.)
        action = actions_at(splines, phase)
        action = torch.where((entered & active)[:, None], action, torch.zeros_like(action))
        action_variation += (action - previous_action).square().sum(-1) * active
        previous_action = action
        _, reward, _, _ = env.step(action)
        returns += reward * active
        cargo_square += env.last_cargo_accel_norm.square() * active
        cargo_samples += active
        min_up = torch.minimum(min_up, torch.where(active, env.last_up, min_up))
        max_x = torch.maximum(max_x, torch.where(active, env.last_body_x, max_x))
        front = env.last_wheel_x[:, :2].amin(-1)
        rear = env.last_wheel_x[:, 2:].amin(-1)
        max_front = torch.maximum(max_front, torch.where(active, front, max_front))
        max_rear = torch.maximum(max_rear, torch.where(active, rear, max_rear))
        success |= env.last_success & active
        failed |= (env.last_fall | env.last_safety_violation) & active
        active &= ~env.last_done
        if not active.any():
            break
    cargo_rms = torch.sqrt(cargo_square / cargo_samples.clamp(min=1.))
    progress = max_x - initial_x
    front_progress = max_front - initial_front
    rear_progress = max_rear - initial_rear
    front_crossed = max_front > edge + .02
    rear_crossed = max_rear > edge + .02
    # Completion dominates.  Before the first success, the same score provides
    # a continuous slope toward moving the complete chassis past the edge.
    fitness = (600. * success.float() + 70. * front_progress + 190. * rear_progress
               + 70. * front_crossed.float() + 180. * rear_crossed.float()
               + returns / cargo_samples.clamp(min=1.)
               - 35. * torch.relu(.966 - min_up)
               - 2.0 * cargo_rms
               - .002 * action_variation
               - 120. * failed.float())
    return dict(fitness=fitness, success=success, failed=failed, progress=progress,
                front_progress=front_progress, rear_progress=rear_progress,
                front_crossed=front_crossed, rear_crossed=rear_crossed,
                cargo_rms=cargo_rms, min_up=min_up, max_x=max_x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--elite", type=float, default=.12)
    parser.add_argument("--knots", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=340)
    parser.add_argument("--activation-seconds", type=float, default=3.2)
    parser.add_argument("--seed", type=int, default=260920)
    parser.add_argument("--initialize-expert", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    env = SaiStairWorld(args.population, args.seed, max_steps=args.horizon + 10,
                        min_rise_mm=40., terrain_stage="high", position_objective=True,
                        payload_clamp=True, domain_randomization=False,
                        cargo_accel_weight=.02, action_rate_weight=.08,
                        lift_event_weight=.2, support_weight=4., isolated_step=True)
    env.curriculum_count = 1
    env.curriculum_edges = 1
    env.reset(torch.arange(args.population, device="cuda"))
    mean = seed_mean(args.knots, "cuda")
    if args.initialize_expert is not None:
        source = json.loads(args.initialize_expert.read_text())
        loaded = torch.tensor(source["knots"], device="cuda", dtype=mean.dtype)
        if loaded.shape != mean.shape:
            raise ValueError(f"Expert knot shape mismatch: {loaded.shape} != {mean.shape}")
        mean.copy_(loaded)
    std = torch.full_like(mean, .45)
    elite_count = max(4, int(args.population * args.elite))
    best_fitness = -float("inf")
    best = mean.clone()
    history = []
    started = time.time()
    try:
        for iteration in range(1, args.iterations + 1):
            samples = (mean[None] + std[None] * torch.randn(
                (args.population, args.knots, PARAMETERS), device="cuda")).clamp(-1., 1.)
            # Search may brake during weight transfer, but the settling phase
            # must keep both axles driving forward instead of parking the rear
            # axle at the riser.
            samples[:, -2:, 4:6].clamp_(min=.15)
            samples[0] = best
            samples[0, -2:, 4:6].clamp_(min=.15)
            env.reset(torch.arange(args.population, device="cuda"))
            center_reset(env)
            report = evaluate(env, samples, args.horizon, args.activation_seconds)
            elite_ids = report["fitness"].topk(elite_count).indices
            elite = samples[elite_ids]
            elite_fitness = report["fitness"][elite_ids]
            weights = torch.softmax((elite_fitness - elite_fitness.max()) / 12., dim=0)
            mean = (elite * weights[:, None, None]).sum(0)
            variance = ((elite - mean) ** 2 * weights[:, None, None]).sum(0)
            std = (.72 * std + .28 * torch.sqrt(variance + .01)).clamp(.08, .65)
            top = int(elite_ids[0])
            top_fitness = float(report["fitness"][top])
            if top_fitness > best_fitness:
                best_fitness = top_fitness
                best = samples[top].clone()
            row = {
                "iteration": iteration, "elapsed_s": time.time() - started,
                "top_fitness": top_fitness, "best_fitness": best_fitness,
                "successes": int(report["success"].sum()),
                "top_success": bool(report["success"][top]),
                "top_failed": bool(report["failed"][top]),
                "top_progress_m": float(report["progress"][top]),
                "top_front_progress_m": float(report["front_progress"][top]),
                "top_rear_progress_m": float(report["rear_progress"][top]),
                "top_front_crossed": bool(report["front_crossed"][top]),
                "top_rear_crossed": bool(report["rear_crossed"][top]),
                "top_cargo_accel_rms": float(report["cargo_rms"][top]),
                "top_min_upright": float(report["min_up"][top]),
                "mean_std": float(std.mean()),
            }
            history.append(row)
            artifact = {
                "schema_version": 1, "kind": "privileged-task-space-spline-expert",
                "deployment": "forbidden; training data only", "control_hz": 50,
                "physics_hz": 500, "rise_m": .04, "goal_edges": 1,
                "terrain_layout": "isolated-step-long-landing",
                "activation_distance_m": .30, "activation_seconds": args.activation_seconds,
                "knots": best.detach().cpu().tolist(), "parameter_layout": [
                    "front_dx", "rear_dx", "front_dz", "rear_dz",
                    "front_wheel_speed", "rear_wheel_speed", "height", "pitch",
                    "reserved_intensity_logit"],
                "best_fitness": best_fitness, "history": history,
            }
            (args.output / "expert.json").write_text(json.dumps(artifact, indent=2) + "\n")
            (args.output / "progress.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps({"expert_search": row}), flush=True)
            if int(report["success"].sum()) >= max(6, elite_count // 2) and iteration >= 4:
                break
    finally:
        env.close()
    return 0 if any(row["successes"] for row in history) else 2


if __name__ == "__main__":
    raise SystemExit(main())
