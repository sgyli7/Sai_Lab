"""Search *physical* MuJoCo pickup demonstrations for subsequent policy training.

This is a low dimensional teacher search, not a trained policy. Each candidate
plays the original body motion with bounded neck/head/yaw corrections and a
beak closing time. Only the environment's held out physical success predicate
qualifies an episode for imitation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim2sim.pickup_env import MAX_STEPS, PickupEnv
from sim2sim.policy import PolicyBundle


def rollout(env: PickupEnv, source: PolicyBundle, target: tuple[float, float],
            params: tuple[float, float, float, float], seed: int,
            capture: bool = False) -> tuple[dict, tuple[np.ndarray, np.ndarray] | None]:
    neck, head, yaw, close_s = params
    obs = env.reset(seed, x=target[0], lateral=target[1])
    seen, labels = [], []
    reward_sum = 0.
    for k in range(MAX_STEPS):
        base = source.infer(obs[:61])
        action = np.r_[base, 0.].astype(np.float32)
        action[5] += neck
        action[6] += head
        action[7] += yaw
        action[14] = .48 if k * .02 < close_s else 0.
        if capture:
            seen.append(obs.copy())
            labels.append(action.copy())
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            break
    result = {"target_x": target[0], "target_y": target[1], "seed": seed,
              "neck": neck, "head": head, "yaw": yaw, "close_s": close_s,
              "success": bool(info["success"]), "grip": env.grip_ever,
              "opposed_contact": env.opposed_contact_ever,
              "peak_lift_m": float(env.peak_lift),
              "stable_hold_s": float(info["stable_hold_s"]),
              "return": float(reward_sum), "events": env.events if capture else []}
    episode = (np.asarray(seen, np.float32), np.asarray(labels, np.float32)) if capture else None
    return result, episode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260919)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = PickupEnv(args.scene)
    source = PolicyBundle(args.source)
    rng = np.random.default_rng(args.seed)
    rows, observations, actions = [], [], []
    # Include a known viable pose so the search cannot finish with an empty
    # training set; all other attempts are randomized and graded identically.
    candidates = [((.07, 0.), (0., 0., 0., .46))]
    for _ in range(args.count-1):
        target = (float(rng.uniform(.045, .115)), float(rng.uniform(-.012, .012)))
        params = (float(rng.uniform(-.45, .25)), float(rng.uniform(-.40, .25)),
                  float(rng.uniform(-.18, .18)), float(rng.uniform(.30, .75)))
        candidates.append((target, params))
    for n, (target, params) in enumerate(candidates):
        row, _ = rollout(env, source, target, params, args.seed+n)
        rows.append(row)
        if row["success"]:
            verified, episode = rollout(env, source, target, params, args.seed+n, capture=True)
            if verified["success"] and episode is not None:
                observations.append(episode[0])
                actions.append(episode[1])
        if (n+1) % 100 == 0:
            print(json.dumps({"attempts": n+1, "physical_successes": sum(r["success"] for r in rows),
                              "opposed_contacts": sum(r["opposed_contact"] for r in rows)}), flush=True)
    (args.out / "search.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    if observations:
        np.savez_compressed(args.out / "demonstrations.npz",
                            obs=np.concatenate(observations), action=np.concatenate(actions),
                            episode_lengths=np.array([len(o) for o in observations], np.int32))
    manifest = {"schema_version": 1, "kind": "privileged_state_teacher_search",
                "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "attempts": len(rows), "successes": sum(r["success"] for r in rows),
                "recorded_demonstrations": len(observations), "seed": args.seed}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
