"""Throughput bench: sim2sim-bench-godot --workers 1 4 8 16 --steps 500."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sim2sim.paths import sim2sim_root
from sim2sim.train.vec_env import GodotVecEnv, load_walk_cfg


def _free_g() -> str:
    try:
        out = subprocess.check_output(["free", "-g"], text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        return f"(free -g failed: {e})"
    return out.strip()


def main(argv: list[str] | None = None) -> int:
    root = sim2sim_root()
    p = argparse.ArgumentParser(prog="sim2sim-bench-godot")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--config", type=Path, default=root / "configs/walk_godot.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=root / "results/bench_godot.json")
    args = p.parse_args(argv)

    cfg = load_walk_cfg(args.config)
    cfg["obs_noise"]["enabled"] = False
    cfg["pushes"]["enabled"] = False
    cfg["faults_jsonl"] = str(root / "results/bench_faults.jsonl")

    print("=== mem before ===")
    print(_free_g())

    rows: list[dict] = []
    for n in args.workers:
        n = int(n)
        print(f"\n=== workers={n} steps={args.steps} ===")
        env = None
        try:
            env = GodotVecEnv(cfg, num_envs=n, device=args.device, seed=args.seed, headless=True)
            stats = env.bench_step_rate(args.steps)
        except Exception as e:
            stats = {
                "num_envs": float(n),
                "n_steps": float(args.steps),
                "elapsed_s": None,
                "steps_per_s": None,
                "latency_s": None,
                "faults": None,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"FAIL {stats['error']}")
        finally:
            if env is not None:
                env.close()
        rows.append(stats)
        if stats.get("steps_per_s") is not None:
            print(
                f"aggregate={stats['steps_per_s']:.1f} steps/s  "
                f"latency={1000.0 * stats['latency_s']:.2f} ms/vec-step  "
                f"faults={int(stats['faults'])}"
            )

    print("\n=== mem after ===")
    print(_free_g())

    print("\nworkers  steps/s  latency_ms  faults")
    for r in rows:
        if r.get("steps_per_s") is None:
            print(f"{int(r['num_envs']):7d}  FAIL  {r.get('error')}")
        else:
            print(
                f"{int(r['num_envs']):7d}  {r['steps_per_s']:7.1f}  "
                f"{1000.0 * r['latency_s']:10.2f}  {int(r['faults'])}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workers": [int(x) for x in args.workers],
        "steps": int(args.steps),
        "device": str(args.device),
        "rows": rows,
        "mem_after": _free_g(),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
