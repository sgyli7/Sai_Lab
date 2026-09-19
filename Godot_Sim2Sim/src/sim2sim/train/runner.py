"""Godot/Jolt PPO fine-tune entry: ``sim2sim-train`` (rsl_rl 5.0.1 OnPolicyRunner)."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from sim2sim.paths import policies_dir, sim2sim_root
from sim2sim.train.manifest import git_snapshot
from sim2sim.train.onnx_import import (
    DEFAULT_FROZEN_COUNT,
    PARITY_FAIL_ABS,
    build_actor_state_dict,
    parse_mlp_onnx,
    verify_parity,
)
from sim2sim.train.vec_env import GodotVecEnv, load_walk_cfg

STEPS_LO = 24
STEPS_HI = 512
DEFAULT_SAMPLES_PER_ITER = 2048
PPO_CLASS = "sim2sim.train.ppo_finetune:PPOFinetune"
ONNX_ALIASES = {"alpha": "alpha_walking.onnx", "alpha_walking": "alpha_walking.onnx"}
DEFAULT_TORCH_THREADS = 2


def configure_cpu_threads(n: int = DEFAULT_TORCH_THREADS) -> int:
    """Cap torch/OpenMP threads so they do not contend with core-pinned Godot workers."""
    import os

    raw = os.environ.get("SIM2SIM_TORCH_THREADS")
    n = int(raw) if raw else int(n)
    n = max(1, n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return int(torch.get_num_threads())


class InitParityError(RuntimeError):
    """Actor vs ONNX mean-action disagreement exceeds the init threshold."""


def choose_num_steps_per_env(num_envs: int, samples_per_iter: int) -> int:
    """Pick rollout length so ``num_envs * steps ≈ samples_per_iter``, clamped to [24, 512]."""
    nenv = max(1, int(num_envs))
    target = int(round(int(samples_per_iter) / nenv))
    return max(STEPS_LO, min(STEPS_HI, target))


def resolve_onnx_path(spec: str | Path) -> Path:
    """Resolve ``alpha`` / a filename / an absolute path to an ONNX file."""
    raw = str(spec)
    name = ONNX_ALIASES.get(raw, raw)
    candidates: list[Path] = []
    p = Path(name)
    if p.is_file():
        return p.resolve()
    candidates.append(p)
    pol = policies_dir()
    candidates.append(pol / name)
    candidates.append(pol / Path(name).name)
    candidates.append(Path("/home/ethan/Projects/MicroDuck/policies") / Path(name).name)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"ONNX not found: {spec}")


def build_train_cfg(
    ppo: Mapping[str, Any],
    *,
    num_envs: int,
    samples_per_iter: int,
    save_interval: int,
    seed: int,
    run_name: str,
    max_iterations: int,
) -> dict[str, Any]:
    """rsl_rl 5.0.1 ``OnPolicyRunner`` train_cfg from the yaml ``ppo:`` block."""
    hidden = [int(x) for x in (ppo.get("hidden_dims") or [512, 256, 128])]
    activation = str(ppo.get("activation") or "elu")
    init_std = float(ppo.get("init_std", 0.25))
    obs_norm = bool(ppo.get("obs_normalization", True))
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": hidden,
        "activation": activation,
        "obs_normalization": obs_norm,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": init_std,
            "std_type": "scalar",
        },
    }
    critic = {
        "class_name": "MLPModel",
        "hidden_dims": list(hidden),
        "activation": activation,
        "obs_normalization": obs_norm,
    }
    algorithm = {
        "class_name": PPO_CLASS,
        "num_learning_epochs": int(ppo.get("num_learning_epochs", 5)),
        "num_mini_batches": int(ppo.get("num_mini_batches", 4)),
        "learning_rate": float(ppo.get("learning_rate", 3.0e-4)),
        "schedule": str(ppo.get("schedule", "adaptive")),
        "gamma": float(ppo.get("gamma", 0.99)),
        "lam": float(ppo.get("lam", 0.95)),
        "entropy_coef": float(ppo.get("entropy_coef", 0.001)),
        "desired_kl": float(ppo.get("desired_kl", 0.01)),
        "max_grad_norm": float(ppo.get("max_grad_norm", 1.0)),
        "value_loss_coef": float(ppo.get("value_loss_coef", 1.0)),
        "use_clipped_value_loss": bool(ppo.get("use_clipped_value_loss", True)),
        "clip_param": float(ppo.get("clip_param", 0.2)),
        "normalize_advantage_per_mini_batch": bool(ppo.get("normalize_advantage_per_mini_batch", False)),
        "optimizer": str(ppo.get("optimizer", "adam")),
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    }
    return {
        "seed": int(seed),
        "num_steps_per_env": choose_num_steps_per_env(num_envs, samples_per_iter),
        "max_iterations": int(max_iterations),
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "save_interval": max(1, int(save_interval)),
        "experiment_name": "walk_godot",
        "run_name": str(run_name),
        "logger": "tensorboard",
        "resume": False,
        "clip_actions": None,
        "class_name": "OnPolicyRunner",
        "actor": actor,
        "critic": critic,
        "algorithm": algorithm,
        "check_for_nan": True,
    }


def set_actor_trainable(actor: Any, trainable: bool) -> None:
    """Enable/disable grads on actor MLP + Gaussian std (not the normalizer buffers)."""
    for p in actor.mlp.parameters():
        p.requires_grad = bool(trainable)
    dist = getattr(actor, "distribution", None)
    if dist is not None:
        for p in dist.parameters():
            p.requires_grad = bool(trainable)


def freeze_actor_normalizer(actor: Any) -> None:
    """Stop EmpiricalNormalization updates by setting ``until`` so ``count >= until``."""
    norm = getattr(actor, "obs_normalizer", None)
    if norm is None or not hasattr(norm, "until") or not hasattr(norm, "count"):
        return
    count = int(norm.count.detach().cpu().item())
    norm.until = count


def snapshot_normalizer_mean(actor: Any) -> Any:
    import torch

    norm = actor.obs_normalizer
    return norm._mean.detach().cpu().clone()


def normalizer_mean_equal(actor: Any, snapshot: Any) -> bool:
    import torch

    return bool(torch.equal(actor.obs_normalizer._mean.detach().cpu(), snapshot.cpu()))


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_yaml_clean(data), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _yaml_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _yaml_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def _write_git_txt(path: Path) -> None:
    snap = git_snapshot()
    status = ""
    try:
        import subprocess

        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sim2sim_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        status = r.stdout or ""
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"commit: {snap.get('commit')}\n"
        f"branch: {snap.get('branch')}\n"
        f"dirty: {snap.get('dirty')}\n"
        f"--- status ---\n{status}",
        encoding="utf-8",
    )


def _critic_compatible(critic: Any, state_dict: Mapping[str, Any]) -> bool:
    cur = critic.state_dict()
    for key, tensor in cur.items():
        if key not in state_dict:
            return False
        other = state_dict[key]
        if tuple(getattr(other, "shape", ())) != tuple(tensor.shape):
            return False
    return True


def _snapshot_godot_pids(env: Any) -> list[int]:
    pids: list[int] = []
    for w in getattr(env, "_workers", []) or []:
        if w is None:
            continue
        proc = getattr(w, "_proc", None)
        if proc is None:
            continue
        pid = getattr(proc, "pid", None)
        if pid:
            pids.append(int(pid))
    return pids


def _reap_pids(pids: list[int], *, log=print) -> None:
    import os as _os

    alive: list[int] = []
    for pid in pids:
        try:
            _os.kill(pid, 0)
        except OSError:
            continue
        alive.append(pid)
        try:
            _os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if not alive:
        return
    deadline = time.time() + 3.0
    leftover: list[int] = []
    for pid in alive:
        while time.time() < deadline:
            try:
                _os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:
            leftover.append(pid)
    if leftover:
        log(f"WARNING: stray godot pids after close: {leftover}")


def _device_str(device: str) -> str:
    import torch

    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _load_init_onnx_into_actor(actor: Any, onnx_path: Path, *, init_std: float, count: int) -> None:
    rec = parse_mlp_onnx(onnx_path)
    sd = build_actor_state_dict(rec, init_std=init_std, count=count)
    actor.load_state_dict(sd, strict=True)


def _run_parity(actor: Any, onnx_path: Path, n: int = 10000) -> float:
    return float(verify_parity(actor, onnx_path, n=n, seed=0))


def _write_init_check(path: Path, *, onnx: Path, err: float, n: int) -> dict[str, Any]:
    payload = {
        "onnx": str(onnx),
        "n": int(n),
        "max_abs_err": float(err),
        "threshold": float(PARITY_FAIL_ABS),
        "ok": bool(err < PARITY_FAIL_ABS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train(
    config: str | Path | Mapping[str, Any],
    *,
    init_onnx: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    init_onnx_ref: str | Path | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    run_name: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    log_root: str | Path | None = None,
    samples_per_iter: int | None = None,
    critic_warmup_iters: int | None = None,
    save_interval: int | None = None,
    export_onnx: str | Path | None = None,
    log=print,
) -> Path:
    """Build GodotVecEnv + OnPolicyRunner, train, return the run directory."""
    n_threads = configure_cpu_threads()
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils import check_nan

    n_modes = sum(x is not None for x in (init_onnx, init_checkpoint, resume))
    if n_modes > 1:
        raise ValueError("exactly one of --init-onnx / --init-checkpoint / --resume")

    cfg = load_walk_cfg(config)
    ppo = dict(cfg.get("ppo") or {})
    train_block = dict(cfg.get("train") or {})

    nenv = int(num_envs if num_envs is not None else cfg.get("num_envs", 8))
    seed_i = int(seed if seed is not None else cfg.get("seed", 0))
    device_s = _device_str(str(device if device is not None else cfg.get("device", "cpu")))
    run = str(run_name if run_name is not None else train_block.get("run_name") or "walk")
    max_it = int(
        max_iterations
        if max_iterations is not None
        else train_block.get("max_iterations") or ppo.get("max_iterations") or 3000
    )
    samples = int(
        samples_per_iter
        if samples_per_iter is not None
        else train_block.get("samples_per_iter")
        or ppo.get("samples_per_iter")
        or DEFAULT_SAMPLES_PER_ITER
    )
    save_every = int(
        save_interval
        if save_interval is not None
        else train_block.get("save_interval") or ppo.get("save_interval") or 50
    )
    warmup = int(
        critic_warmup_iters
        if critic_warmup_iters is not None
        else ppo.get("critic_warmup_iters") or train_block.get("critic_warmup_iters") or 0
    )
    init_std = float(ppo.get("init_std", 0.25))
    unfreeze_lr = float(ppo.get("unfreeze_learning_rate", 3.0e-5))

    if n_modes == 0:
        init_onnx = cfg.get("init_onnx")

    root = sim2sim_root()
    log_root_s = str(log_root if log_root is not None else train_block.get("log_root") or "logs/walk_godot")
    log_root_p = Path(log_root_s)
    if not log_root_p.is_absolute():
        log_root_p = root / log_root_p
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root_p / f"{stamp}_{run}"
    log_dir.mkdir(parents=True, exist_ok=True)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    train_log = log_dir / "train.log"
    metrics_jsonl = log_dir / "metrics.jsonl"

    cfg["num_envs"] = nenv
    cfg["seed"] = seed_i
    cfg["device"] = device_s
    cfg["faults_jsonl"] = str(log_dir / "faults.jsonl")
    _dump_yaml(params_dir / "env.yaml", cfg)
    _write_git_txt(params_dir / "git.txt")

    train_cfg = build_train_cfg(
        ppo,
        num_envs=nenv,
        samples_per_iter=samples,
        save_interval=save_every,
        seed=seed_i,
        run_name=run,
        max_iterations=max_it,
    )
    _dump_yaml(params_dir / "agent.yaml", train_cfg)
    runner_cfg = copy.deepcopy(train_cfg)

    mode = "resume" if resume is not None else ("checkpoint" if init_checkpoint is not None else "onnx")
    start_line = (
        f"start mode={mode} num_envs={nenv} steps/env={train_cfg['num_steps_per_env']} "
        f"samples/iter~{nenv * train_cfg['num_steps_per_env']} max_it={max_it} "
        f"warmup={warmup} unfreeze_lr={unfreeze_lr:.3e} torch_threads={n_threads} "
        f"device={device_s} log_dir={log_dir}"
    )
    _append_line(train_log, start_line)
    log(start_line)

    env: GodotVecEnv | None = None
    pids: list[int] = []
    stop = [False]
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_stop(signum, _frame) -> None:
        stop[0] = True
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGINT, _on_stop)
        signal.signal(signal.SIGTERM, _on_stop)
        env = GodotVecEnv(cfg, num_envs=nenv, device=device_s, seed=seed_i, headless=True)
        pids = _snapshot_godot_pids(env)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=device_s)
        try:
            runner.add_git_repo_to_log(str(root / "pyproject.toml"))
        except Exception:
            pass

        verify_mean = None
        onnx_for_parity: Path | None = None
        if resume is not None:
            ckpt = Path(resume)
            if not ckpt.is_file():
                raise FileNotFoundError(f"--resume not found: {ckpt}")
            infos = runner.load(str(ckpt), map_location=device_s)
            msg = f"resume {ckpt} iter={runner.current_learning_iteration} infos={infos}"
            _append_line(train_log, msg)
            log(msg)
            start_it = int(runner.current_learning_iteration) + 1
        elif init_checkpoint is not None:
            ckpt = Path(init_checkpoint)
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "actor_state_dict" not in payload:
                raise ValueError(f"{ckpt} is not an rsl_rl checkpoint")
            runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
            freeze_actor_normalizer(runner.alg.actor)
            verify_mean = snapshot_normalizer_mean(runner.alg.actor)
            csd = payload.get("critic_state_dict")
            if isinstance(csd, dict) and _critic_compatible(runner.alg.critic, csd):
                runner.alg.critic.load_state_dict(csd, strict=True)
                msg = f"init-checkpoint {ckpt}: actor+critic loaded"
            else:
                msg = (
                    f"init-checkpoint {ckpt}: actor loaded; critic shapes mismatch "
                    f"(kept freshly initialized critic)"
                )
            _append_line(train_log, msg)
            log(msg)
            if init_onnx_ref is not None:
                onnx_for_parity = resolve_onnx_path(init_onnx_ref)
            runner.current_learning_iteration = 0
            start_it = 0
        else:
            onnx_for_parity = resolve_onnx_path(init_onnx or cfg.get("init_onnx"))
            _load_init_onnx_into_actor(
                runner.alg.actor,
                onnx_for_parity,
                init_std=init_std,
                count=DEFAULT_FROZEN_COUNT,
            )
            freeze_actor_normalizer(runner.alg.actor)
            verify_mean = snapshot_normalizer_mean(runner.alg.actor)
            runner.current_learning_iteration = 0
            start_it = 0
            msg = f"init-onnx {onnx_for_parity} init_std={init_std} actor-normalizer frozen"
            _append_line(train_log, msg)
            log(msg)

        if onnx_for_parity is not None:
            err = _run_parity(runner.alg.actor, onnx_for_parity, n=10000)
            payload = _write_init_check(params_dir / "init_check.json", onnx=onnx_for_parity, err=err, n=10000)
            line = f"init_check max_abs_err={err:.6e} threshold={PARITY_FAIL_ABS} ok={payload['ok']}"
            _append_line(train_log, line)
            log(line)
            if not payload["ok"]:
                raise InitParityError(line)

        last_path = _learn_loop(
            runner,
            env,
            start_it=start_it,
            num_iterations=max_it,
            warmup_iters=warmup,
            unfreeze_lr=unfreeze_lr,
            prime_unfreeze=(resume is None),
            train_log=train_log,
            metrics_jsonl=metrics_jsonl,
            stop=stop,
            verify_mean=verify_mean,
            log=log,
        )
        if export_onnx is None:
            export_onnx = (cfg.get("export") or {}).get("onnx")
        if export_onnx is not None and last_path is not None:
            from sim2sim.train.export import DEFAULT_DESCRIPTION, export_actor, load_rsl_checkpoint_actor

            exp = dict(cfg.get("export") or {})
            sd, meta = load_rsl_checkpoint_actor(last_path)
            export_actor(
                state_dict=sd,
                source=str(last_path),
                out=Path(export_onnx),
                checkpoint=meta.get("iter"),
                run=run,
                description=str(exp.get("description") or DEFAULT_DESCRIPTION),
                name=str(exp.get("name") or "walk_godot"),
                kind=str(exp.get("kind") or "perpetual"),
                slot=str(exp.get("slot") or "walk"),
                use_stand_policy=bool(exp.get("use_stand_policy", True)),
            )
        return log_dir
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if env is not None:
            try:
                pids = list(dict.fromkeys(pids + _snapshot_godot_pids(env)))
            except Exception:
                pass
            try:
                env.close()
            except Exception as e:
                log(f"env.close failed: {e}")
            _reap_pids(pids, log=log)


def _mean_or_nan(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def _learn_loop(
    runner: Any,
    env: GodotVecEnv,
    *,
    start_it: int,
    num_iterations: int,
    warmup_iters: int,
    unfreeze_lr: float,
    prime_unfreeze: bool,
    train_log: Path,
    metrics_jsonl: Path,
    stop: list[bool],
    verify_mean: Any | None,
    log=print,
) -> Path | None:
    import os
    import torch
    from rsl_rl.utils import check_nan

    cfg = runner.cfg
    device = runner.device
    alg = runner.alg
    logger = runner.logger
    nenv = int(env.num_envs)
    nsteps = int(cfg["num_steps_per_env"])
    save_every = int(cfg["save_interval"])
    log_dir = Path(logger.log_dir)
    same_device = str(device) == str(env.device) or (
        getattr(device, "type", None) == "cpu" and getattr(env.device, "type", None) == "cpu"
    )

    obs = env.get_observations()
    if not same_device:
        obs = obs.to(device)
    alg.train_mode()
    logger.init_logging_writer()

    total_it = start_it + int(num_iterations)
    last_path: Path | None = None
    completed: int | None = None
    writer = logger.writer
    primed = not bool(prime_unfreeze)
    if start_it >= int(warmup_iters) and hasattr(alg, "_lr_guard"):
        # Resume past warmup never calls mark_unfreeze; keep per-iter KL adapt on.
        alg._lr_guard = True
        alg.min_learning_rate = 1e-5
        alg.max_learning_rate = max(float(getattr(alg, "_base_lr", 3.0e-4)), float(unfreeze_lr))
    profile_collect = os.environ.get("SIM2SIM_PROFILE_COLLECT", "").strip() not in ("", "0", "false", "False")

    def _save(it: int) -> Path:
        nonlocal last_path
        runner.current_learning_iteration = int(it)
        path = log_dir / f"model_{it}.pt"
        runner.save(str(path), infos={"iter": int(it)})
        last_path = path
        return path

    try:
        for it in range(start_it, total_it):
            if stop[0]:
                break
            frozen = bool(it < int(warmup_iters))
            set_actor_trainable(alg.actor, trainable=not frozen)
            if (not frozen) and (not primed) and hasattr(alg, "mark_unfreeze"):
                alg.mark_unfreeze(unfreeze_lr)
                primed = True
                std0 = getattr(alg, "_unfreeze_std_before", None)
                msg = f"unfreeze iter={it} lr={float(alg.learning_rate):.3e} actor_std={std0}"
                _append_line(train_log, msg)
                log(msg)
            cur_vals = env.set_curriculum(it)

            t0 = time.time()
            step_rew_sum = 0.0
            fall_events = 0.0
            term_sum: dict[str, float] = {}
            n_term = 0

            def _collect() -> None:
                nonlocal obs, step_rew_sum, fall_events, n_term
                for _ in range(nsteps):
                    actions = alg.act(obs)
                    step_in = actions if same_device else actions.to(env.device)
                    obs, rewards, dones, extras = env.step(step_in)
                    if not same_device:
                        obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                    if cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    alg.process_env_step(obs, rewards, dones, extras)
                    logger.process_env_step(rewards, dones, extras, None)
                    elog = extras.get("log") or {}
                    step_rew_sum += float(elog.get("step_reward", 0.0))
                    fall_events += float(elog.get("Episode_Termination/fell", 0.0)) * nenv
                    n_term += 1
                    for k, v in elog.items():
                        term_sum[k] = term_sum.get(k, 0.0) + float(v)

            with torch.inference_mode():
                if profile_collect and it == start_it:
                    import cProfile
                    import pstats

                    prof = cProfile.Profile()
                    prof.enable()
                    _collect()
                    prof.disable()
                    stats_path = log_dir / "collect.prof"
                    prof.dump_stats(str(stats_path))
                    pstats.Stats(prof).sort_stats("cumtime").print_stats(25)
                    _append_line(train_log, f"cProfile collect dumped {stats_path}")
                else:
                    _collect()
                collect_time = time.time() - t0
                t1 = time.time()
                alg.compute_returns(obs)

            loss_dict = alg.update()
            learn_time = time.time() - t1
            runner.current_learning_iteration = it
            completed = it

            if verify_mean is not None and it == start_it:
                if not normalizer_mean_equal(alg.actor, verify_mean):
                    raise RuntimeError(
                        "actor obs_normalizer._mean changed after a training iteration "
                        "(freeze failed)"
                    )
                _append_line(train_log, "actor_normalizer _mean bit-identical after first iter")

            std = None
            try:
                std = alg.get_policy().output_std
            except (AttributeError, TypeError):
                std = None
            if std is not None:
                std_mean = float(std.mean().item())
            else:
                from sim2sim.train.ppo_finetune import _actor_std_mean

                std_mean = float(_actor_std_mean(alg.get_policy()) or float("nan"))
            lr = float(alg.learning_rate)
            fps = (nsteps * nenv) / max(collect_time + learn_time, 1e-9)
            ep_rew = _mean_or_nan(list(logger.rewbuffer))
            ep_len = _mean_or_nan(list(logger.lenbuffer))
            step_rew = step_rew_sum / max(n_term, 1)
            faults = int(getattr(env, "faults", 0))
            kl = float(loss_dict.get("kl", getattr(alg, "last_kl", 0.0)) or 0.0)
            kl_max = float(loss_dict.get("kl_max", kl) or kl)
            early_mb = int(loss_dict.get("early_stop_mb", 0) or 0)
            terms = {
                k: (term_sum[k] / max(n_term, 1))
                for k in term_sum
                if k.startswith("Episode_Reward/")
            }
            term_s = " ".join(f"{k.split('/', 1)[-1]}={v:.4f}" for k, v in terms.items())
            cur_s = " ".join(f"{k}={v:.4g}" for k, v in cur_vals.items())
            mb_kl = list(getattr(alg, "minibatch_kl", []) or [])
            mb_lr = list(getattr(alg, "minibatch_lr", []) or [])
            line = (
                f"iter={it} fps={fps:.1f} step_rew={step_rew:.4f} ep_rew={ep_rew:.4f} "
                f"ep_len={ep_len:.2f} falls={fall_events:.1f} faults={faults} "
                f"lr={lr:.3e} std={std_mean:.4f} kl={kl:.4f} kl_max={kl_max:.4f} "
                f"early_stop={early_mb} frozen={int(frozen)} "
                f"value={float(loss_dict.get('value', float('nan'))):.4f} "
                f"surrogate={float(loss_dict.get('surrogate', float('nan'))):.4f} "
                f"entropy={float(loss_dict.get('entropy', float('nan'))):.4f} {term_s}"
            ).rstrip()
            if cur_s:
                line = f"{line} cur[{cur_s}]"
            _append_line(train_log, line)
            log(line)
            if (not frozen) and mb_kl and it == int(warmup_iters):
                mb_line = (
                    f"unfreeze_mb n={len(mb_kl)} kl={['%.4f' % x for x in mb_kl]} "
                    f"lr={['%.3e' % x for x in mb_lr]} "
                    f"std_before={getattr(alg, '_unfreeze_std_before', None)} "
                    f"std_after={getattr(alg, '_unfreeze_std_after', None)}"
                )
                _append_line(train_log, mb_line)
                log(mb_line)
            rec = {
                "iter": it,
                "fps": fps,
                "step_rew": step_rew,
                "ep_rew": ep_rew,
                "ep_len": ep_len,
                "falls": fall_events,
                "faults": faults,
                "lr": lr,
                "std": std_mean,
                "kl": kl,
                "kl_max": kl_max,
                "early_stop_mb": early_mb,
                "frozen": frozen,
                "collect_s": collect_time,
                "learn_s": learn_time,
                "losses": {k: float(v) for k, v in loss_dict.items()},
                "terms": terms,
                "curriculum": cur_vals,
            }
            if mb_kl:
                rec["minibatch_kl"] = mb_kl
                rec["minibatch_lr"] = mb_lr
            _append_line(metrics_jsonl, json.dumps(rec))

            logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=lr,
                action_std=std,
                rnd_weight=None,
            )
            if writer is not None:
                writer.add_scalar("Phase/actor_frozen", 1.0 if frozen else 0.0, it)
                writer.add_scalar("Train/mean_step_reward", step_rew, it)
                writer.add_scalar("Train/falls", fall_events, it)
                writer.add_scalar("Train/faults", float(faults), it)
                writer.add_scalar("Loss/kl", kl, it)
                writer.add_scalar("Loss/kl_max", kl_max, it)
                for ck, cv in cur_vals.items():
                    writer.add_scalar(f"Curriculum/{ck}", float(cv), it)

            if it % save_every == 0:
                _save(it)
    except KeyboardInterrupt:
        _append_line(train_log, f"SIGINT at iter={completed}")
        log(f"SIGINT: saving checkpoint (completed iter={completed})")
    finally:
        if completed is not None:
            last_path = _save(completed)
        elif start_it > 0:
            last_path = _save(start_it - 1)
        else:
            last_path = _save(0)
        try:
            logger.stop_logging_writer()
        except Exception:
            pass
        _append_line(train_log, f"saved {last_path}")
        log(f"saved {last_path}")
    return last_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sim2sim-train")
    p.add_argument("--config", type=Path, default=None, help="walk_godot.yaml (default: configs/walk_godot.yaml)")
    p.add_argument("--init-onnx", default=None, help="ONNX to recover actor from (default: config init_onnx / alpha)")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="rsl_rl model_k.pt (actor; critic if shapes match)")
    p.add_argument("--resume", type=Path, default=None, help="full runner.load of model_k.pt (optimizer+iter)")
    p.add_argument("--init-onnx-ref", default=None, help="optional ONNX for actor parity in --init-checkpoint")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None, help="cpu|cuda (default: config device, usually cpu)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-root", default=None)
    p.add_argument("--samples-per-iter", type=int, default=None)
    p.add_argument("--critic-warmup-iters", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        help="optional export of the final checkpoint (typically policies/Walk_Godot.onnx)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_cpu_threads()
    args = parse_args(argv)
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = sim2sim_root() / "configs/walk_godot.yaml"
    elif not Path(cfg_path).is_file():
        alt = sim2sim_root() / cfg_path
        cfg_path = alt if alt.is_file() else cfg_path
    try:
        train(
            cfg_path,
            init_onnx=args.init_onnx,
            init_checkpoint=args.init_checkpoint,
            resume=args.resume,
            init_onnx_ref=args.init_onnx_ref,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            run_name=args.run_name,
            device=args.device,
            seed=args.seed,
            log_root=args.log_root,
            samples_per_iter=args.samples_per_iter,
            critic_warmup_iters=args.critic_warmup_iters,
            save_interval=args.save_interval,
            export_onnx=args.export_onnx,
        )
    except InitParityError as e:
        print(f"INIT PARITY FAIL: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
