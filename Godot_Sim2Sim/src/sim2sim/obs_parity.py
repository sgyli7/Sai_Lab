"""Parity: MujocoBackend obs vs official infer_policy.PolicyInference."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from sim2sim.backends.mujoco_backend import MujocoBackend
from sim2sim.obs import DEFAULT_HOME, build_obs, command_13
from sim2sim.paths import microduck_rl, policies_dir
from sim2sim.runner import apply_home_qpos

MICRODUCK_RL = microduck_rl()
ONNX = policies_dir() / "alpha_walking.onnx"
MJCF = MICRODUCK_RL / "src/mjlab_microduck/robot/microduck/scene.xml"


def load_infer():
    path = MICRODUCK_RL / "scripts/infer_policy.py"
    spec = importlib.util.spec_from_file_location("infer_policy", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, default=ONNX)
    args = p.parse_args(argv)
    be = MujocoBackend(MJCF, timestep=0.005, current_limit_a=1.75)
    apply_home_qpos(be, DEFAULT_HOME, z=0.125)
    be.reset(qpos=be.data.qpos.copy(), qvel=be.data.qvel.copy(), ctrl=DEFAULT_HOME)
    mod = load_infer()
    pol = mod.PolicyInference(
        be.model,
        be.data,
        walking_onnx_path=str(args.onnx),
        action_scale=1.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )
    pol.vel_cmd[:] = [0.25, 0.0, 0.0]
    pol._update_command()
    obs_ref = pol.get_observations()
    st = be._state()
    obs = build_obs(st, pol.last_action, pol.command, home=pol.default_pose)
    if obs.shape != obs_ref.shape:
        print("SHAPE", obs.shape, obs_ref.shape)
        return 1
    err = np.max(np.abs(obs - obs_ref))
    print(f"obs_parity max_abs_err={err:.6e} dim={obs.size}")
    if err > 1e-5:
        print("ours", obs)
        print("ref ", obs_ref)
        return 1
    # one control step
    action = pol.infer()
    pol.apply_action(action)
    for _ in range(4):
        be.step(be.data.ctrl.copy(), n_substeps=1)
    # PolicyInference.apply_action already wrote ctrl; mj_step via backend used same data
    obs_ref2 = pol.get_observations()
    st2 = be._state()
    obs2 = build_obs(st2, pol.last_action, pol.command, home=pol.default_pose)
    err2 = np.max(np.abs(obs2 - obs_ref2))
    print(f"obs_parity after_step max_abs_err={err2:.6e}")
    be.close()
    return 0 if err2 <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
