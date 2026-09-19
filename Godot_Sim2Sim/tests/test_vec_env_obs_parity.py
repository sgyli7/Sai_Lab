"""GodotVecEnv actor obs / fallen must match sim2sim.obs.build_obs and sim2sim.fall.fallen."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from sim2sim.coords import quat_rotate_inverse_wxyz
from sim2sim.fall import fallen as ref_fallen
from sim2sim.godot_proc import godot_bin
from sim2sim.obs import DEFAULT_HOME, build_obs
from sim2sim.paths import sim2sim_root

ROOT = sim2sim_root()
YAML = ROOT / "configs/walk_godot.yaml"
CFG_PATH = ROOT / "robots/microduck.json"
_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
PARITY_ABS = 1e-9


def _skip_reason() -> str | None:
    try:
        import torch  # noqa: F401
        import tensordict  # noqa: F401
    except ImportError:
        return "train extra not installed"
    from sim2sim.runner import load_robot_cfg

    cfg = load_robot_cfg(CFG_PATH)
    spec = Path(cfg["godot_spec"])
    if not spec.is_file():
        return f"missing spec {spec}"
    if not Path(godot_bin()).is_file():
        return f"missing godot {godot_bin()}"
    if not YAML.is_file():
        return f"missing {YAML}"
    return None


_SKIP = _skip_reason()


def _walk_cfg() -> dict:
    from sim2sim.train.vec_env import load_walk_cfg

    cfg = load_walk_cfg(YAML)
    cfg["obs_noise"]["enabled"] = False
    cfg["pushes"]["enabled"] = False
    cfg["recv_timeout_s"] = 15.0
    cfg["spawn_stagger_s"] = 0.1
    cfg["reset"] = {"yaw_range": [0.0, 0.0], "joint_noise_rad": 0.0}
    cfg["commands"] = {
        "resample_s": [100.0, 100.0],
        "vx": [0.15, 0.15],
        "vy": [0.0, 0.0],
        "wz": [0.0, 0.0],
        "standing_frac": 0.0,
        "turn_in_place_frac": 0.0,
    }
    cfg["faults_jsonl"] = str(Path("/tmp/sim2sim_obs_parity_faults.jsonl"))
    return cfg


def _rpy_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _assert_obs_parity(env, when: str) -> list[float]:
    from sim2sim.train.vec_env import NUM_ACTIONS

    actor = env.get_observations()["actor"].detach().cpu().numpy()
    states = env.debug_states()
    errs: list[float] = []
    for i, st in enumerate(states):
        ref = build_obs(st, env._last_action[i], env.commands.cmd[i], home=env.home)
        err = float(np.max(np.abs(actor[i] - ref)))
        errs.append(err)
        gyro_ref = np.asarray(st.base_angvel_local, dtype=np.float32).reshape(3)
        grav_ref = quat_rotate_inverse_wxyz(st.base_quat_wxyz, _DOWN).astype(np.float32)
        gyro_err = float(np.max(np.abs(actor[i, 0:3] - gyro_ref)))
        grav_err = float(np.max(np.abs(actor[i, 3:6] - grav_ref)))
        msg = (
            f"{when} env={i} max_abs={err:.3e} gyro_err={gyro_err:.3e} "
            f"grav_err={grav_err:.3e} last_action={env._last_action[i, :3]} "
            f"cmd={env.commands.cmd[i, :3]}"
        )
        if err >= PARITY_ABS:
            delta = actor[i] - ref
            j = int(np.argmax(np.abs(delta)))
            msg += f" peak_idx={j} vec={actor[i, j]:.8f} ref={ref[j]:.8f}"
        assert err < PARITY_ABS, msg
        assert gyro_err < PARITY_ABS, msg
        assert grav_err < PARITY_ABS, msg
        assert actor[i].shape == (61,)
        assert ref.shape[0] == 61
        assert env._last_action[i].shape == (NUM_ACTIONS,)
    return errs


class TestBatchedQuatRotate(unittest.TestCase):
    @unittest.skipIf(_SKIP, str(_SKIP))
    def test_quat_rotate_inv_n_matches_scalar(self) -> None:
        from sim2sim.train.vec_env import _DOWN as VDOWN
        from sim2sim.train.vec_env import _quat_rotate_inv_n

        rng = np.random.default_rng(0)
        q = rng.normal(size=(32, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        batched = _quat_rotate_inv_n(q, VDOWN)
        ref = np.stack([quat_rotate_inverse_wxyz(qi, VDOWN) for qi in q], axis=0)
        err = float(np.max(np.abs(batched - ref)))
        self.assertLess(err, 1e-12, msg=f"batched vs scalar max_abs={err:.3e}")


class TestResetPoseSampler(unittest.TestCase):
    def test_yaw_only_and_joint_noise(self) -> None:
        from sim2sim.runner import load_robot_cfg
        from sim2sim.train.reset_poses import HomePoseSampler

        cfg = load_robot_cfg(CFG_PATH)
        sampler = HomePoseSampler(cfg)
        try:
            rng = np.random.default_rng(7)
            yaw = 0.7
            poses, q0, ctrl = sampler.sample(rng, yaw_range=(yaw, yaw), joint_noise_rad=0.0)
            np.testing.assert_allclose(q0, sampler.home, atol=1e-7)
            np.testing.assert_allclose(ctrl, sampler.home, atol=1e-7)
            adr = sampler.mj.free_qposadr
            q_root = np.asarray(sampler.mj.data.qpos[adr + 3 : adr + 7], dtype=np.float64)
            want = np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])
            np.testing.assert_allclose(q_root, want, atol=1e-7)
            roll, pitch, got_yaw = _rpy_wxyz(q_root)
            self.assertAlmostEqual(roll, 0.0, places=5)
            self.assertAlmostEqual(pitch, 0.0, places=5)
            self.assertAlmostEqual(got_yaw, yaw, places=5)
            self.assertIn("trunk_base", {p["name"] for p in poses})

            rng2 = np.random.default_rng(9)
            _, qn, ctrl_n = sampler.sample(rng2, yaw_range=(0.0, 0.0), joint_noise_rad=0.05)
            np.testing.assert_allclose(ctrl_n, sampler.home, atol=1e-7)
            delta = qn.astype(np.float64) - sampler.home.astype(np.float64)
            self.assertTrue(np.all(np.abs(delta) <= 0.05 + 1e-6), msg=f"joint delta={delta}")
            self.assertGreater(float(np.max(np.abs(delta))), 1e-6)
        finally:
            sampler.close()


@unittest.skipIf(_SKIP, str(_SKIP))
class TestGodotVecEnvObsParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sim2sim.train.vec_env import GodotVecEnv

        cls.env = GodotVecEnv(_walk_cfg(), num_envs=2, device="cpu", seed=1, headless=True)

    @classmethod
    def tearDownClass(cls) -> None:
        env = getattr(cls, "env", None)
        if env is not None:
            env.close()

    def test_01_reset_obs_last_action_and_parity(self) -> None:
        import torch

        env = self.env
        env.reset_idx([0, 1])
        obs = env.get_observations()["actor"].detach().cpu().numpy()
        self.assertTrue(np.allclose(env._last_action, 0.0), msg="reset last_action must be zeros, not home")
        self.assertTrue(np.allclose(obs[:, 34:48], 0.0), msg="first obs last_action slot must be zeros")
        np.testing.assert_allclose(obs[:, 48:61], env.commands.cmd, atol=0.0)
        # Command is applied on the reset obs (same step as play/eval first obs).
        self.assertTrue(np.allclose(obs[:, 48:51], np.array([0.15, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(obs[:, 51:61], 0.0))
        errs = _assert_obs_parity(env, "after_reset")
        self.assertLess(max(errs), PARITY_ABS)
        for i, st in enumerate(env.debug_states()):
            self.assertFalse(
                ref_fallen(st.base_quat_wxyz, st.base_pos, tilt_deg=env.tilt_deg, min_z=env.min_z),
                msg=f"reset pose already fallen env={i} z={st.base_pos[2]}",
            )
        self.assertTrue(torch.all(env.episode_length_buf == 0))

        env.yaw_range = (0.7, 0.7)
        env.reset_idx([0])
        st0 = env.debug_states()[0]
        roll, pitch, got_yaw = _rpy_wxyz(st0.base_quat_wxyz)
        self.assertAlmostEqual(roll, 0.0, places=3, msg=f"body roll={roll} (want yaw-only reset)")
        self.assertAlmostEqual(pitch, 0.0, places=3, msg=f"body pitch={pitch} (want yaw-only reset)")
        self.assertAlmostEqual(got_yaw, 0.7, places=3, msg=f"body yaw={got_yaw}")
        env.yaw_range = (0.0, 0.0)
        env.reset_idx([0, 1])

    def test_02_step_fixed_action_parity_and_fallen(self) -> None:
        import torch

        env = self.env
        env.reset_idx([0, 1])
        act = torch.full((2, 14), 0.05, dtype=torch.float32)
        last_dones = None
        for step_i in range(20):
            _obs, _rew, dones, _extras = env.step(act)
            last_dones = dones
            errs = _assert_obs_parity(env, f"after_step_{step_i}")
            self.assertLess(max(errs), PARITY_ABS, msg=f"step={step_i} errs={errs}")
            for i in range(env.num_envs):
                pred = ref_fallen(
                    env._last_term_quat[i],
                    env._last_term_pos[i],
                    tilt_deg=env.tilt_deg,
                    min_z=env.min_z,
                )
                self.assertEqual(
                    bool(pred),
                    bool(env._last_fell[i]),
                    msg=(
                        f"step={step_i} env={i} fallen(ref)={pred} "
                        f"vec={bool(env._last_fell[i])} z={env._last_term_pos[i, 2]} "
                        f"grav={env._last_term_grav[i]}"
                    ),
                )
                grav_ref = quat_rotate_inverse_wxyz(env._last_term_quat[i], _DOWN).astype(np.float32)
                grav_err = float(np.max(np.abs(env._last_term_grav[i] - grav_ref)))
                self.assertLess(grav_err, PARITY_ABS, msg=f"term grav frame step={step_i} env={i} err={grav_err:.3e}")
                gyro_ref = np.asarray(env.debug_states()[i].base_angvel_local, dtype=np.float32).reshape(3)
                # After a fall-reset the current gyro is the reset state's, not term gyro.
                if not bool(dones[i]):
                    gyro_now = env.get_observations()["actor"][i, 0:3].detach().cpu().numpy()
                    self.assertLess(float(np.max(np.abs(gyro_now - gyro_ref))), PARITY_ABS)
            if not bool(dones.any()):
                np.testing.assert_allclose(env._last_action, 0.05, atol=0.0)
        self.assertIsNotNone(last_dones)

    def test_03_reset_landing_vs_runner(self) -> None:
        """Feet on the ground after 0.5 s of ctrl=home; z matches a runner-style reset."""
        import torch

        from sim2sim.backends.godot_backend import GodotBackend
        from sim2sim.play import capture_home_poses
        from sim2sim.runner import load_robot_cfg

        env = self.env
        env.reset_idx([0, 1])
        zeros = torch.zeros(2, 14, dtype=torch.float32)
        n_half = int(round(0.5 / env.dt))
        z_reset = [float(st.base_pos[2]) for st in env.debug_states()]
        for _ in range(n_half):
            env.step(zeros)
        z_half = [float(st.base_pos[2]) for st in env.debug_states()]
        contact = env.get_observations()["critic"][:, 64:66].detach().cpu().numpy()
        self.assertTrue(np.all(contact > 0.5), msg=f"feet contact after 0.5s stand={contact}")
        for i, z in enumerate(z_half):
            self.assertFalse(
                ref_fallen(
                    env.debug_states()[i].base_quat_wxyz,
                    env.debug_states()[i].base_pos,
                    tilt_deg=env.tilt_deg,
                    min_z=env.min_z,
                ),
                msg=f"fell while standing env={i} z={z}",
            )
            # Not floating (stuck near spawn) or penetrating through the floor.
            self.assertGreater(z, 0.09, msg=f"env={i} z={z} after 0.5s (penetrating?)")
            self.assertLess(z, 0.14, msg=f"env={i} z={z} after 0.5s (floating?)")

        robot = load_robot_cfg(CFG_PATH)
        poses = capture_home_poses(robot)
        home = np.asarray(robot.get("home", DEFAULT_HOME), dtype=np.float32)
        be = GodotBackend(
            Path(robot["godot_spec"]),
            timestep=float(robot.get("timestep", 0.005)),
            headless=True,
            base_body=str(robot.get("base_body", "trunk_base")),
            current_limit_a=float(robot.get("current_limit_a", 1.75)),
            recv_timeout=15.0,
        )
        try:
            st = be.reset(ctrl=home, bodies=poses)
            z_run0 = float(st.base_pos[2])
            for _ in range(n_half):
                st = be.step(home, n_substeps=int(robot.get("decimation", 4)), report="lite")
            z_run = float(st.base_pos[2])
        finally:
            be.close()
        for i, z in enumerate(z_half):
            self.assertLess(
                abs(z - z_run),
                0.01,
                msg=(
                    f"env={i} stand z={z:.4f} vs runner z={z_run:.4f} "
                    f"(reset z vec={z_reset[i]:.4f} runner={z_run0:.4f})"
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
