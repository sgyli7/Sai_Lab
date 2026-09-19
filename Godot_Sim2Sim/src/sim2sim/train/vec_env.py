"""rsl_rl 5.0.1 VecEnv: one Godot/Jolt worker per robot, lockstep TCP."""

from __future__ import annotations

import json
import os
import select
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.fall import fallen_mask
from sim2sim.godot_proc import stop_godot
from sim2sim.obs import DEFAULT_HOME
from sim2sim.paths import apply_path_defaults, expand_cfg, load_robot_json, sim2sim_root
from sim2sim.train.commands import CommandConfig, CommandSampler
from sim2sim.train.curriculum import Curriculum, apply_curriculum
from sim2sim.train.reset_poses import HomePoseSampler
from sim2sim.train.rewards import RewardComputer, RewardConfig, RewardInputs, sit_target_q, world_to_yaw_frame

FOOT_NAMES = ("ankle_left", "ankle_right")
ACTOR_DIM = 61
CRITIC_DIM = 70
NUM_ACTIONS = 14
_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _quat_rotate_inv_n(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Batch inverse-rotate ``vec`` by wxyz quats. Matches ``quat_rotate_inverse_wxyz``."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    w = q[:, :1]
    xyz = q[:, 1:4]
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def load_walk_cfg(path_or_dict: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    root = sim2sim_root()
    os.environ["SIM2SIM_ROOT"] = str(root)
    for key, ok in (
        ("MICRODUCK_POLICIES", lambda p: Path(p).is_dir()),
        ("MICRODUCK_RL", lambda p: (Path(p) / "src/mjlab_microduck").is_dir()),
    ):
        val = os.environ.get(key)
        if val and not ok(val):
            os.environ.pop(key, None)
    apply_path_defaults(root)
    if isinstance(path_or_dict, Mapping):
        cfg = expand_cfg(dict(path_or_dict))
    else:
        raw = yaml.safe_load(Path(path_or_dict).read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"walk cfg is not a mapping: {path_or_dict}")
        cfg = expand_cfg(raw)
    robot_ref = cfg.get("robot")
    if isinstance(robot_ref, dict):
        cfg["robot_cfg"] = expand_cfg(robot_ref)
        return cfg
    robot_path = Path(robot_ref) if isinstance(robot_ref, str) else (root / "robots/microduck.json")
    if not robot_path.is_file():
        robot_path = root / "robots/microduck.json"
    cfg["robot"] = str(robot_path)
    cfg["robot_cfg"] = load_robot_json(robot_path)
    return cfg


def _as_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def _foot_pack(
    state, ankle_z_nominal: np.ndarray, foot_names: tuple[str, str] = FOOT_NAMES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feet = (state.extra or {}).get("feet") or []
    by = {str(f.get("name")): f for f in feet if isinstance(f, dict)}
    contact = np.zeros(2, dtype=np.float32)
    height = np.zeros(2, dtype=np.float32)
    xy_speed = np.zeros(2, dtype=np.float32)
    for i, name in enumerate(foot_names):
        f = by.get(name)
        if f is None:
            continue
        contact[i] = 1.0 if f.get("contact") else 0.0
        pos = f.get("pos") or [0.0, 0.0, 0.0]
        height[i] = float(pos[2]) - float(ankle_z_nominal[i])
        lv = f.get("linvel") or [0.0, 0.0, 0.0]
        xy_speed[i] = float(np.hypot(float(lv[0]), float(lv[1])))
    return contact, height, xy_speed


def _mouth_z(state) -> float:
    bodies = (state.extra or {}).get("bodies")
    if not bodies:
        raw = (state.extra or {}).get("raw") or {}
        bodies = raw.get("bodies") or []
    for b in bodies:
        if not isinstance(b, dict):
            continue
        if str(b.get("name")) == "jaw_soft":
            pos = b.get("pos") or [0.0, 0.0, 0.0]
            return float(pos[2])
    return float("nan")


def _state_finite(state) -> bool:
    chunks = (state.q, state.qd, state.base_pos, state.base_quat_wxyz, state.base_linvel, state.base_angvel_local)
    return all(np.isfinite(np.asarray(x)).all() for x in chunks)


class GodotVecEnv(VecEnv):
    """One-robot-per-process Godot workers matching rsl_rl ``VecEnv``."""

    num_actions: int = NUM_ACTIONS

    def __init__(
        self,
        cfg_path_or_dict: str | Path | Mapping[str, Any],
        num_envs: int | None = None,
        device: str = "cpu",
        seed: int | None = None,
        headless: bool | None = None,
    ) -> None:
        cfg = load_walk_cfg(cfg_path_or_dict)
        robot = cfg["robot_cfg"]
        spec = Path(robot["godot_spec"])
        tscn = spec.with_name("robot.tscn")
        if not spec.is_file() or not tscn.is_file():
            from sim2sim.play import ensure_godot_scene

            ensure_godot_scene(robot)
        self.cfg = cfg
        self.num_envs = int(num_envs if num_envs is not None else cfg.get("num_envs", 8))
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.device = _as_device(device)
        seed = int(cfg.get("seed", 0) if seed is None else seed)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.headless = cfg.get("headless", True) if headless is None else bool(headless)
        self.home = np.asarray(robot.get("home", DEFAULT_HOME), dtype=np.float32).reshape(-1)
        if self.home.size != NUM_ACTIONS:
            raise ValueError(f"home len {self.home.size} != {NUM_ACTIONS}")
        self.scale = float(robot.get("action_scale", 1.0))
        self.dt_phys = float(robot.get("timestep", 0.005))
        self.decimation = int(robot.get("decimation", 4))
        self.dt = self.dt_phys * self.decimation
        self.episode_s = float(cfg.get("episode_s", 20.0))
        self.max_episode_length = int(round(self.episode_s / self.dt))
        self.recv_timeout = float(cfg.get("recv_timeout_s", 10.0))
        self.stagger_s = float(cfg.get("spawn_stagger_s", 0.0))
        self.max_faults_per_step = int(cfg.get("max_faults_per_step", 8))
        self.faults_jsonl = Path(cfg.get("faults_jsonl") or (sim2sim_root() / "results/faults.jsonl"))
        self.faults_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.spec_path = Path(robot["godot_spec"])
        self.current_limit_a = float(robot.get("current_limit_a", 1.75))
        self.base_body = str(robot.get("base_body", "trunk_base"))
        self.tilt_deg = float((cfg.get("termination") or {}).get("tilt_deg", 70.0))
        self.min_z = float((cfg.get("termination") or {}).get("min_z", 0.055))
        self.fall_enabled = bool((cfg.get("termination") or {}).get("fallen", True))
        reset_cfg = cfg.get("reset") or {}
        self.yaw_range = tuple(float(x) for x in reset_cfg.get("yaw_range", (-np.pi, np.pi)))
        self.joint_noise = float(reset_cfg.get("joint_noise_rad", 0.05))
        self.sit_frac = float(reset_cfg.get("sit_frac", 0.0))
        self.sit_z = float((cfg.get("rewards") or {}).get("sit_z", 0.060))
        self.sit_q = sit_target_q(self.home)
        noise_cfg = cfg.get("obs_noise") or {}
        self.noise_enabled = bool(noise_cfg.get("enabled", True))
        self.noise_kind = str(noise_cfg.get("kind", "uniform"))
        self.noise_amp = {
            "gyro": float(noise_cfg.get("gyro", 0.03)),
            "grav": float(noise_cfg.get("grav", 0.01)),
            "q": float(noise_cfg.get("q", 0.001)),
            "qd": float(noise_cfg.get("qd", 0.25)),
        }
        push_cfg = cfg.get("pushes") or {}
        self.push_enabled = bool(push_cfg.get("enabled", True))
        self.push_interval = tuple(float(x) for x in push_cfg.get("interval_s", (3.0, 6.0)))
        self.push_xy_speed = float(push_cfg.get("xy_speed", 0.3))

        self.sampler = HomePoseSampler(robot)
        self.foot_names = tuple(self.sampler.support_names)
        self.commands = CommandSampler(CommandConfig.from_dict(cfg.get("commands")), self.num_envs, self._rng)
        self.rew = RewardComputer(
            RewardConfig.from_dict(cfg.get("rewards")),
            self.num_envs,
            self.dt,
            self.home,
            self.sampler.joint_lo.astype(np.float32),
            self.sampler.joint_hi.astype(np.float32),
        )
        self.curriculum = Curriculum.from_dict(cfg.get("curriculum"))
        self._curriculum_values: dict[str, float] = {}

        self._workers: list[GodotBackend | None] = [None] * self.num_envs
        self._states: list[Any] = [None] * self.num_envs
        self._last_action = np.zeros((self.num_envs, NUM_ACTIONS), dtype=np.float32)
        self._push_ttl = np.zeros(self.num_envs, dtype=np.float64)
        self._last_fell = np.zeros(self.num_envs, dtype=bool)
        self._last_term_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self._last_term_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._last_term_gyro = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._last_term_grav = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.faults = 0
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = torch.from_numpy(self._ep_len)
        # Double-buffer obs so rsl_rl's act() can hold the previous TensorDict while we fill the next.
        self._actor_np = [np.zeros((self.num_envs, ACTOR_DIM), dtype=np.float32) for _ in range(2)]
        self._critic_np = [np.zeros((self.num_envs, CRITIC_DIM), dtype=np.float32) for _ in range(2)]
        self._obs_td = [
            TensorDict(
                {
                    "actor": torch.from_numpy(self._actor_np[i]),
                    "critic": torch.from_numpy(self._critic_np[i]),
                },
                batch_size=[self.num_envs],
            )
            for i in range(2)
        ]
        self._obs_i = 0
        self._obs = self._obs_td[0]
        self.ankle_z_nominal = np.array(self.sampler.ankle_z_nominal, dtype=np.float32, copy=True)
        self._closed = False
        try:
            self._spawn_all()
            self._calibrate_ankle_z()
            self.reset_idx(list(range(self.num_envs)))
        except Exception:
            self.close()
            raise

    def _spawn_all(self) -> None:
        for i in range(self.num_envs):
            self._workers[i] = self._spawn_one()
            if self.stagger_s > 0 and i + 1 < self.num_envs:
                time.sleep(self.stagger_s)

    def _spawn_one(self) -> GodotBackend:
        return GodotBackend(
            self.spec_path,
            timestep=self.dt_phys,
            headless=self.headless,
            base_body=self.base_body,
            current_limit_a=self.current_limit_a,
            recv_timeout=self.recv_timeout,
        )

    def _calibrate_ankle_z(self) -> None:
        w0 = self._workers[0]
        assert w0 is not None
        st = self._reset_backend(w0, self.sampler.nominal())
        feet = (st.extra or {}).get("feet") or []
        by = {str(f.get("name")): f for f in feet if isinstance(f, dict)}
        zs = []
        for name in self.foot_names:
            f = by.get(name)
            if f is None:
                zs.append(float(self.sampler.ankle_z_nominal[len(zs)]))
            else:
                zs.append(float((f.get("pos") or [0, 0, 0])[2]))
        self.ankle_z_nominal = np.asarray(zs, dtype=np.float32)

    def _reset_backend(self, worker: GodotBackend, poses: list[dict]):
        st = worker.reset(ctrl=self.home, bodies=poses)
        raw = (st.extra or {}).get("raw") or {}
        missing = raw.get("missing") or []
        if missing:
            raise RuntimeError(f"reset missing bodies: {missing}")
        applied = raw.get("applied") or []
        names = {p["name"] for p in poses}
        if names and applied and len(applied) < len(names):
            raise RuntimeError(f"reset applied {len(applied)}/{len(names)} bodies")
        return st

    def _reset_one(self, i: int) -> None:
        w = self._workers[i]
        assert w is not None
        sit = self.sit_frac > 0.0 and float(self._rng.random()) < self.sit_frac
        poses, _q0, _ctrl = self.sampler.sample(
            self._rng,
            yaw_range=self.yaw_range,
            joint_noise_rad=self.joint_noise,
            z=self.sit_z if sit else None,
            q_base=self.sit_q if sit else None,
        )
        self._states[i] = self._reset_backend(w, poses)
        self.commands.reset([i])
        self._last_action[i] = 0.0
        self.rew.zero_sums([i])
        self._push_ttl[i] = float(self._rng.uniform(*self.push_interval)) if self.push_enabled else 1e9
        self._ep_len[i] = 0

    def reset_idx(self, env_ids: list[int] | np.ndarray | torch.Tensor) -> None:
        ids = [int(i) for i in np.asarray(env_ids, dtype=np.int64).reshape(-1)]
        for i in ids:
            self._reset_one(i)
        self._pack_obs(ids)

    def get_observations(self) -> TensorDict:
        return self._obs

    def set_curriculum(self, it: int) -> dict[str, float]:
        """Apply iteration-indexed reward weights and command mix. Returns live values."""
        values = self.curriculum.values_at(int(it))
        apply_curriculum(self.rew.cfg, self.commands.cfg, values)
        self._curriculum_values = values
        return values

    def debug_states(self) -> list:
        """Latest SimState per worker (post-reset if the last step terminated)."""
        return list(self._states)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self._closed:
            raise RuntimeError("GodotVecEnv is closed")
        n = self.num_envs
        if actions.device.type == "cpu":
            act = np.array(actions.detach().numpy(), dtype=np.float32, copy=True).reshape(n, NUM_ACTIONS)
        else:
            act = np.asarray(actions.detach().to("cpu").numpy(), dtype=np.float32).reshape(n, NUM_ACTIONS)
        nan_act = ~np.isfinite(act).all(axis=1)
        act = np.where(nan_act[:, None], 0.0, act).astype(np.float32)
        ctrl = self.home[None, :] + act * self.scale

        self._ep_len += 1
        self._maybe_push()

        send_faults: list[tuple[int, str]] = []
        for i, w in enumerate(self._workers):
            assert w is not None
            try:
                w.send_step(ctrl[i], n_substeps=self.decimation, report="lite")
            except Exception as e:
                send_faults.append((i, f"send:{type(e).__name__}:{e}"))

        send_bad = {i for i, _ in send_faults}
        recv_faults = self._recv_all(send_bad)

        faults = send_faults + recv_faults
        if len(faults) > self.max_faults_per_step:
            raise RuntimeError(f"faults this step {len(faults)} > max_faults_per_step={self.max_faults_per_step}")

        fault_ids = {i for i, _ in faults}
        for i, reason in faults:
            self._handle_fault(i, reason)

        q, qd, gyro, quat, pos, linvel, contact, height, xy_speed, finite = self._pull_states()
        grav = _quat_rotate_inv_n(quat, _DOWN).astype(np.float32)

        lin_yaw = world_to_yaw_frame(quat, linvel)
        cmd = self.commands.step(self.dt)
        mouth = np.array([_mouth_z(self._states[i]) for i in range(n)], dtype=np.float32)
        ep_t = self._ep_len.astype(np.float32) * self.dt
        total, terms = self.rew.compute(
            RewardInputs(
                q=q,
                gyro=gyro,
                grav=grav,
                base_linvel_yaw=lin_yaw,
                cmd13=cmd,
                action=act,
                last_action=self._last_action,
                contact=contact,
                foot_height=height,
                foot_xy_speed=xy_speed,
                joint_lo=self.sampler.joint_lo,
                joint_hi=self.sampler.joint_hi,
                home=self.home,
                base_z=pos[:, 2].astype(np.float32),
                mouth_z=mouth,
                ep_t=ep_t,
            )
        )

        fell = fallen_mask(grav, pos, tilt_deg=self.tilt_deg, min_z=self.min_z)
        if not self.fall_enabled:
            fell = np.zeros(n, dtype=bool)
        self._last_fell = np.asarray(fell, dtype=bool).copy()
        self._last_term_quat = np.asarray(quat, dtype=np.float64).copy()
        self._last_term_pos = np.asarray(pos, dtype=np.float64).copy()
        self._last_term_gyro = np.asarray(gyro, dtype=np.float32).copy()
        self._last_term_grav = np.asarray(grav, dtype=np.float32).copy()
        nan_state = nan_act | ~finite | ~np.isfinite(total)
        time_out = self._ep_len >= self.max_episode_length
        is_fault = np.array([i in fault_ids for i in range(n)], dtype=bool)

        # Faults already reset: zero their physics reward (no bogus terminal).
        total = np.where(is_fault | nan_state, 0.0, total).astype(np.float32)

        term = fell | nan_state
        done = term | time_out | is_fault
        to = (time_out | is_fault) & ~term

        reset_ids = [i for i in range(n) if done[i] and not is_fault[i]]
        for i in reset_ids:
            try:
                self._reset_one(i)
            except Exception as e:
                self._handle_fault(i, f"reset:{type(e).__name__}:{e}")
                is_fault[i] = True
                done[i] = True
                to[i] = True

        self._last_action = act.copy()
        self._last_action[done] = 0.0

        self._obs_i ^= 1
        self._fill_obs_arrays(self._obs_i)
        self._obs = self._obs_td[self._obs_i]

        log: dict[str, float] = {f"Episode_Reward/{k}": float(np.mean(v)) for k, v in terms.items()}
        log["Episode_Termination/fell"] = float(np.mean(fell.astype(np.float32)))
        log["Episode_Termination/nan_state"] = float(np.mean(nan_state.astype(np.float32)))
        log["Episode_Termination/time_out"] = float(np.mean(time_out.astype(np.float32)))
        log["faults"] = float(self.faults)
        log["step_reward"] = float(np.mean(total))

        extras = {
            "time_outs": torch.as_tensor(to, dtype=torch.bool, device=self.device),
            "log": log,
        }
        rewards = torch.as_tensor(total, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        return self._obs, rewards, dones, extras

    def _recv_all(self, send_bad: set[int]) -> list[tuple[int, str]]:
        pending = [i for i in range(self.num_envs) if i not in send_bad]
        faults: list[tuple[int, str]] = []
        if not pending:
            return faults
        if len(pending) == 1:
            i = pending[0]
            w = self._workers[i]
            assert w is not None
            try:
                self._states[i] = w.recv_step()
            except Exception as e:
                faults.append((i, f"recv:{type(e).__name__}:{e}"))
            return faults
        deadline = time.monotonic() + self.recv_timeout
        while pending:
            remain = deadline - time.monotonic()
            if remain <= 0:
                for i in pending:
                    faults.append((i, "recv:TimeoutError:select"))
                break
            socks: list[Any] = []
            by_fd: dict[int, int] = {}
            for i in pending:
                w = self._workers[i]
                assert w is not None
                sock = w._client.sock
                socks.append(sock)
                by_fd[int(sock.fileno())] = i
            try:
                ready, _, _ = select.select(socks, [], [], remain)
            except (OSError, ValueError) as e:
                for i in pending:
                    faults.append((i, f"recv:{type(e).__name__}:{e}"))
                break
            if not ready:
                for i in pending:
                    faults.append((i, "recv:TimeoutError:select"))
                break
            done_now: list[int] = []
            for sock in ready:
                i = by_fd[int(sock.fileno())]
                w = self._workers[i]
                assert w is not None
                try:
                    self._states[i] = w.recv_step()
                except Exception as e:
                    faults.append((i, f"recv:{type(e).__name__}:{e}"))
                done_now.append(i)
            pending = [i for i in pending if i not in done_now]
        return faults

    def _pull_states(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        n = self.num_envs
        q = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        qd = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        gyro = np.zeros((n, 3), dtype=np.float32)
        quat = np.zeros((n, 4), dtype=np.float64)
        pos = np.zeros((n, 3), dtype=np.float64)
        linvel = np.zeros((n, 3), dtype=np.float32)
        contact = np.zeros((n, 2), dtype=np.float32)
        height = np.zeros((n, 2), dtype=np.float32)
        xy_speed = np.zeros((n, 2), dtype=np.float32)
        finite = np.ones(n, dtype=bool)
        for i in range(n):
            st = self._states[i]
            q[i] = np.asarray(st.q, dtype=np.float32).reshape(-1)[:NUM_ACTIONS]
            qd[i] = np.asarray(st.qd, dtype=np.float32).reshape(-1)[:NUM_ACTIONS]
            gyro[i] = np.asarray(st.base_angvel_local, dtype=np.float32).reshape(3)
            quat[i] = np.asarray(st.base_quat_wxyz, dtype=np.float64).reshape(4)
            pos[i] = np.asarray(st.base_pos, dtype=np.float64).reshape(3)
            linvel[i] = np.asarray(st.base_linvel, dtype=np.float32).reshape(3)
            contact[i], height[i], xy_speed[i] = _foot_pack(
                st, self.ankle_z_nominal, self.foot_names
            )
            finite[i] = _state_finite(st) and np.isfinite(contact[i]).all() and np.isfinite(height[i]).all()
        return q, qd, gyro, quat, pos, linvel, contact, height, xy_speed, finite

    def _fill_obs_arrays(self, buf: int) -> None:
        """Write actor/critic obs into double-buffer slot ``buf`` (in-place numpy)."""
        q, qd, gyro, quat, _pos, linvel, contact, height, _xy, _finite = self._pull_states()
        n = self.num_envs
        grav = _quat_rotate_inv_n(quat, _DOWN).astype(np.float32)
        actor = self._actor_np[buf]
        actor[:, 0:3] = gyro
        actor[:, 3:6] = grav
        actor[:, 6:20] = q - self.home.reshape(1, NUM_ACTIONS)
        actor[:, 20:34] = qd
        actor[:, 34:48] = self._last_action
        actor[:, 48:61] = self.commands.cmd
        critic = self._critic_np[buf]
        critic[:, :ACTOR_DIM] = actor  # privileged critic keeps the clean 61-D prefix
        extra = critic[:, ACTOR_DIM:]
        extra[:, 0:3] = world_to_yaw_frame(quat, linvel)
        extra[:, 3:5] = contact
        extra[:, 5:7] = height
        extra[:, 7:9] = self.rew.air.air_time
        self._noise_actor_inplace(actor)

    def _build_obs_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy of the live buffer (tests / debug)."""
        self._fill_obs_arrays(self._obs_i)
        return self._actor_np[self._obs_i].copy(), self._critic_np[self._obs_i].copy()

    def _noise_actor_inplace(self, o: np.ndarray) -> None:
        if not self.noise_enabled:
            return
        n = o.shape[0]

        def draw(amp: float, dim: int) -> np.ndarray:
            if amp <= 0:
                return np.zeros((n, dim), dtype=np.float32)
            if self.noise_kind == "gaussian":
                return self._rng.normal(0.0, amp, size=(n, dim)).astype(np.float32)
            return self._rng.uniform(-amp, amp, size=(n, dim)).astype(np.float32)

        o[:, 0:3] += draw(self.noise_amp["gyro"], 3)
        o[:, 3:6] += draw(self.noise_amp["grav"], 3)
        o[:, 6:20] += draw(self.noise_amp["q"], 14)
        o[:, 20:34] += draw(self.noise_amp["qd"], 14)

    def _noise_actor(self, obs: np.ndarray) -> np.ndarray:
        o = np.array(obs, dtype=np.float32, copy=True)
        self._noise_actor_inplace(o)
        return o

    def _pack_obs(self, env_ids: list[int] | None = None) -> None:
        del env_ids
        self._fill_obs_arrays(self._obs_i)
        self._obs = self._obs_td[self._obs_i]

    def _td(self, actor: np.ndarray, critic: np.ndarray) -> TensorDict:
        return TensorDict(
            {
                "actor": torch.as_tensor(actor, dtype=torch.float32, device=self.device),
                "critic": torch.as_tensor(critic, dtype=torch.float32, device=self.device),
            },
            batch_size=[self.num_envs],
        )

    def _maybe_push(self) -> None:
        if not self.push_enabled:
            return
        self._push_ttl -= self.dt
        for i in range(self.num_envs):
            if self._push_ttl[i] > 0:
                continue
            w = self._workers[i]
            assert w is not None
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            spd = float(self._rng.uniform(0.0, self.push_xy_speed))
            lin = np.array([spd * np.cos(ang), spd * np.sin(ang), 0.0], dtype=np.float64)
            try:
                w.nudge(lin)
            except Exception as e:
                self._handle_fault(i, f"nudge:{type(e).__name__}:{e}")
            self._push_ttl[i] = float(self._rng.uniform(*self.push_interval))

    def _handle_fault(self, i: int, reason: str) -> None:
        self.faults += 1
        rec = {
            "t": time.time(),
            "worker": int(i),
            "reason": str(reason)[:500],
            "faults": int(self.faults),
        }
        try:
            with self.faults_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        self._respawn(i)
        try:
            self._reset_one(i)
        except Exception as e:
            # Last-ditch respawn.
            self._respawn(i)
            self._reset_one(i)
            rec2 = dict(rec)
            rec2["reason"] = f"reset-after-respawn:{type(e).__name__}:{e}"
            try:
                with self.faults_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec2) + "\n")
            except OSError:
                pass

    def _respawn(self, i: int) -> None:
        w = self._workers[i]
        if w is not None:
            self._kill_worker(w)
        self._workers[i] = self._spawn_one()

    def _kill_worker(self, worker: GodotBackend) -> None:
        proc = getattr(worker, "_proc", None)
        client = getattr(worker, "_client", None)
        if client is not None:
            try:
                client.sock.close()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        try:
            stop_godot(proc, None)
        except Exception:
            pass

    def bench_step_rate(self, n_steps: int) -> dict[str, float]:
        n_steps = int(n_steps)
        zeros = torch.zeros(self.num_envs, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step(zeros)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        total = n_steps * self.num_envs
        return {
            "num_envs": float(self.num_envs),
            "n_steps": float(n_steps),
            "elapsed_s": float(elapsed),
            "steps_per_s": float(total / elapsed),
            "latency_s": float(elapsed / n_steps),
            "faults": float(self.faults),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i, w in enumerate(self._workers):
            if w is None:
                continue
            try:
                w.close()
            except Exception:
                self._kill_worker(w)
            self._workers[i] = None
        try:
            self.sampler.close()
        except Exception:
            pass

    def __enter__(self) -> GodotVecEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
