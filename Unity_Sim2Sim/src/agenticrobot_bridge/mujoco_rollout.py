"""Deterministic, headless MuJoCo rollouts for the locked MicroDuck policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import mujoco
import numpy as np
import onnxruntime as ort


OBSERVATION_SIZE = 61
ACTION_SIZE = 14
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035


class RolloutError(RuntimeError):
    """Raised when a scenario or model violates the rollout contract."""


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """One threshold comparison from a scenario's acceptance contract."""

    name: str
    observed: float
    threshold: float
    comparison: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Metrics and explicit verdict for one policy scenario."""

    policy_name: str
    role: str
    passed: bool
    physics_steps: int
    policy_steps: int
    metrics: dict[str, float]
    checks: tuple[AcceptanceCheck, ...]
    trace_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "policyName": self.policy_name,
            "role": self.role,
            "passed": self.passed,
            "physicsSteps": self.physics_steps,
            "policySteps": self.policy_steps,
            "metrics": self.metrics,
            "checks": [check.to_dict() for check in self.checks],
            "tracePath": self.trace_path.as_posix(),
        }


@dataclass(frozen=True, slots=True)
class RolloutSuiteReport:
    """Aggregate verdict and provenance for the configured behavior baseline."""

    results: tuple[RolloutResult, ...]
    calibrations: dict[str, dict[str, Any]]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        passed_count = sum(result.passed for result in self.results)
        return {
            "schemaVersion": 1,
            "passed": self.passed,
            "summary": {
                "passed": passed_count,
                "failed": len(self.results) - passed_count,
                "total": len(self.results),
            },
            "calibrations": self.calibrations,
            "results": [result.to_dict() for result in self.results],
        }


def run_all_policy_scenarios(
    project_root: str | Path,
    output_directory: str | Path,
) -> RolloutSuiteReport:
    """Run every configured policy and write traces plus one JSON verdict report."""

    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    config = json.loads(
        (root / "config" / "policy-scenarios.json").read_text(encoding="utf-8")
    )
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise RolloutError("policy-scenarios.json must define at least one scenario")

    walking = scenarios.get("alpha_walking.onnx")
    if not isinstance(walking, dict) or float(walking.get("actionScale", -1.0)) != 1.1:
        raise RolloutError(
            "The headless MuJoCo walking baseline requires calibrated actionScale 1.1"
        )

    trace_directory = output / "traces"
    results = tuple(
        run_policy_scenario(
            root,
            policy_name,
            trace_directory / f"{Path(policy_name).stem}.jsonl",
        )
        for policy_name in scenarios
    )
    report = RolloutSuiteReport(
        results=results,
        calibrations={
            "alpha_walking.onnx": {
                "actionScale": 1.1,
                "scope": "headless MuJoCo plain-XML behavior baseline",
                "productionRobotdActionScale": 0.9,
                "officialInferPolicyDefaultActionScale": 1.0,
                "reason": (
                    "Empirically required to cross the configured 0.03 m gait "
                    "threshold with the plain-XML contact model."
                ),
            }
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "rollout-report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def run_policy_scenario(
    project_root: str | Path,
    policy_name: str,
    trace_path: str | Path,
    *,
    duration_seconds: float | None = None,
) -> RolloutResult:
    """Run one configured policy in MuJoCo and write its versioned JSONL trace."""

    root = Path(project_root).resolve()
    config_path = root / "config" / "policy-scenarios.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        scenario = config["scenarios"][policy_name]
    except KeyError as exc:
        raise RolloutError(f"No configured scenario for {policy_name}") from exc

    timestep = float(config["physicsTimestepSeconds"])
    decimation = int(config["policyDecimation"])
    configured_duration = float(scenario["durationSeconds"])
    duration = configured_duration if duration_seconds is None else float(duration_seconds)
    if timestep <= 0 or decimation <= 0 or duration <= 0:
        raise RolloutError("Timestep, policy decimation, and duration must be positive")
    physics_steps = int(round(duration / timestep))
    if not math.isclose(physics_steps * timestep, duration, abs_tol=1e-12):
        raise RolloutError(f"Duration {duration} is not an integer multiple of timestep {timestep}")

    model_root = (
        root
        / ".cache"
        / "upstream"
        / "microduck_rl"
        / "src"
        / "mjlab_microduck"
        / "robot"
        / "microduck"
    )
    scene_path = model_root / scenario["scene"]
    policy_path = root / ".cache" / "upstream" / "microduck" / "policies" / policy_name
    if not scene_path.is_file() or not policy_path.is_file():
        raise RolloutError(f"Missing scene or policy for {policy_name}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if (
        len(input_meta) != 1
        or len(output_meta) != 1
        or tuple(input_meta[0].shape) != (1, OBSERVATION_SIZE)
        or tuple(output_meta[0].shape) != (1, ACTION_SIZE)
    ):
        raise RolloutError(f"{policy_name} does not implement [1,61] -> [1,14]")

    joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    if model.nu != ACTION_SIZE or any(joint_id < 0 for joint_id in joint_ids):
        raise RolloutError(f"Scene {scene_path.name} must expose 14 joint servos")
    joint_qpos_indices = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int32
    )
    joint_qvel_indices = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids], dtype=np.int32
    )
    servo_names = [
        _required_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    passive_wheel_names, passive_wheel_qvel = _passive_wheels(model)

    root_joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    root_qvel_address = int(model.jnt_dofadr[root_joint_id])
    trunk_body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro_sensor_id = _required_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    gyro_address = int(model.sensor_adr[gyro_sensor_id])

    home = np.asarray(config["defaultJointPositionRad"], dtype=np.float32)
    if home.shape != (ACTION_SIZE,):
        raise RolloutError("defaultJointPositionRad must contain 14 values")
    data.qpos[root_qpos_address : root_qpos_address + 3] = [
        0.0,
        0.0,
        0.1385 if scenario["robotVariant"] == "roller" else 0.125,
    ]
    data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[joint_qpos_indices] = home
    data.ctrl[:] = home

    if scenario["robotVariant"] == "roller":
        for qvel_address in passive_wheel_qvel:
            model.dof_frictionloss[qvel_address] = 0.003

    ball_qpos_address = _optional_joint_address(model, "ball_free", qpos=True)
    ball_qvel_address = _optional_joint_address(model, "ball_free", qpos=False)
    if scenario["role"] in {"kick-left", "kick-right"}:
        if ball_qpos_address is None or ball_qvel_address is None:
            raise RolloutError(f"Kick scenario {policy_name} has no ball_free joint")
        _place_ball(
            data,
            root_qpos_address,
            ball_qpos_address,
            ball_qvel_address,
            left=scenario["role"] == "kick-left",
        )

    mujoco.mj_forward(model, data)
    initial_root = data.qpos[root_qpos_address : root_qpos_address + 3].copy()
    initial_joints = data.qpos[joint_qpos_indices].copy()
    joint_min = initial_joints.copy()
    joint_max = initial_joints.copy()
    min_height = max_height = float(initial_root[2])
    min_upright = _upright_dot(data, trunk_body_id)
    previous_pitch = _root_pitch(data.qpos[root_qpos_address + 3 : root_qpos_address + 7])
    cumulative_pitch = 0.0
    initial_ball = (
        data.qpos[ball_qpos_address : ball_qpos_address + 3].copy()
        if ball_qpos_address is not None
        else None
    )

    last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
    last_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    last_command = np.zeros(13, dtype=np.float32)
    last_target = home.copy()
    previous_target: np.ndarray | None = None
    current_policy_step = -1
    policy_steps = 0
    finite_state = True
    head_indices = frozenset(int(index) for index in config["runtimeFilter"]["headJointIndices"])
    head_alpha = float(config["runtimeFilter"]["headAlpha"])
    leg_alpha = float(config["runtimeFilter"]["legAlpha"])

    destination = Path(trace_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as trace:
        _write_record(
            trace,
            {
                "recordType": "header",
                "schemaVersion": 1,
                "producer": "mujoco",
                "policyName": policy_name,
                "policySha256": _sha256(policy_path),
                "scene": scenario["scene"],
                "sceneSha256": _sha256(scene_path),
                "role": scenario["role"],
                "robotVariant": scenario["robotVariant"],
                "physicsTimestepSeconds": timestep,
                "policyDecimation": decimation,
                "observationSize": OBSERVATION_SIZE,
                "actionSize": ACTION_SIZE,
                "servoNames": servo_names,
                "passiveWheelNames": list(passive_wheel_names),
            },
        )

        for physics_step in range(physics_steps):
            time_before_step = physics_step * timestep
            if physics_step % decimation == 0:
                current_policy_step = policy_steps
                last_command = _command_at_time(scenario, time_before_step)
                last_observation = _observation(
                    data,
                    trunk_body_id,
                    gyro_address,
                    joint_qpos_indices,
                    joint_qvel_indices,
                    home,
                    last_action,
                    last_command,
                )
                last_action = session.run(
                    [output_meta[0].name],
                    {input_meta[0].name: last_observation.reshape(1, -1)},
                )[0].reshape(-1).astype(np.float32)
                unfiltered = home + last_action * float(scenario["actionScale"])
                last_target = _filter_targets(
                    unfiltered,
                    previous_target,
                    head_indices,
                    head_alpha,
                    leg_alpha,
                )
                previous_target = last_target.copy()
                data.ctrl[:] = last_target
                policy_steps += 1

            mujoco.mj_step(model, data)
            joints = data.qpos[joint_qpos_indices].copy()
            joint_min = np.minimum(joint_min, joints)
            joint_max = np.maximum(joint_max, joints)
            root_position = data.qpos[root_qpos_address : root_qpos_address + 3].copy()
            min_height = min(min_height, float(root_position[2]))
            max_height = max(max_height, float(root_position[2]))
            min_upright = min(min_upright, _upright_dot(data, trunk_body_id))
            pitch = _root_pitch(data.qpos[root_qpos_address + 3 : root_qpos_address + 7])
            cumulative_pitch += _wrapped_angle_difference(pitch, previous_pitch)
            previous_pitch = pitch
            finite_state = finite_state and bool(
                np.isfinite(data.qpos).all()
                and np.isfinite(data.qvel).all()
                and np.isfinite(last_observation).all()
                and np.isfinite(last_action).all()
                and np.isfinite(last_target).all()
            )

            _write_record(
                trace,
                {
                    "recordType": "frame",
                    "physicsStep": physics_step,
                    "policyStep": current_policy_step,
                    "timeSeconds": (physics_step + 1) * timestep,
                    "rootPosition": root_position.tolist(),
                    "rootQuaternionWxyz": data.qpos[
                        root_qpos_address + 3 : root_qpos_address + 7
                    ].tolist(),
                    "rootLinearVelocity": data.qvel[
                        root_qvel_address : root_qvel_address + 3
                    ].tolist(),
                    "rootAngularVelocity": data.qvel[
                        root_qvel_address + 3 : root_qvel_address + 6
                    ].tolist(),
                    "jointPositionRad": joints.tolist(),
                    "jointVelocityRadPerSecond": data.qvel[joint_qvel_indices].tolist(),
                    "passiveWheelVelocityRadPerSecond": data.qvel[
                        list(passive_wheel_qvel)
                    ].tolist(),
                    "observation": last_observation.tolist(),
                    "rawAction": last_action.tolist(),
                    "targetPositionRad": last_target.tolist(),
                    "command": last_command.tolist(),
                    "contacts": _contacts(model, data),
                },
            )

        metrics = _metrics(
            data,
            root_qpos_address,
            trunk_body_id,
            initial_root,
            joint_min,
            joint_max,
            min_height,
            max_height,
            min_upright,
            cumulative_pitch,
            initial_ball,
            ball_qpos_address,
            finite_state,
        )
        checks = _acceptance_checks(scenario["acceptance"], metrics)
        passed = finite_state and all(check.passed for check in checks)
        result = RolloutResult(
            policy_name=policy_name,
            role=scenario["role"],
            passed=passed,
            physics_steps=physics_steps,
            policy_steps=policy_steps,
            metrics=metrics,
            checks=checks,
            trace_path=destination,
        )
        _write_record(trace, {"recordType": "result", **result.to_dict()})

    return result


def _observation(
    data: mujoco.MjData,
    trunk_body_id: int,
    gyro_address: int,
    joint_qpos_indices: np.ndarray,
    joint_qvel_indices: np.ndarray,
    home: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    quaternion = data.xquat[trunk_body_id].astype(np.float32)
    projected_gravity = _quat_rotate_inverse(
        quaternion, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    )
    observation = np.concatenate(
        (
            data.sensordata[gyro_address : gyro_address + 3].astype(np.float32),
            projected_gravity,
            data.qpos[joint_qpos_indices].astype(np.float32) - home,
            data.qvel[joint_qvel_indices].astype(np.float32),
            last_action,
            command,
        )
    ).astype(np.float32)
    if observation.shape != (OBSERVATION_SIZE,):
        raise RolloutError(f"Observation shape is {observation.shape}, expected (61,)")
    return observation


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion[0]
    xyz = quaternion[1:4]
    cross = np.cross(xyz, vector) * 2.0
    return vector - w * cross + np.cross(xyz, cross)


def _command_at_time(scenario: dict[str, Any], time_seconds: float) -> np.ndarray:
    program = scenario["commandProgram"]
    active = program[0]
    for step in program:
        if float(step["atSeconds"]) <= time_seconds + 1e-12:
            active = step
        else:
            break
    command = np.asarray(active["command"], dtype=np.float32).copy()
    if command.shape != (13,):
        raise RolloutError("Scenario commands must contain 13 values")
    if "phasePeriodSeconds" in active:
        period = float(active["phasePeriodSeconds"])
        phase_end = float(active["phaseEnd"])
        phase = min(max((time_seconds - float(active["atSeconds"])) / period, 0.0), phase_end)
        command[0] = math.cos(math.tau * phase)
        command[1] = math.sin(math.tau * phase)
        command[2] = 0.0
    return command


def _filter_targets(
    target: np.ndarray,
    previous: np.ndarray | None,
    head_indices: frozenset[int],
    head_alpha: float,
    leg_alpha: float,
) -> np.ndarray:
    if previous is None:
        return target.astype(np.float32)
    filtered = target.copy()
    for index in range(ACTION_SIZE):
        alpha = head_alpha if index in head_indices else leg_alpha
        filtered[index] = alpha * target[index] + (1.0 - alpha) * previous[index]
    return filtered.astype(np.float32)


def _passive_wheels(model: mujoco.MjModel) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names: list[str] = []
    qvel_addresses: list[int] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name.startswith("passive_"):
            names.append(name)
            qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
    return tuple(names), tuple(qvel_addresses)


def _place_ball(
    data: mujoco.MjData,
    root_qpos_address: int,
    ball_qpos_address: int,
    ball_qvel_address: int,
    *,
    left: bool,
) -> None:
    x, y = (float(value) for value in data.qpos[root_qpos_address : root_qpos_address + 2])
    qw, qx, qy, qz = data.qpos[root_qpos_address + 3 : root_qpos_address + 7]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    lateral = BALL_OFFSET_ABS_Y if left else -BALL_OFFSET_ABS_Y
    ball_x = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * lateral
    ball_y = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * lateral
    data.qpos[ball_qpos_address : ball_qpos_address + 7] = [
        ball_x,
        ball_y,
        BALL_RADIUS,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qvel[ball_qvel_address : ball_qvel_address + 6] = 0.0


def _metrics(
    data: mujoco.MjData,
    root_qpos_address: int,
    trunk_body_id: int,
    initial_root: np.ndarray,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
    min_height: float,
    max_height: float,
    min_upright: float,
    cumulative_pitch: float,
    initial_ball: np.ndarray | None,
    ball_qpos_address: int | None,
    finite_state: bool,
) -> dict[str, float]:
    final_root = data.qpos[root_qpos_address : root_qpos_address + 3]
    ball_displacement = 0.0
    if initial_ball is not None and ball_qpos_address is not None:
        final_ball = data.qpos[ball_qpos_address : ball_qpos_address + 3]
        ball_displacement = float(np.linalg.norm(final_ball - initial_ball))
    return {
        "forwardDistanceMeters": float(final_root[0] - initial_root[0]),
        "rootDriftMeters": float(np.linalg.norm(final_root[:2] - initial_root[:2])),
        "minimumUprightDot": float(min_upright),
        "finalUprightDot": _upright_dot(data, trunk_body_id),
        "heightExcursionMeters": float(max_height - min_height),
        "jointExcursionRad": float(np.max(joint_max - joint_min)),
        "ballDisplacementMeters": ball_displacement,
        "pitchRotationRad": abs(float(cumulative_pitch)),
        "finiteState": float(finite_state),
    }


def _acceptance_checks(
    acceptance: dict[str, float], metrics: dict[str, float]
) -> tuple[AcceptanceCheck, ...]:
    mapping = {
        "minForwardDistanceMeters": ("forwardDistanceMeters", ">="),
        "minUprightDot": ("minimumUprightDot", ">="),
        "maxRootDriftMeters": ("rootDriftMeters", "<="),
        "minHeightExcursionMeters": ("heightExcursionMeters", ">="),
        "minFinalUprightDot": ("finalUprightDot", ">="),
        "minJointExcursionRad": ("jointExcursionRad", ">="),
        "minBallDisplacementMeters": ("ballDisplacementMeters", ">="),
        "minPitchRotationRad": ("pitchRotationRad", ">="),
    }
    checks: list[AcceptanceCheck] = []
    for name, threshold_value in acceptance.items():
        if name not in mapping:
            raise RolloutError(f"Unsupported acceptance metric {name}")
        metric_name, comparison = mapping[name]
        observed = metrics[metric_name]
        threshold = float(threshold_value)
        passed = observed >= threshold if comparison == ">=" else observed <= threshold
        checks.append(
            AcceptanceCheck(
                name=name,
                observed=observed,
                threshold=threshold,
                comparison=comparison,
                passed=passed,
            )
        )
    return tuple(checks)


def _upright_dot(data: mujoco.MjData, trunk_body_id: int) -> float:
    return float(data.xmat[trunk_body_id].reshape(3, 3)[2, 2])


def _root_pitch(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    matrix_xx = 1.0 - 2.0 * (y * y + z * z)
    matrix_zx = 2.0 * (x * z - w * y)
    return math.atan2(-matrix_zx, matrix_xx)


def _wrapped_angle_difference(current: float, previous: float) -> float:
    return (current - previous + math.pi) % math.tau - math.pi


def _contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        contacts.append(
            {
                "geom1": _geom_name(model, int(contact.geom1)),
                "geom2": _geom_name(model, int(contact.geom2)),
                "distance": float(contact.dist),
            }
        )
    return contacts


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name or f"__geom_{geom_id:03d}"


def _optional_joint_address(model: mujoco.MjModel, name: str, *, qpos: bool) -> int | None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id] if qpos else model.jnt_dofadr[joint_id])


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RolloutError(f"Required {object_type.name} {name!r} is missing")
    return int(object_id)


def _required_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    if name is None:
        raise RolloutError(f"Required {object_type.name} at index {object_id} is unnamed")
    return name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(trace: TextIO, record: dict[str, Any]) -> None:
    trace.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    trace.write("\n")
