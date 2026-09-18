"""Task-space skill residual shared by training and native deployment.

The learned policy never owns an absolute leg pose.  It modifies a continuously
running rolling prior in wheel-end coordinates; deterministic IK and a bounded
target-rate projection produce motor targets.
"""
from __future__ import annotations

import numpy as np

from sim2sim.sai_stair_v3 import observation_numpy as observation_v3_numpy


CONTRACT = "sai-task-space-skills-v6"
ACTION_SIZE = 16
OBSERVATION_SIZE = 258
SIDES = np.array([1.0, -1.0, 1.0, -1.0])
FRONTS = np.array([1.0, 1.0, -1.0, -1.0])
THIGH_M = 0.09
SHIN_M = 0.11
THETA0 = np.arctan2(0.05, 0.074833147)
BETA0 = np.arctan2(0.05, 0.09797959) + THETA0
JOINT_LIMITS = np.tile(np.array([0.45, 0.70, 1.20, np.inf]), 4)
# The physical hip envelope is +/-0.70 rad and deployment rejects measured
# positions at 96% of that limit.  Continuous four-edge contact produced up to
# 0.126 rad of measured hip deflection beyond the requested target, so reserve
# 0.172 rad instead of relying on the simulator's hard joint stop.
JOINT_SOFT_LIMITS = np.tile(np.array([0.30, 0.50, 1.02, np.inf]), 4)
TARGET_SLEW_RAD_S = np.tile(np.array([2.0, 4.0, 6.0, 20.0]), 4)
LOCAL_STEP_THRESHOLD_SCALED = 0.06  # 12 mm after the observation's x5 scale
LOCAL_TERRAIN_INDICES = np.r_[np.arange(56, 95), np.arange(104, 242)]
STAIR_CLEARANCE_M = 0.03
PITCH_GUARD_START_RAD = np.deg2rad(7.0)
PITCH_GUARD_FULL_RAD = np.deg2rad(11.0)


def continuous_stair_heights(x, start: float, tread: float, rise: float,
                             levels: int = 4) -> np.ndarray:
    """Return the collision-matched height of a continuous box staircase.

    The first physical box begins at ``start`` and already has height
    ``rise``.  Keeping this formula shared and directly tested prevents the
    terrain observation from lagging the collision geometry by one tread.
    """
    points = np.asarray(x, dtype=np.float64)
    # Decimal tread boundaries such as -0.10 + 2 * 0.18 can land one ULP
    # below the box boundary.  The terrain grid is 20 mm, so this tiny bias
    # only makes the analytic profile agree with the exact box interval.
    level = np.floor((points - start) / tread + 1e-6).astype(np.int32) + 1
    level = np.clip(level, 0, levels)
    level = np.where(points < start, 0, level)
    return level.astype(np.float32) * np.float32(rise)


def localize_terrain_levels(observation: np.ndarray) -> np.ndarray:
    """Express terrain by the nearest higher/lower level around the supports.

    A four-step staircase otherwise appears as 40/80/120/160 mm while the
    same local maneuver on an isolated step appears as 40/40/40/40 mm.  The
    nearest-level representation makes a learned edge skill translation
    invariant without hiding the sign or height of the immediate edge.
    """
    result = np.asarray(observation, dtype=np.float32).copy()
    if result.shape != (OBSERVATION_SIZE,):
        raise ValueError("Task-space observation must contain 258 values")
    values = result[LOCAL_TERRAIN_INDICES]
    higher = values[values > LOCAL_STEP_THRESHOLD_SCALED]
    lower = values[values < -LOCAL_STEP_THRESHOLD_SCALED]
    if higher.size:
        values = np.minimum(values, float(higher.min()))
    if lower.size:
        values = np.maximum(values, float(lower.max()))
    result[LOCAL_TERRAIN_INDICES] = values
    return result


def project_action_safety(action, projected_up) -> np.ndarray:
    """Project the task pitch residual away from a growing body tilt.

    This is a maneuver-independent equipment envelope.  At small attitude
    errors the policy is untouched.  From 7 to 11 degrees it increasingly
    requires the endpoint pitch command to oppose the measured body pitch.
    """
    result = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).copy()
    up = np.asarray(projected_up, dtype=float)
    if result.shape != (ACTION_SIZE,) or up.shape != (3,):
        raise ValueError("Pitch safety projection expects 16 actions and a 3-vector")
    pitch = float(np.arctan2(-up[0], up[2]))
    correction = float(np.clip(
        (abs(pitch) - PITCH_GUARD_START_RAD)
        / (PITCH_GUARD_FULL_RAD - PITCH_GUARD_START_RAD), 0.0, 1.0))
    if correction > 0.0:
        if pitch > 0.0:
            result[14] = max(result[14], correction)
        elif pitch < 0.0:
            result[14] = min(result[14], -correction)
    return result.astype(np.float32)


def project_target_safety(target) -> np.ndarray:
    """Apply the equipment envelope after every controller contribution.

    Task-space IK already observes these bounds, but suspension is composed
    afterwards.  The final projection prevents that valid correction from
    adding back into a hip or knee target beyond the reserved contact margin.
    """
    result = np.asarray(target, dtype=float).copy()
    if result.shape != (16,):
        raise ValueError("Target safety projection expects 16 motor targets")
    finite = np.isfinite(JOINT_SOFT_LIMITS)
    result[finite] = np.clip(
        result[finite], -JOINT_SOFT_LIMITS[finite], JOINT_SOFT_LIMITS[finite]
    )
    return result.astype(np.float32)


def observation_numpy(state: dict, command, previous_action, base_target) -> np.ndarray:
    """Dense simulator observation plus the rolling prior being modified.

    Exact ray heights intentionally make this a ``godot-oracle-terrain``
    contract.  They must be replaced by a sensor-backed estimator before a
    Sim2Real claim.
    """
    base = observation_v3_numpy(state, command, previous_action)
    prior = np.asarray(base_target, dtype=np.float32)
    if base.shape != (242,) or prior.shape != (16,):
        raise ValueError(f"Invalid v6 observation inputs: {base.shape}, {prior.shape}")
    return localize_terrain_levels(np.concatenate((base, prior)).astype(np.float32))


def _foot_from_joints(hip: np.ndarray, knee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = FRONTS * THETA0 - hip / SIDES
    beta = -FRONTS * BETA0 - knee / SIDES
    dx = THIGH_M * np.sin(theta) + SHIN_M * np.sin(theta + beta)
    down = THIGH_M * np.cos(theta) + SHIN_M * np.cos(theta + beta)
    return dx, down


def _joints_from_foot(dx: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius2 = dx * dx + down * down
    cosine = np.clip((radius2 - THIGH_M**2 - SHIN_M**2) / (2 * THIGH_M * SHIN_M), -1.0, 1.0)
    beta = -FRONTS * np.arccos(cosine)
    theta = np.arctan2(dx, down) - np.arctan2(
        SHIN_M * np.sin(beta), THIGH_M + SHIN_M * np.cos(beta)
    )
    hip = SIDES * (FRONTS * THETA0 - theta)
    knee = SIDES * (-FRONTS * BETA0 - beta)
    return hip, knee


def targets_numpy(action, base_target, previous_residual, dt: float = 0.02) -> np.ndarray:
    """Apply one bounded task-space skill residual to a rolling target.

    Action layout is ``dx[4], dz[4], wheel_speed[4], height, roll, pitch,
    intensity``. Positive ``dz`` lifts a wheel.  Skill intensity is continuous
    in ``[0, 1]`` and zero action therefore leaves the rolling prior unchanged.
    """
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    base = np.asarray(base_target, dtype=float)
    previous = np.asarray(previous_residual, dtype=float)
    if action.shape != (16,) or base.shape != (16,) or previous.shape != (16,):
        raise ValueError("v6 actions and targets must contain 16 values")

    intensity = float(np.clip(action[15], 0.0, 1.0))
    result = base.copy()
    hip = base[1::4]
    knee = base[2::4]
    dx, down = _foot_from_joints(hip, knee)

    dx += intensity * 0.060 * action[0:4]
    lift = np.where(action[4:8] >= 0.0, 0.080 * action[4:8], 0.010 * action[4:8])
    down -= intensity * lift
    down += intensity * 0.025 * action[12]
    down += intensity * SIDES * 0.146 * np.tan(np.deg2rad(8.0) * action[13])
    down += intensity * FRONTS * 0.115 * np.tan(np.deg2rad(12.0) * action[14])

    # Keep the requested endpoint inside the two-link workspace with a margin
    # that avoids the singular fully-extended configuration.
    radius = np.sqrt(dx * dx + down * down)
    scale = np.minimum(0.194 / np.maximum(radius, 1e-8), 1.0)
    dx *= scale
    down = np.clip(down * scale, 0.105, 0.194)
    new_hip, new_knee = _joints_from_foot(dx, down)

    # HAA remains owned by the rolling/stability prior.  Only hip/knee endpoint
    # IK and wheel speed receive learned residuals.
    result[0::4] = np.clip(base[0::4], -0.25, 0.25)
    result[1::4] = new_hip
    result[2::4] = new_knee
    # Policy wheel residuals use body-forward convention.  Right wheel joints
    # have the opposite positive axis, exactly like the rolling prior.
    # The task skill may bias front versus rear traction, while the bounded
    # heading loop exclusively owns left/right differential steering.
    wheel = np.asarray(action[8:12], dtype=float).copy()
    wheel[:2] = wheel[:2].mean()
    wheel[2:] = wheel[2:].mean()
    result[3::4] = base[3::4] + intensity * 6.0 * wheel * SIDES
    finite = np.isfinite(JOINT_SOFT_LIMITS)
    result[finite] = np.clip(result[finite], -JOINT_SOFT_LIMITS[finite], JOINT_SOFT_LIMITS[finite])

    # Rate-limit only what the skill adds.  Limiting the complete target also
    # filters the continuously changing rolling prior, so a nominally zero
    # residual is no longer an identity operation.
    desired_residual = result - base
    delta = np.clip(desired_residual - previous,
                    -TARGET_SLEW_RAD_S * dt, TARGET_SLEW_RAD_S * dt)
    return base + previous + delta
