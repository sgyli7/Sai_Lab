"""Deterministic low-level impedance used by the v6 task-space skill policy."""
from __future__ import annotations

import numpy as np


SIDES = np.array([1., -1., 1., -1.])
FRONTS = np.array([1., 1., -1., -1.])
LEG_AXES = np.array([i for i in range(16) if i % 4 != 3])
FLEX_AXES = np.array([1, 2, 5, 6, 9, 10, 13, 14])
SUPPORT_RIDGE = np.array([1e-8, 1e-6, 1e-6])


def support_weights(contact: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Allocate compressive support while matching force and CoM moment.

    The small diagonal term gives the same least-squares behavior when only
    one axle is in contact.  Without it the native Godot implementation
    dropped all feedforward support exactly while a wheel pair crossed an
    edge, whereas MuJoCo's pseudoinverse kept a finite solution.
    """
    contact = np.asarray(contact, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if contact.shape != (4,) or xy.shape != (4, 2):
        raise ValueError("Support allocation expects four contacts and 2-D offsets")
    if not np.isfinite(contact).all() or not np.isfinite(xy).all():
        raise ValueError("Support allocation inputs must be finite")
    A = np.vstack([np.ones(4), xy.T])
    active = contact > 1e-4
    weights = np.zeros(4)
    for _ in range(4):
        if not np.any(active):
            break
        B = A[:, active]
        c = contact[active]
        matrix = (B * c) @ B.T + np.diag(SUPPORT_RIDGE)
        solution = np.linalg.solve(matrix, np.array([1., 0., 0.]))
        values = c * (B.T @ solution)
        if np.min(values) >= -1e-8:
            weights[active] = np.maximum(values, 0.)
            break
        ids = np.flatnonzero(active)
        active[ids[np.argmin(values)]] = False
    if weights.sum() > 0:
        weights *= min(1., float(contact.sum())) / weights.sum()
    return weights


class TaskSpaceImpedance:
    """Soft stance gains plus analytic vertical support feedforward.

    The formula depends only on public controller state and robot constants, so
    MuJoCo training, CPU evaluation, and Godot can execute the same low layer.
    """

    def __init__(self, controller):
        self.controller = controller
        self.stance = None
        self.height_reference = None
        self.previous_feedforward = np.zeros(16)
        self.robot_mass = float(controller.model.body_subtreemass[controller.model.body("chassis").id])

    def apply(self, result: dict, state: dict) -> dict:
        ground = np.asarray(state.get("wheel_ground_heights", []), dtype=float)
        wheels = np.asarray(state.get("wheel_positions", []), dtype=float)
        if ground.shape != (4,) or wheels.shape != (4, 3):
            return result
        gap = wheels[:, 2] - ground - .048
        contact = np.clip(1. - (gap - .002) / .01, 0., 1.)
        if self.stance is None:
            self.stance = np.ones(4)
        self.stance += np.clip(contact - self.stance, -.25, .25)
        contact = self.stance

        gains = np.full(16, 80.)
        damping = np.full(16, 2.)
        gains[FLEX_AXES] = 80. + (45.05744684106064 - 80.) * np.repeat(contact, 2)
        damping[FLEX_AXES] = 2. + (1.1532485502878331 - 2.) * np.repeat(contact, 2)
        desired_height = float(ground.mean() + .2192)
        if self.height_reference is None:
            self.height_reference = desired_height
        delta = (desired_height - self.height_reference) * (.02 / (.15023543636516334 + .02))
        self.height_reference += delta
        force = (self.robot_mass * 9.81 * .9497251199685409
                 + 473.84859697921144 * (self.height_reference - float(state["base_position"][2]))
                 + 49.991230469877635 * (delta / .02 - float(state["base_linear_world"][2])))
        force = float(np.clip(force, 0., self.robot_mass * 9.81 * 1.5))
        com = np.asarray(state.get("robot_com_position", state["base_position"]), dtype=float)
        weights = support_weights(contact, wheels[:, :2] - com[:2])

        q = np.asarray(state["q"], dtype=float)[:16]
        theta0 = np.arctan2(.05, .074833147)
        beta0 = np.arctan2(.05, .09797959) + theta0
        theta = FRONTS * theta0 - q[1::4] / SIDES
        beta = -FRONTS * beta0 - q[2::4] / SIDES
        dx = .09 * np.sin(theta) + .11 * np.sin(theta + beta)
        jac_hip = -dx / SIDES
        jac_knee = -.11 * np.sin(theta + beta) / SIDES
        feedforward = np.zeros(16)
        feedforward[1::4] = -jac_hip * force * weights
        feedforward[2::4] = -jac_knee * force * weights
        bias = self.controller.data.qfrc_bias[self.controller.adapter.vadr[:16]]
        feedforward[LEG_AXES] += .9497251199685409 * bias[LEG_AXES] * np.repeat(contact, 3)
        self.previous_feedforward += (feedforward - self.previous_feedforward) * (.02 / .06)
        result.update(
            leg_kp=gains.tolist(), leg_kd=damping.tolist(),
            leg_feedforward=self.previous_feedforward.tolist(),
            impedance_contract="sai-joint-impedance-v1",
            support_force_N=force, stance_weights=weights.tolist(),
        )
        return result
