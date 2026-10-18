"""Measured-link Jacobian for an explicitly privileged offline TRAIN collector.

This module is not imported by the LingBot inference service. All link/actor
truth is restricted to data collection and CPU kinematics validation.
"""
import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw
from experiments.lingbot_vla2_g1.contract import UPPER_NAMES, WAIST_NAMES


def rotation_wxyz(quaternion):
    w, x, y, z = quaternion
    return rotation_xyzw([0, 0, 0, x, y, z, w])


def rotation_vector(matrix):
    v = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]) / 2
    sine = np.linalg.norm(v)
    angle = np.arctan2(sine, np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    return v * (angle / sine) if sine > 1e-10 else v


class MeasuredJacobian:
    def __init__(self, geometry, actual_body_names):
        self.geometry = geometry
        self.names = actual_body_names
        self.body_id = {name: index for index, name in enumerate(actual_body_names)}
        self.joints = {joint['name']: joint for joint in geometry['joints']}

    def palm_pose(self, row, side):
        return np.asarray(row['acceptance_robot_body_link_poses_w'][self.body_id[side + '_hand_palm_link']], float)

    def jacobian(self, row, side, include_waist=False):
        names = list(UPPER_NAMES[:7] if side == 'left' else UPPER_NAMES[7:14])
        if include_waist:
            names += list(WAIST_NAMES)
        palm = self.palm_pose(row, side)[:3]
        values = []
        for name in names:
            joint = self.joints[name]
            child_name = self.geometry['bodies'][joint['child']]['name']
            child = np.asarray(row['acceptance_robot_body_link_poses_w'][self.body_id[child_name]], float)
            rotation = rotation_xyzw(child)
            frame = joint['frame_child']
            pivot = child[:3] + rotation @ np.asarray(frame['position'])
            axis_local = np.eye(3)['XYZ'.index(joint['axis'])]
            axis_world = rotation @ rotation_wxyz(frame['rotation_wxyz']) @ axis_local
            values.append(np.r_[np.cross(axis_world, palm - pivot), axis_world])
        return np.stack(values, axis=1), names

    def position_orientation_step(self, row, side, desired_position, desired_rotation, damping=.03, max_joint_step=.10):
        palm = self.palm_pose(row, side)
        jacobian, names = self.jacobian(row, side)
        error = np.r_[np.asarray(desired_position) - palm[:3],
                      .2 * rotation_vector(np.asarray(desired_rotation) @ rotation_xyzw(palm).T)]
        weighted = jacobian.copy()
        weighted[3:] *= .2
        change = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping ** 2 * np.eye(6), error)
        change = np.clip(change, -max_joint_step, max_joint_step)
        q = np.asarray([row['joint_positions'][row['source_joint_names'].index(name)] for name in names])
        return q + change
