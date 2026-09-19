"""Offline decomposition of Jolt hinge constraint errors from same-frame poses.

This is kinematics, not a physical solver or an online policy correction. Each
snapshot uses inertial COM frames as reported by the frozen Godot runtime.
"""
from dataclasses import dataclass
import numpy as np
from sim2sim.coords import quat_wxyz_to_mat, mat_to_quat_wxyz


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    return quat_wxyz_to_mat(q / norm)


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return rotation(np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)])


def twist_angle(matrix, axis):
    q = mat_to_quat_wxyz(matrix)
    angle = 2 * np.arctan2(np.dot(q[1:], axis), q[0])
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Hinge:
    name: str
    parent: str
    child: str
    actuator: int
    axis: np.ndarray
    anchor_parent: np.ndarray
    anchor_child: np.ndarray
    rest: np.ndarray


class JointResidualModel:
    def __init__(self, spec):
        self.bodies = {b['name']: b for b in spec['bodies']}
        self.inertial_rotation = {name: rotation(b['iquat_wxyz']) for name, b in self.bodies.items()}
        initial = {}
        for name, b in self.bodies.items():
            r = np.asarray(b['ximat']).reshape(3, 3) @ self.inertial_rotation[name].T
            initial[name] = (np.asarray(b['xipos']) - r @ b['ipos'], r)
        actuators = {a['joint']: int(a['id']) for a in spec['actuators']}
        self.hinges = []
        for j in spec['joints']:
            if j['type'] != 'hinge':
                continue
            parent, child = j['parent'], j['body']
            pp, rp = initial[parent]; pc, rc = initial[child]
            axis = np.asarray(j['axis_parent_body'], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            anchor = np.asarray(j['anchor_world'])
            self.hinges.append(Hinge(j['name'], parent, child, actuators[j['name']], axis,
                                     rp.T @ (anchor - pp), rc.T @ (anchor - pc),
                                     rotation(j['rest_rel_q0_wxyz'])))

    def body_poses(self, poses):
        result = {}
        for pose in poses:
            name = pose['name']
            r = rotation(pose['quat']) @ self.inertial_rotation[name].T
            result[name] = (np.asarray(pose['pos']) - r @ self.bodies[name]['ipos'], r)
        needed = {name for h in self.hinges for name in [h.parent, h.child]}
        if not needed.issubset(result):
            raise ValueError('Missing constrained-body poses: ' + str(sorted(needed - result.keys())))
        return result

    def measure(self, body_poses, angles):
        translations, swings, recovered, relatives = [], [], [], []
        for h in self.hinges:
            pp, rp = body_poses[h.parent]; pc, rc = body_poses[h.child]
            translations.append(rp.T @ (pc + rc @ h.anchor_child - pp) - h.anchor_parent)
            relative = rp.T @ rc
            delta = relative @ h.rest.T
            q = twist_angle(delta, h.axis)
            swing = delta @ axis_rotation(h.axis, q).T
            qs = mat_to_quat_wxyz(swing)
            if qs[0] < 0:
                qs = -qs
            length = np.linalg.norm(qs[1:])
            swings.append(qs[1:] * (2 * np.arctan2(length, qs[0]) / length) if length > 1e-15 else np.zeros(3))
            recovered.append(q); relatives.append(relative)
        expected = np.asarray([angles[h.actuator] for h in self.hinges])
        error = (np.asarray(recovered) - expected + np.pi) % (2 * np.pi) - np.pi
        return dict(translation_parent=np.asarray(translations), swing_parent=np.asarray(swings),
                    recovered_q=np.asarray(recovered), q_error=error, relative_rotation=np.asarray(relatives))

    def reconstruct(self, body_poses, angles, measured, *, translation, swing):
        """Counterfactual geometry only; roots retain their measured transforms."""
        result = dict(body_poses)
        for i, h in enumerate(self.hinges):
            pp, rp = result[h.parent]
            relative = measured['relative_rotation'][i] if swing else axis_rotation(h.axis, angles[h.actuator]) @ h.rest
            rc = rp @ relative
            offset = measured['translation_parent'][i] if translation else np.zeros(3)
            pc = pp + rp @ (h.anchor_parent + offset) - rc @ h.anchor_child
            result[h.child] = (pc, rc)
        return {name: (p + r @ self.bodies[name]['ipos'], r @ self.inertial_rotation[name])
                for name, (p, r) in result.items()}
