"""Versioned residual locomotion state, carried in three unused command slots.

The frozen anchor sees zero in slots 58:61 through an ONNX input mask. Only
the new residual sees forward/lateral COM velocity and trunk body height.
Existing models retain the complete original 61-dimensional contract.
"""
from __future__ import annotations

import numpy as np

from sim2sim.coords import quat_wxyz_to_mat

STATE_KEY = 'sim2sim_brake_state_input'
# Retain the historical metadata key/version for existing roller exports.
BRAKE_STATE_V1 = 'planar_com_velocity_height_v1'


def state_input(metadata):
    mode = metadata.get(STATE_KEY, '')
    if mode not in ('', BRAKE_STATE_V1):
        raise ValueError('Unknown policy state input: ' + str(mode))
    return mode


def inject_state(obs, state, mode):
    if not mode:
        return obs
    state_input({STATE_KEY: mode})
    result = np.array(obs, dtype=np.float32, copy=True)
    if result.shape != (61,):
        raise ValueError('State input requires a single 61-dimensional observation')
    rotation = quat_wxyz_to_mat(state.base_quat_wxyz)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = np.asarray(state.base_linvel, np.float64)[:2]
    result[58:61] = [c*vx+s*vy, -s*vx+c*vy, state.base_pos[2]]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite policy state input')
    return result


def wrap_anchor(source, destination):
    """Preserve every original node and DOUBLE tensor; mask only its new input."""
    import hashlib
    from pathlib import Path
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(source)
    meta = {p.key: p.value for p in model.metadata_props}
    if state_input(meta):
        raise ValueError('Anchor already declares residual state input')
    if len(model.graph.input) != 1:
        raise ValueError('Expected one actor input')
    original_input = model.graph.input[0].name
    names = {n for node in model.graph.node for n in [*node.input, *node.output]}
    new_input, mask_name = 'residual_state/obs', 'residual_state/anchor_mask'
    if {new_input, mask_name} & names:
        raise ValueError('State wrapper name collision')
    model.graph.input[0].name = new_input
    mask = np.ones((1, 61), np.float32); mask[:, 58:61] = 0
    model.graph.initializer.append(numpy_helper.from_array(mask, mask_name))
    model.graph.node.insert(0, helper.make_node('Mul', [new_input, mask_name], [original_input]))
    meta[STATE_KEY] = BRAKE_STATE_V1
    meta['sim2sim_state_anchor_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    del model.metadata_props[:]
    for key, value in meta.items():
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return destination
