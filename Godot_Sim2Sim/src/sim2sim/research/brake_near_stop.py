"""Test extra velocity feedback close to rest while preserving faster braking."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim2sim.policy_state import BRAKE_STATE_V1
from .brake_pose_search import action_offset
from .models import NativeAnchor


def prediction(anchor, observations, parameters):
    x = np.asarray(observations, np.float32).reshape(-1, 61)
    velocity = x[:, 58:59]
    near = np.clip((np.float32(.15) - np.abs(velocity)) / np.float32(.10), 0., 1.)
    error = np.clip(velocity / np.float32(.05), -1., 1.)
    brake = np.clip(-x[:, 48:49] / np.float32(.05), 0., 1.)
    return anchor(x) + ((brake * near) * error) * action_offset(parameters)


def export(source, parameters, target):
    anchor = NativeAnchor(source)
    if anchor.state_input != BRAKE_STATE_V1:
        raise ValueError('Near-stop feedback requires the declared forward-velocity input')
    offset = action_offset(parameters)
    model = onnx.load(source)
    prefix = 'brake_near_stop/'
    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    while any(name.startswith(prefix) for name in names):
        prefix += 'next/'
    def n(label): return prefix + label
    constants = dict(velocity_index=np.array([58], np.int64), throttle_index=np.array([48], np.int64),
        range=np.array(.15, np.float32), width=np.array(.10, np.float32), scale=np.array(.05, np.float32),
        throttle_scale=np.array(-.05, np.float32), zero=np.array(0., np.float32),
        one=np.array(1., np.float32), minus_one=np.array(-1., np.float32), offset=offset.reshape(1, 14))
    for label, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, n(label)))
    original = model.graph.output[0].name
    input_name = model.graph.input[0].name
    model.graph.node.extend([
        helper.make_node('Gather', [input_name, n('velocity_index')], [n('velocity')], axis=1),
        helper.make_node('Abs', [n('velocity')], [n('absolute_velocity')]),
        helper.make_node('Sub', [n('range'), n('absolute_velocity')], [n('deficit')]),
        helper.make_node('Div', [n('deficit'), n('width')], [n('normalized_near')]),
        helper.make_node('Clip', [n('normalized_near'), n('zero'), n('one')], [n('near')]),
        helper.make_node('Div', [n('velocity'), n('scale')], [n('normalized_velocity')]),
        helper.make_node('Clip', [n('normalized_velocity'), n('minus_one'), n('one')], [n('error')]),
        helper.make_node('Gather', [input_name, n('throttle_index')], [n('throttle')], axis=1),
        helper.make_node('Div', [n('throttle'), n('throttle_scale')], [n('negative_fraction')]),
        helper.make_node('Clip', [n('negative_fraction'), n('zero'), n('one')], [n('brake_gate')]),
        helper.make_node('Mul', [n('brake_gate'), n('near')], [n('near_brake')]),
        helper.make_node('Mul', [n('near_brake'), n('error')], [n('gate')]),
        helper.make_node('Mul', [n('gate'), n('offset')], [n('delta')]),
        helper.make_node('Add', [original, n('delta')], [n('actions')]),
    ])
    model.graph.output[0].name = n('actions')
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata['sim2sim_brake_near_stop'] = json.dumps(dict(parent_sha256=anchor.sha256,
        parameters=np.asarray(parameters).tolist(), velocity_range=.15, velocity_width=.10, velocity_scale=.05))
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    x = np.random.default_rng(915003).normal(0, .3, (512, 61)).astype(np.float32)
    x[:, 58] = np.linspace(-.3, .3, len(x))
    x[:64, 48] = 0.
    x[64:128, 58] = 0.
    actual = NativeAnchor(target)(x)
    error = float(np.max(abs(actual - prediction(anchor, x, parameters))))
    preserved = (x[:, 48] >= 0) | (x[:, 58] == 0) | (np.abs(x[:, 58]) >= np.float32(.15))
    exact = np.array_equal(actual[preserved], anchor(x[preserved]))
    if error >= 1e-5 or not exact:
        raise RuntimeError('Near-stop feedback export parity failed')
    return dict(max_abs=error, nonbraking_zero_and_high_speed_exact=exact,
        sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())
