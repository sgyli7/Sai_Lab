"""Test height feedback around a declared stable brake stance, inside ONNX."""
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
    height_error = np.clip((x[:, 60:61] - np.float32(.09)) / np.float32(.02), -1., 1.)
    brake = np.clip(-x[:, 48:49] / np.float32(.05), 0., 1.)
    return anchor(x) + (brake * height_error) * action_offset(parameters)


def export(source, parameters, target):
    anchor = NativeAnchor(source)
    if anchor.state_input != BRAKE_STATE_V1:
        raise ValueError('Height feedback requires the declared body-height input')
    offset = action_offset(parameters)
    model = onnx.load(source)
    prefix = 'brake_height_feedback/'
    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    while any(name.startswith(prefix) for name in names):
        prefix += 'next/'
    def n(label): return prefix + label
    constants = dict(height_index=np.array([60], np.int64), throttle_index=np.array([48], np.int64),
        height_reference=np.array(.09, np.float32), height_scale=np.array(.02, np.float32),
        throttle_scale=np.array(-.05, np.float32), zero=np.array(0., np.float32),
        one=np.array(1., np.float32), minus_one=np.array(-1., np.float32), offset=offset.reshape(1, 14))
    for label, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, n(label)))
    original = model.graph.output[0].name
    input_name = model.graph.input[0].name
    model.graph.node.extend([
        helper.make_node('Gather', [input_name, n('height_index')], [n('height')], axis=1),
        helper.make_node('Sub', [n('height'), n('height_reference')], [n('error')]),
        helper.make_node('Div', [n('error'), n('height_scale')], [n('normalized_error')]),
        helper.make_node('Clip', [n('normalized_error'), n('minus_one'), n('one')], [n('bounded_error')]),
        helper.make_node('Gather', [input_name, n('throttle_index')], [n('throttle')], axis=1),
        helper.make_node('Div', [n('throttle'), n('throttle_scale')], [n('negative_fraction')]),
        helper.make_node('Clip', [n('negative_fraction'), n('zero'), n('one')], [n('brake_gate')]),
        helper.make_node('Mul', [n('brake_gate'), n('bounded_error')], [n('gate')]),
        helper.make_node('Mul', [n('gate'), n('offset')], [n('delta')]),
        helper.make_node('Add', [original, n('delta')], [n('actions')]),
    ])
    model.graph.output[0].name = n('actions')
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata['sim2sim_brake_height_feedback'] = json.dumps(dict(parent_sha256=anchor.sha256,
        parameters=np.asarray(parameters).tolist(), height_reference=.09, height_scale=.02))
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    x = np.random.default_rng(915004).normal(0, .3, (512, 61)).astype(np.float32)
    x[:, 60] = np.linspace(.075, .14, len(x))
    x[:64, 48] = 0.
    x[64:128, 60] = np.float32(.09)
    actual = NativeAnchor(target)(x)
    error = float(np.max(abs(actual - prediction(anchor, x, parameters))))
    preserved = (x[:, 48] >= 0) | (x[:, 60] == np.float32(.09))
    exact = np.array_equal(actual[preserved], anchor(x[preserved]))
    if error >= 1e-5 or not exact:
        raise RuntimeError('Height feedback export parity failed')
    return dict(max_abs=error, nonbraking_reference_height_exact=exact,
        sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())
