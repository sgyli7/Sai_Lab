"""A bounded angular-velocity feedback residual active only during braking."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .brake_pose_search import action_offset
from .models import NativeAnchor


def prediction(anchor, observations, parameters):
    x = np.asarray(observations, np.float32).reshape(-1, 61)
    rate = np.clip(x[:, 1:2] / np.float32(2.), -1., 1.)
    brake = np.clip(-x[:, 48:49] / np.float32(.05), 0., 1.)
    return anchor(x) + (brake * rate) * action_offset(parameters)


def export(source, parameters, target):
    anchor = NativeAnchor(source)
    offset = action_offset(parameters)
    model = onnx.load(source)
    prefix = 'brake_pitch_rate/'
    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    while any(name.startswith(prefix) for name in names):
        prefix += 'next/'
    def n(label): return prefix + label
    constants = dict(rate_index=np.array([1], np.int64), throttle_index=np.array([48], np.int64),
        rate_scale=np.array(2., np.float32), throttle_scale=np.array(-.05, np.float32),
        zero=np.array(0., np.float32), one=np.array(1., np.float32),
        minus_one=np.array(-1., np.float32), offset=offset.reshape(1, 14))
    for label, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, n(label)))
    original = model.graph.output[0].name
    input_name = model.graph.input[0].name
    model.graph.node.extend([
        helper.make_node('Gather', [input_name, n('rate_index')], [n('rate')], axis=1),
        helper.make_node('Div', [n('rate'), n('rate_scale')], [n('normalized_rate')]),
        helper.make_node('Clip', [n('normalized_rate'), n('minus_one'), n('one')], [n('bounded_rate')]),
        helper.make_node('Gather', [input_name, n('throttle_index')], [n('throttle')], axis=1),
        helper.make_node('Div', [n('throttle'), n('throttle_scale')], [n('negative_fraction')]),
        helper.make_node('Clip', [n('negative_fraction'), n('zero'), n('one')], [n('brake_gate')]),
        helper.make_node('Mul', [n('brake_gate'), n('bounded_rate')], [n('gate')]),
        helper.make_node('Mul', [n('gate'), n('offset')], [n('delta')]),
        helper.make_node('Add', [original, n('delta')], [n('actions')]),
    ])
    model.graph.output[0].name = n('actions')
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata['sim2sim_brake_pitch_rate'] = json.dumps(dict(parent_sha256=anchor.sha256,
        parameters=np.asarray(parameters).tolist(), rate_index=1, rate_scale=2.))
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    x = np.random.default_rng(915007).normal(0, .3, (512, 61)).astype(np.float32)
    x[:64, 48] = 0.
    x[64:128, 1] = 0.
    actual = NativeAnchor(target)(x)
    error = float(np.max(abs(actual - prediction(anchor, x, parameters))))
    preserved = (x[:, 48] >= 0) | (x[:, 1] == 0)
    exact = np.array_equal(actual[preserved], anchor(x[preserved]))
    if error >= 1e-5 or not exact:
        raise RuntimeError('Pitch-rate residual export parity failed')
    return dict(max_abs=error, nonbraking_zero_rate_exact=exact,
        sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())
