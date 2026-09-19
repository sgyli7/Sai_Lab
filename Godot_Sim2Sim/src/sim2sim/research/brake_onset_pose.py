"""A brake residual restricted to low-speed, high-body-height starting states."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper,numpy_helper

from sim2sim.policy_state import BRAKE_STATE_V1
from .brake_pose_search import action_offset
from .models import NativeAnchor


def prediction(anchor,observations,parameters):
    x=np.asarray(observations,np.float32).reshape(-1,61)
    height=np.clip((x[:,60:61]-np.float32(.1))/np.float32(.015),0.,1.)
    low_speed=np.clip((np.float32(.6)-x[:,58:59])/np.float32(.15),0.,1.)
    brake=np.clip(-x[:,48:49]/np.float32(.05),0.,1.)
    return anchor(x)+(brake*height*low_speed)*action_offset(parameters)


def export(source,parameters,target):
    anchor=NativeAnchor(source)
    if anchor.state_input!=BRAKE_STATE_V1:raise ValueError('Onset residual requires velocity and height input')
    offset=action_offset(parameters);model=onnx.load(source);prefix='brake_onset/'
    names={name for node in model.graph.node for name in [*node.input,*node.output]}
    while any(name.startswith(prefix) for name in names):prefix+='next/'
    def n(label):return prefix+label
    constants=dict(height_index=np.array([60],np.int64),velocity_index=np.array([58],np.int64),
        throttle_index=np.array([48],np.int64),height_start=np.array(.1,np.float32),height_width=np.array(.015,np.float32),
        velocity_upper=np.array(.6,np.float32),velocity_width=np.array(.15,np.float32),
        throttle_scale=np.array(-.05,np.float32),zero=np.array(0.,np.float32),one=np.array(1.,np.float32),
        offset=offset.reshape(1,14))
    for label,value in constants.items():model.graph.initializer.append(numpy_helper.from_array(value,n(label)))
    original=model.graph.output[0].name;input_name=model.graph.input[0].name
    model.graph.node.extend([
        helper.make_node('Gather',[input_name,n('height_index')],[n('height')],axis=1),
        helper.make_node('Sub',[n('height'),n('height_start')],[n('excess_height')]),
        helper.make_node('Div',[n('excess_height'),n('height_width')],[n('height_fraction')]),
        helper.make_node('Clip',[n('height_fraction'),n('zero'),n('one')],[n('height_gate')]),
        helper.make_node('Gather',[input_name,n('velocity_index')],[n('velocity')],axis=1),
        helper.make_node('Sub',[n('velocity_upper'),n('velocity')],[n('velocity_deficit')]),
        helper.make_node('Div',[n('velocity_deficit'),n('velocity_width')],[n('velocity_fraction')]),
        helper.make_node('Clip',[n('velocity_fraction'),n('zero'),n('one')],[n('velocity_gate')]),
        helper.make_node('Gather',[input_name,n('throttle_index')],[n('throttle')],axis=1),
        helper.make_node('Div',[n('throttle'),n('throttle_scale')],[n('negative_fraction')]),
        helper.make_node('Clip',[n('negative_fraction'),n('zero'),n('one')],[n('brake_gate')]),
        helper.make_node('Mul',[n('brake_gate'),n('height_gate')],[n('braking_high')]),
        helper.make_node('Mul',[n('braking_high'),n('velocity_gate')],[n('gate')]),
        helper.make_node('Mul',[n('gate'),n('offset')],[n('delta')]),
        helper.make_node('Add',[original,n('delta')],[n('actions')]),
    ])
    model.graph.output[0].name=n('actions')
    metadata={item.key:item.value for item in model.metadata_props}
    metadata['sim2sim_brake_onset_pose']=json.dumps(dict(parent_sha256=anchor.sha256,
        parameters=np.asarray(parameters).tolist(),height_start=.1,height_width=.015,velocity_upper=.6,velocity_width=.15))
    helper.set_model_props(model,metadata);onnx.checker.check_model(model);onnx.save(model,target)
    x=np.random.default_rng(915006).normal(0,.3,(512,61)).astype(np.float32)
    x[:,60]=np.linspace(.075,.14,len(x));x[:64,48]=0.;x[64:128,48]=.6
    expected=prediction(anchor,x,parameters);actual=NativeAnchor(target)(x)
    error=float(np.max(abs(expected-actual)));preserved=(x[:,48]>=0)|(x[:,60]<=.1)|(x[:,58]>=.6)
    exact=np.array_equal(actual[preserved],anchor(x[preserved]))
    if error>=1e-5 or not exact:raise RuntimeError('Onset residual export parity failed')
    return dict(max_abs=error,nonbraking_low_height_high_speed_exact=exact,
        sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())
