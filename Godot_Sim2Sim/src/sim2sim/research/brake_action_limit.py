"""Export a brake-only action-range experiment without changing actuator physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim2sim.paths import load_robot_json, sim2sim_root
from sim2sim.research.models import NativeAnchor


def action_bounds(margin=0.):
    if not 0 <= margin <= .15:
        raise ValueError('Margin must be between zero and .15 radians')
    cfg=load_robot_json(sim2sim_root()/'robots/microduck_roller.json')
    spec=json.loads(Path(cfg['godot_spec']).read_text())
    joints={j['id']:j for j in spec['joints']}
    ranges=np.array([joints[a['joint_id']]['range'] for a in spec['actuators']],np.float32)
    home=np.array(cfg['home'],np.float32)
    low=(ranges[:,0]+margin-home)/np.float32(cfg['action_scale'])
    high=(ranges[:,1]-margin-home)/np.float32(cfg['action_scale'])
    if np.any(low>=high):raise ValueError('Empty action interval')
    return low,high


def export(source,target,margin=0.,blend=1.):
    if not 0 < blend <= 1:raise ValueError('Blend must be in (0,1]')
    low,high=action_bounds(margin)
    model=onnx.load(source);original=model.graph.output[0].name
    names={n for node in model.graph.node for n in [*node.input,*node.output]}
    prefix='brake_action_limit/'
    if any(n.startswith(prefix) for n in names):raise ValueError('Already wrapped')
    def name(s):return prefix+s
    constants={'index':np.array([48],np.int64),'low':low[None],'high':high[None],
        'scale':np.array(-20.,np.float32),'zero':np.array(0.,np.float32),
        'one':np.array(1.,np.float32),'blend':np.array(blend,np.float32)}
    model.graph.initializer.extend(numpy_helper.from_array(v,name(k)) for k,v in constants.items())
    model.graph.node.extend([
        helper.make_node('Gather',[model.graph.input[0].name,name('index')],[name('throttle')],axis=1),
        helper.make_node('Mul',[name('throttle'),name('scale')],[name('weight')]),
        helper.make_node('Clip',[name('weight'),name('zero'),name('one')],[name('gate')]),
        helper.make_node('Max',[original,name('low')],[name('lower_bounded')]),
        helper.make_node('Min',[name('lower_bounded'),name('high')],[name('bounded')]),
        helper.make_node('Sub',[name('bounded'),original],[name('change')]),
        helper.make_node('Mul',[name('change'),name('blend')],[name('blended')]),
        helper.make_node('Mul',[name('blended'),name('gate')],[name('delta')]),
        helper.make_node('Add',[original,name('delta')],[name('actions')]),
    ])
    model.graph.output[0].name=name('actions')
    meta={p.key:p.value for p in model.metadata_props}
    meta.update(sim2sim_brake_action_limit=json.dumps(dict(margin=margin,blend=blend,
        low=low.tolist(),high=high.tolist(),parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())))
    helper.set_model_props(model,meta);onnx.checker.check_model(model)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,target)
    old,new=NativeAnchor(source),NativeAnchor(target)
    x=np.random.default_rng(915006).normal(size=(256,61)).astype(np.float32)
    x[:64,48]=0.;base=old(x)
    expected=base+np.clip(-20*x[:,48:49],0,1)*(np.clip(base,low,high)-base)*np.float32(blend)
    actual=new(x);error=float(np.max(np.abs(actual-expected)))
    preserved=x[:,48]>=0
    assert error<1e-5 and np.array_equal(actual[preserved],base[preserved])
    return dict(max_abs=error,positive_and_neutral_exact=True,margin=margin,blend=blend,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('target')
    p.add_argument('--margin',type=float,default=0.);p.add_argument('--blend',type=float,default=1.)
    a=p.parse_args();print(json.dumps(export(a.source,a.target,a.margin,a.blend)))
