"""An explicit two-expert neural policy contained entirely in one ONNX.

This architecture is reported separately from monolithic distillation. A time
input blends frozen roll and balance networks inside the graph. The simulator
executes this single actor throughout; no external policy takes control.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import compose,helper,numpy_helper
from .tasks import TASKS


def build(roll_source,dest,start=1.9,end=2.1):
    if not 0<=start<end<=5:raise ValueError('Invalid five-second maneuver blend')
    sources=[Path(roll_source),TASKS['standing'].source]
    originals=[onnx.load(p) for p in sources]
    if any(float(next((p.value for p in m.metadata_props if p.key=='sim2sim_time_input_s'),'0')) for m in originals):
        raise ValueError('These experts require their original zero-command contract')
    models=[compose.add_prefix(m,prefix) for m,prefix in zip(originals,['roll/','balance/'])]
    mask=np.ones((1,61),np.float32);mask[:,48:]=0
    arrays={'expert_mask':mask,'time_start':np.array([48],np.int64),'time_end':np.array([49],np.int64),
        'feature_axis':np.array([1],np.int64),'five':np.array(5.,np.float32),'start':np.array(start,np.float32),
        'width':np.array(end-start,np.float32),'zero':np.array(0.,np.float32),'one':np.array(1.,np.float32)}
    nodes=[helper.make_node('Mul',['obs','expert_mask'],['expert_obs']),
        helper.make_node('Slice',['obs','time_start','time_end','feature_axis'],['time_fraction']),
        helper.make_node('Mul',['time_fraction','five'],['seconds']),helper.make_node('Sub',['seconds','start'],['after_start']),
        helper.make_node('Div',['after_start','width'],['unclipped_gate']),helper.make_node('Clip',['unclipped_gate','zero','one'],['gate']),
        helper.make_node('Sub',['one','gate'],['roll_weight'])]
    for m in models:
        old=m.graph.input[0].name
        for node in m.graph.node:
            for i,n in enumerate(node.input):
                if n==old:node.input[i]='expert_obs'
        nodes+=list(m.graph.node)
    nodes += [helper.make_node('Mul',[models[0].graph.output[0].name,'roll_weight'],['roll_action']),
        helper.make_node('Mul',[models[1].graph.output[0].name,'gate'],['balance_action']),
        helper.make_node('Add',['roll_action','balance_action'],['actions'])]
    graph=helper.make_graph(nodes,'explicit_neural_expert_mixture',
        [helper.make_tensor_value_info('obs',onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info('actions',onnx.TensorProto.FLOAT,[1,14])],
        initializer=[v for m in models for v in m.graph.initializer]+[numpy_helper.from_array(v,k) for k,v in arrays.items()])
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]);m.ir_version=10
    meta=dict(sim2sim_task='roulade',sim2sim_time_input_s='5',sim2sim_command_mode='one_shot_time',
        sim2sim_model_architecture='explicit_two_neural_experts',sim2sim_blend_seconds=json.dumps([start,end]),
        sim2sim_experts=json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]))
    for k,v in meta.items():m.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(m);dest=Path(dest);dest.parent.mkdir(exist_ok=True,parents=True);onnx.save(m,str(dest));return dest


def parity(roll_source,exported,start,end,n=10000):
    from .models import NativeAnchor
    x=np.random.default_rng(931).normal(0,.5,(n,61)).astype(np.float32);z=x.copy();z[:,48:]=0
    gate=np.clip((x[:,48:49]*np.float32(5)-np.float32(start))/np.float32(end-start),0,1)
    expected=NativeAnchor(roll_source)(z)*(1-gate)+NativeAnchor(TASKS['standing'].source)(z)*gate
    error=float(np.max(np.abs(expected-NativeAnchor(exported)(x))))
    return dict(samples=n,max_abs=error,threshold=1e-5,passed=error<1e-5)


def retime(source,dest,start,end):
    """Change only declared expert-blend constants, retaining learned branches."""
    if not 0<=start<end<=5:raise ValueError('Invalid blend')
    model=onnx.load(source);meta={p.key:p.value for p in model.metadata_props}
    old_start,old_end=json.loads(meta['sim2sim_blend_seconds'])
    counts={}
    for key,value,prior in [('start',start,old_start),('width',end-start,old_end-old_start)]:
        candidates=[v for v in model.graph.initializer if v.name==key or v.name.endswith('/'+key)]
        if not candidates or any(not np.allclose(numpy_helper.to_array(v),prior,atol=1e-7) for v in candidates):
            raise ValueError('Ambiguous expert-blend constants')
        for v in candidates:v.CopyFrom(numpy_helper.from_array(np.array(value,np.float32),v.name))
        counts[key]=len(candidates)
    if counts['start']!=counts['width']:raise ValueError('Unpaired blend constants')
    meta.update(sim2sim_blend_seconds=json.dumps([start,end]),
        sim2sim_retimed_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    del model.metadata_props[:]
    for k,v in meta.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest));return dest


def warp_clock(source,dest,start=1.2,extra_gain=.2):
    """Advance all neural phase-dependent branches together, after launch.

    Only the actor's internal time feature changes. Physics/control dt and the
    full five-second evaluation remain unchanged, and no body pose is imposed.
    """
    original=onnx.load(source);meta={p.key:p.value for p in original.metadata_props}
    if float(meta.get('sim2sim_time_input_s',0))!=5 or not 0<=start<5 or not 0<=extra_gain<=1:
        raise ValueError('Clock warp needs a declared five-second timed actor')
    parent=compose.add_prefix(original,'clock_parent/')
    mask=np.zeros((1,61),np.float32);mask[:,48]=1.
    arrays=dict(clock_start=np.array([48],np.int64),clock_end=np.array([49],np.int64),
        clock_axis=np.array([1],np.int64),clock_threshold=np.array(start/5,np.float32),
        clock_gain=np.array(extra_gain,np.float32),clock_one=np.array(1.,np.float32),clock_mask=mask)
    nodes=[helper.make_node('Slice',['obs','clock_start','clock_end','clock_axis'],['clock_fraction']),
        helper.make_node('Sub',['clock_fraction','clock_threshold'],['clock_excess']),
        helper.make_node('Relu',['clock_excess'],['clock_positive_excess']),
        helper.make_node('Mul',['clock_positive_excess','clock_gain'],['clock_delta']),
        helper.make_node('Add',['clock_fraction','clock_delta'],['clock_unbounded']),
        helper.make_node('Min',['clock_unbounded','clock_one'],['clock_bounded']),
        helper.make_node('Sub',['clock_bounded','clock_fraction'],['clock_offset']),
        helper.make_node('Mul',['clock_offset','clock_mask'],['clock_feature_offset']),
        helper.make_node('Add',['obs','clock_feature_offset'],[parent.graph.input[0].name])]
    nodes+=list(parent.graph.node)+[helper.make_node('Identity',[parent.graph.output[0].name],['actions'])]
    graph=helper.make_graph(nodes,'explicit_neural_clock_warp',
        [helper.make_tensor_value_info('obs',onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info('actions',onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(parent.graph.initializer)+[numpy_helper.from_array(v,k) for k,v in arrays.items()])
    model=helper.make_model(graph,opset_imports=list(original.opset_import));model.ir_version=original.ir_version
    meta.update(sim2sim_clock_warp=json.dumps(dict(start_s=start,extra_gain=extra_gain)),
                sim2sim_clock_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    for k,v in meta.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest));return dest


def clock_parity(source,exported,start,extra_gain,n=10000):
    from .models import NativeAnchor
    x=np.random.default_rng(491).normal(0,.5,(n,61)).astype(np.float32)
    expected=x.copy();fraction=x[:,48:49]
    mapped=np.minimum(fraction+np.maximum(fraction-np.float32(start/5),0)*np.float32(extra_gain),np.float32(1))
    expected[:,48:49]=fraction+(mapped-fraction)
    error=float(np.max(np.abs(NativeAnchor(source)(expected)-NativeAnchor(exported)(x))))
    return dict(samples=n,max_abs=error,threshold=1e-5,passed=error<1e-5)


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('dest',type=Path)
    p.add_argument('--start',type=float,default=1.9);p.add_argument('--end',type=float,default=2.1)
    a=p.parse_args();print(build(a.source,a.dest,a.start,a.end))


if __name__=='__main__':main()
