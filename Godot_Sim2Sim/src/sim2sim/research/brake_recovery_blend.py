"""Speed-conditioned blend of a braking anchor and an upright residual actor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import compose,helper,numpy_helper

from sim2sim.policy_state import BRAKE_STATE_V1,STATE_KEY
from .budget import require_supervision
from .brake_pose_search import rollout
from .models import NativeAnchor
from .queue import atomic_json


def export(base,recovery,target,threshold,strength):
    if threshold<=0 or not 0<strength<=1:raise ValueError('Invalid blend parameters')
    old,up=NativeAnchor(base),NativeAnchor(recovery)
    if old.state_input!=BRAKE_STATE_V1 or up.state_input!=BRAKE_STATE_V1:
        raise ValueError('Both actors must declare the same velocity/height input contract')
    originals=[onnx.load(base),onnx.load(recovery)]
    a,b=[compose.add_prefix(m,prefix) for m,prefix in zip(originals,['braking/','recovery/'])]
    for model in (a,b):
        prior=model.graph.input[0].name
        for node in model.graph.node:
            for i,name in enumerate(node.input):
                if name==prior:node.input[i]='obs'
    const={'indices':np.array([58,59],np.int64),'axis':np.array([1],np.int64),
        'throttle_index':np.array([48],np.int64),'threshold':np.array(threshold,np.float32),
        'strength':np.array(strength,np.float32),'zero':np.array(0.,np.float32),
        'one':np.array(1.,np.float32),'throttle_scale':np.array(-20.,np.float32)}
    def n(name):return 'phase_blend/'+name
    nodes=list(a.graph.node)+list(b.graph.node)+[
        helper.make_node('Gather',['obs',n('indices')],[n('velocity')],axis=1),
        helper.make_node('Mul',[n('velocity'),n('velocity')],[n('velocity_squared')]),
        helper.make_node('ReduceSum',[n('velocity_squared'),n('axis')],[n('speed_squared')],keepdims=1),
        helper.make_node('Sqrt',[n('speed_squared')],[n('speed')]),
        helper.make_node('Div',[n('speed'),n('threshold')],[n('fraction')]),
        helper.make_node('Sub',[n('one'),n('fraction')],[n('recovery_weight')]),
        helper.make_node('Clip',[n('recovery_weight'),n('zero'),n('one')],[n('bounded_weight')]),
        helper.make_node('Mul',[n('bounded_weight'),n('strength')],[n('blend_weight')]),
        helper.make_node('Gather',['obs',n('throttle_index')],[n('throttle')],axis=1),
        helper.make_node('Mul',[n('throttle'),n('throttle_scale')],[n('negative_weight')]),
        helper.make_node('Clip',[n('negative_weight'),n('zero'),n('one')],[n('brake_gate')]),
        helper.make_node('Mul',[n('brake_gate'),n('blend_weight')],[n('gate')]),
        helper.make_node('Sub',[b.graph.output[0].name,a.graph.output[0].name],[n('difference')]),
        helper.make_node('Mul',[n('gate'),n('difference')],[n('delta')]),
        helper.make_node('Add',[a.graph.output[0].name,n('delta')],['actions'])]
    graph=helper.make_graph(nodes,'brake_then_recover',
        [helper.make_tensor_value_info('obs',onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info('actions',onnx.TensorProto.FLOAT,[1,14])],
        initializer=[*a.graph.initializer,*b.graph.initializer,*[numpy_helper.from_array(v,n(k)) for k,v in const.items()]])
    result=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=max(a.ir_version,b.ir_version))
    meta={p.key:p.value for p in originals[0].metadata_props}
    meta.update({STATE_KEY:BRAKE_STATE_V1,'sim2sim_brake_recovery_blend':json.dumps(dict(
        base_sha256=old.sha256,recovery_sha256=up.sha256,speed_threshold=threshold,strength=strength))})
    helper.set_model_props(result,meta);onnx.checker.check_model(result)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);onnx.save(result,target)
    rng=np.random.default_rng(915002);x=rng.normal(0,.3,(512,61)).astype(np.float32);x[:64,48]=0.
    speed=np.sqrt(np.sum(x[:,58:60]**2,axis=1,keepdims=True))
    gate=np.clip(-20*x[:,48:49],0,1)*(np.clip(1-speed/np.float32(threshold),0,1)*np.float32(strength))
    anchor=old(x);expected=anchor+gate*(up(x)-anchor);actual=NativeAnchor(target)(x)
    error=float(np.max(abs(expected-actual)));preserved=x[:,48]>=0
    assert error<1e-5 and np.array_equal(actual[preserved],anchor[preserved])
    return dict(max_abs=error,positive_and_neutral_exact=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def run(experiment,output,workers=2):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    spec=json.loads(Path(experiment).read_text());output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'experiment.json',spec);results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for candidate in spec['candidates']:
            model=output/(candidate['name']+'.onnx')
            parity=export(spec['base'],spec['recovery'],model,candidate['threshold'],candidate['strength'])
            rows=[];jobs=[(case,seed) for case in spec['cases'] for seed in spec['seeds']]
            for start in range(0,len(jobs),workers):
                futures=[pool.submit(rollout,None,NativeAnchor(model),np.zeros(5),seed,output,
                    candidate['name']+'_'+case,case) for case,seed in jobs[start:start+workers]]
                rows.extend(f.result() for f in futures)
            record=dict(**candidate,parity=parity,model=str(model),trials=rows,
                passes=sum(r['metrics']['success'] for r in rows),falls=sum(r['metrics']['fell'] for r in rows),
                objective=float(np.mean([r['objective'] for r in rows])))
            results.append(record);atomic_json(output/(candidate['name']+'.json'),record)
            print(json.dumps({k:record[k] for k in ['name','passes','falls','objective']}),flush=True)
    result=dict(completed=True,candidates=results,best=min(results,key=lambda r:r['objective']),decision='unpromoted')
    atomic_json(output/'completed.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(a.experiment,a.out,a.workers)
