"""Brake-only filtered leg posture, exported beside the untouched policy graph.

This is an explicit control experiment. It changes joint targets, never rigid
body state, motor parameters, collision, gravity, or the positive/coast policy.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .brake_action_limit import action_bounds
from .brake_pose_search import action_offset
from .models import NativeAnchor


def prediction(anchor, obs, target, alpha, feedback, clip_targets=True, feedback_joint=4):
    obs=np.asarray(obs,np.float32).reshape(-1,61);base=anchor(obs)
    if clip_targets:
        low,high=action_bounds();previous=np.clip(obs[:,34:48],low,high)
    else:previous=obs[:,34:48]
    desired=np.broadcast_to(np.asarray(target,np.float32),(len(obs),14)).copy()
    correction=np.clip(-np.float32(feedback[0])*obs[:,3]-np.float32(feedback[1])*obs[:,1],-.2,.2)
    desired[:,feedback_joint]+=correction;desired[:,feedback_joint+9]-=correction
    filtered=previous+np.float32(alpha)*(desired-previous)
    if clip_targets:filtered=np.clip(filtered,low,high)
    mask=np.ones(14,np.float32);mask[5:9]=0
    gate=np.clip(-20*obs[:,48:49],0,1)
    return base+gate*(mask*(filtered-base))


def export(source,target,parameters,alpha=.3,feedback=(-.3,.05),clip_targets=True,feedback_joint=4):
    if not 0<alpha<=1:raise ValueError('Filter alpha must be in (0,1]')
    if feedback_joint not in (2,3,4):raise ValueError('Feedback requires a sagittal leg joint')
    pose=action_offset(parameters);low,high=action_bounds()
    model=onnx.load(source);original=model.graph.output[0].name
    prefix='brake_pose_controller/'
    if any(n.startswith(prefix) for node in model.graph.node for n in [*node.input,*node.output]):
        raise ValueError('Already wrapped with a pose controller')
    def name(s):return prefix+s
    leg_mask=np.ones((1,14),np.float32);leg_mask[:,5:9]=0
    constants={'previous_indices':np.arange(34,48,dtype=np.int64),'throttle_index':np.array([48],np.int64),
        'pitch_index':np.array([3],np.int64),'gyro_index':np.array([1],np.int64),
        'low':low[None],'high':high[None],'pose':pose[None], 'leg_mask':leg_mask,
        'ankles':np.eye(14,dtype=np.float32)[feedback_joint:feedback_joint+1]-np.eye(14,dtype=np.float32)[feedback_joint+9:feedback_joint+10],
        'alpha':np.array(alpha,np.float32),'kp':np.array(-feedback[0],np.float32),
        'kd':np.array(-feedback[1],np.float32),'feedback_low':np.array(-.2,np.float32),
        'feedback_high':np.array(.2,np.float32),'scale':np.array(-20.,np.float32),
        'zero':np.array(0.,np.float32),'one':np.array(1.,np.float32)}
    if not clip_targets:
        del constants['low'],constants['high']
    model.graph.initializer.extend(numpy_helper.from_array(v,name(k)) for k,v in constants.items())
    nodes=[]
    def op(kind,inputs,output,**attrs):
        nodes.append(helper.make_node(kind,[name(x) if x in constants or x in produced else x for x in inputs],[name(output)],**attrs));produced.add(output)
    produced=set();external=model.graph.input[0].name
    op('Gather',[external,'previous_indices'],'previous',axis=1)
    if clip_targets:
        op('Max',['previous','low'],'previous_low');op('Min',['previous_low','high'],'previous_bounded')
    else:op('Identity',['previous'],'previous_bounded')
    op('Gather',[external,'pitch_index'],'pitch',axis=1);op('Gather',[external,'gyro_index'],'gyro',axis=1)
    op('Mul',['pitch','kp'],'p');op('Mul',['gyro','kd'],'d');op('Add',['p','d'],'pd')
    op('Clip',['pd','feedback_low','feedback_high'],'pd_bounded')
    op('Mul',['pd_bounded','ankles'],'ankle_delta');op('Add',['pose','ankle_delta'],'desired')
    op('Sub',['desired','previous_bounded'],'difference');op('Mul',['difference','alpha'],'step')
    op('Add',['previous_bounded','step'],'filtered')
    if clip_targets:
        op('Max',['filtered','low'],'filtered_low');op('Min',['filtered_low','high'],'bounded')
    else:op('Identity',['filtered'],'bounded')
    op('Sub',['bounded',original],'change')
    op('Mul',['change','leg_mask'],'leg_change')
    op('Gather',[external,'throttle_index'],'throttle',axis=1);op('Mul',['throttle','scale'],'weight')
    op('Clip',['weight','zero','one'],'gate');op('Mul',['gate','leg_change'],'delta')
    op('Add',[original,'delta'],'actions')
    model.graph.node.extend(nodes);model.graph.output[0].name=name('actions')
    meta={p.key:p.value for p in model.metadata_props}
    meta['sim2sim_brake_pose_controller']=json.dumps(dict(version=1,parameters=list(parameters),alpha=alpha,
        feedback=list(feedback),clip_targets=clip_targets,feedback_joint=feedback_joint,
        parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()))
    helper.set_model_props(model,meta);onnx.checker.check_model(model)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,target)
    old,new=NativeAnchor(source),NativeAnchor(target)
    obs=np.random.default_rng(915005).normal(size=(256,61)).astype(np.float32);obs[:64,48]=0.
    actual=new(obs);expected=prediction(old,obs,pose,alpha,feedback,clip_targets,feedback_joint)
    error=float(np.max(abs(actual-expected)));preserved=obs[:,48]>=0
    assert error<1e-5 and np.array_equal(actual[preserved],old(obs[preserved]))
    return dict(max_abs=error,positive_and_neutral_exact=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def run_grid(experiment,output,workers=2):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from .budget import require_supervision
    from .brake_pose_search import rollout
    from .queue import atomic_json
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    spec=json.loads(Path(experiment).read_text());output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);rows=[]
    atomic_json(output/'experiment.json',spec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for candidate in spec['candidates']:
            model=output/(candidate['name']+'.onnx')
            parity=export(spec['source'],model,candidate['parameters'],candidate['alpha'],
                candidate.get('feedback',[-.3,.05]),candidate.get('clip_targets',True),candidate.get('feedback_joint',4))
            jobs=[(case,seed) for case in spec['cases'] for seed in spec['seeds']];trials=[]
            for start in range(0,len(jobs),workers):
                futures=[pool.submit(rollout,None,NativeAnchor(model),np.zeros(5),seed,output,
                    candidate['name']+'_'+case,case) for case,seed in jobs[start:start+workers]]
                trials.extend(f.result() for f in futures)
            result=dict(**candidate,model=str(model),parity=parity,trials=trials,
                passes=sum(t['metrics']['success'] for t in trials),falls=sum(t['metrics']['fell'] for t in trials),
                objective=float(np.mean([t['objective'] for t in trials])))
            rows.append(result);atomic_json(output/(candidate['name']+'.json'),result)
            print(json.dumps({k:result[k] for k in ['name','passes','falls','objective']}),flush=True)
    result=dict(completed=True,candidates=rows,best=min(rows,key=lambda r:r['objective']),decision='unpromoted')
    atomic_json(output/'completed.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment');p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();run_grid(a.experiment,a.out,a.workers)
