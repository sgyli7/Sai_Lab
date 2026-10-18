"""Mobile pi05 JSONL service: exact RGB/measured-state input and precision receipt."""
from __future__ import annotations
import argparse
import base64
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from .contract import PROMPT,encode_state

REQUEST_KEYS={'schema','episode_id','control_tick','rgb_shape','rgb_uint8_b64','state','prompt','seed','measured_joint_positions43','measured_joint_names43'}


def observation(req):
    if not isinstance(req,dict) or set(req)!=REQUEST_KEYS or req['schema']!='pi05_g1_observation_v1':
        raise ValueError('Exact RGB/measured-joint request required; extra truth keys rejected')
    if req['prompt']!=PROMPT or req['rgb_shape']!=[480,640,3]:
        raise ValueError('Pinned mobile prompt and original640x480 head RGB required')
    if type(req['control_tick']) is not int or req['control_tick']<0 or type(req['seed']) is not int or not 0<=req['seed']<2**64:
        raise ValueError('Nonnegative integer control tick and uint64 seed required')
    if not isinstance(req['measured_joint_names43'],list) or not isinstance(req['measured_joint_positions43'],list):
        raise ValueError('Measured named43 feedback arrays required')
    state=np.asarray(req['state'],dtype=np.float32)
    measured=encode_state(req['measured_joint_positions43'],req['measured_joint_names43'])
    if state.shape!=(32,) or not np.isfinite(state).all() or not np.array_equal(state,measured):
        raise ValueError('state32 must exactly encode measured named43 feedback; no independent truth state')
    raw=base64.b64decode(req['rgb_uint8_b64'],validate=True)
    if len(raw)!=480*640*3:raise ValueError('RGB byte count differs')
    return np.frombuffer(raw,np.uint8).reshape(480,640,3),state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--norm-stats',required=True)
    p.add_argument('--head-only',action='store_true');p.add_argument('--decoded-evidence',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    # Refuse an existing evidence path before loading a large model.
    if args.decoded_evidence.exists():raise FileExistsError(args.decoded_evidence)
    with contextlib.redirect_stdout(sys.stderr):
        from .model_runtime import ModelRuntime
        runtime=ModelRuntime(args.checkpoint,args.norm_stats,head_only=args.head_only)
    load=time.monotonic()-start
    args.decoded_evidence.parent.mkdir(parents=True,exist_ok=True)
    with args.decoded_evidence.open('x',buffering=1) as evidence:
        evidence.write(json.dumps({'event':'model_loaded','load_seconds':load,'identity':runtime.metadata},allow_nan=False)+'\n')
        for line in sys.stdin:
            req={}
            try:
                if len(line)>4_000_000:raise ValueError('Request exceeds size bound')
                req=json.loads(line);rgb,state=observation(req)
                with contextlib.redirect_stdout(sys.stderr):result=runtime.infer(rgb,state,req['prompt'],req['seed'])
                actions=np.asarray(result['actions'],np.float32)
                if actions.shape!=(50,32) or not np.isfinite(actions).all():raise ValueError('Invalid mobile action chunk')
                elapsed=float(result['inference_seconds'])
                if not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid inference latency')
                evidence.write(json.dumps({'event':'pi05_decoded','episode_id':req['episode_id'],'control_tick':req['control_tick'],'actions32':actions.tolist(),'inference_seconds':elapsed},allow_nan=False)+'\n')
                reply={'episode_id':req['episode_id'],'control_tick':req['control_tick'],'actions':actions.tolist(),'inference_seconds':elapsed,'load_seconds':load,'action_horizon':50}
            except Exception as error:
                reply={'episode_id':req.get('episode_id') if isinstance(req,dict) else None,'control_tick':req.get('control_tick') if isinstance(req,dict) else None,'error':repr(error)}
                evidence.write(json.dumps({'event':'request_rejected',**reply},allow_nan=False)+'\n')
            print(json.dumps(reply,allow_nan=False),flush=True)


if __name__=='__main__':main()
