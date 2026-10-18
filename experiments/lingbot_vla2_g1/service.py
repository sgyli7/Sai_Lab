"""Exact shared mobile JSONL policy protocol; no truth/route inputs admitted."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pi05_g1_mobile.pi05_service import observation
from .data import OWN,sha
from .runtime import Runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--delta',type=Path);p.add_argument('--norm-stats',type=Path,default=OWN/'train_meanstd.json');p.add_argument('--decoded-evidence',type=Path,required=True);args=p.parse_args()
    if args.decoded_evidence.exists():raise FileExistsError(args.decoded_evidence)
    start=time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):runtime=Runtime(delta=args.delta,norm_stats=args.norm_stats)
    load=time.monotonic()-start;args.decoded_evidence.parent.mkdir(parents=True,exist_ok=True)
    with args.decoded_evidence.open('x',buffering=1) as evidence:
        evidence.write(json.dumps({'event':'model_loaded','load_seconds':load,'parameter_elements':runtime.parameter_elements,'trainable_elements':runtime.trainable_elements,'delta':str(args.delta) if args.delta else None,'norm_path':str(runtime.norm_path),'norm_sha256':sha(runtime.norm_path),'contract_sha256':sha(OWN/'contract.json'),'base_conversion_receipt_sha256':sha(OWN/'bf16_conversion_receipt.json'),'runtime_overlay_receipt_sha256':sha(OWN/'runtime_overlay_receipt.json')})+'\n')
        for line in sys.stdin:
            req={}
            try:
                if len(line)>4_000_000:raise ValueError('Request exceeds bound')
                req=json.loads(line);rgb,state=observation(req)
                torch.manual_seed(req['seed']);torch.cuda.manual_seed_all(req['seed'])
                tick=time.monotonic()
                with contextlib.redirect_stdout(sys.stderr):actions=runtime.infer(rgb,state)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                if actions.shape!=(50,32) or not np.isfinite(actions).all():raise ValueError('Invalid LingBot effective32 action chunk')
                reply={'episode_id':req['episode_id'],'control_tick':req['control_tick'],'actions':actions.tolist(),'inference_seconds':elapsed,'load_seconds':load,'action_horizon':50}
                evidence.write(json.dumps({'event':'lingbot_decoded','actions32':actions.tolist(),**reply},allow_nan=False)+'\n')
            except Exception as error:
                reply={'episode_id':req.get('episode_id') if isinstance(req,dict) else None,'control_tick':req.get('control_tick') if isinstance(req,dict) else None,'error':repr(error)}
                evidence.write(json.dumps({'event':'request_rejected',**reply})+'\n')
            print(json.dumps(reply,allow_nan=False),flush=True)


if __name__=='__main__':main()
