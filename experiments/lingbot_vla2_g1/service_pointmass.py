"""Same strict RGB/joint policy protocol with the prospective constant inverse."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from experiments.pi05_g1_mobile.pi05_service import observation
from .constant_inverse import PointMassRuntime
from .data import OWN, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, required=True)
    parser.add_argument('--decoded-evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.decoded_evidence.exists():
        raise FileExistsError(args.decoded_evidence)
    start = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = PointMassRuntime(delta=args.delta, norm_stats=args.norm_stats)
    load = time.monotonic() - start
    args.decoded_evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.decoded_evidence.open('x', buffering=1) as evidence:
        evidence.write(json.dumps({'event': 'model_loaded', 'load_seconds': load,
            'parameter_elements': runtime.parameter_elements, 'trainable_elements': runtime.trainable_elements,
            'delta': str(args.delta), 'norm_path': str(runtime.norm_path), 'norm_sha256': sha(runtime.norm_path),
            'contract_sha256': sha(OWN / 'contract.json'), 'base_conversion_receipt_sha256': sha(OWN / 'bf16_conversion_receipt.json'),
            'runtime_overlay_receipt_sha256': sha(OWN / 'runtime_overlay_receipt.json'),
            'inference_adapter': 'original_train_exact_constant_point_mass_inverse_v1',
            'constant_inverse_sha256': sha(Path(__file__).with_name('constant_inverse.py')),
            'constant_action_indices': np.flatnonzero(runtime.normalizer.constant).tolist()}) + '\n')
        for line in sys.stdin:
            req = {}
            try:
                if len(line) > 4_000_000:
                    raise ValueError('Request exceeds bound')
                req = json.loads(line)
                rgb, state = observation(req)
                torch.manual_seed(req['seed']); torch.cuda.manual_seed_all(req['seed'])
                tick = time.monotonic()
                with contextlib.redirect_stdout(sys.stderr):
                    actions = runtime.infer(rgb, state)
                torch.cuda.synchronize()
                if actions.shape != (50, 32) or not np.isfinite(actions).all():
                    raise ValueError('Invalid real LingBot action chunk')
                reply = {'episode_id': req['episode_id'], 'control_tick': req['control_tick'],
                         'actions': actions.tolist(), 'inference_seconds': time.monotonic() - tick,
                         'load_seconds': load, 'action_horizon': 50}
                evidence.write(json.dumps({'event': 'lingbot_decoded', 'actions32': actions.tolist(), **reply}, allow_nan=False) + '\n')
            except Exception as error:
                reply = {'episode_id': req.get('episode_id') if isinstance(req, dict) else None,
                         'control_tick': req.get('control_tick') if isinstance(req, dict) else None, 'error': repr(error)}
                evidence.write(json.dumps({'event': 'request_rejected', **reply}) + '\n')
            print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
