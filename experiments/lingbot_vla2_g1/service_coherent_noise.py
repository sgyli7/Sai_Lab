"""Same RGB/q/text-only LingBot protocol with declared coherent flow sampling."""
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
from .coherent_sampling import ADAPTER, SEED, infer, noise_identity
from .data import OWN, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, required=True)
    parser.add_argument('--decoded-evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.decoded_evidence.exists():
        raise FileExistsError('Fresh decoder evidence required')
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = PointMassRuntime(delta=args.delta, norm_stats=args.norm_stats)
    load = time.monotonic() - started
    args.decoded_evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.decoded_evidence.open('x', buffering=1) as evidence:
        loaded = {'event': 'model_loaded', 'load_seconds': load,
            'parameter_elements': runtime.parameter_elements, 'trainable_elements': runtime.trainable_elements,
            'delta': str(args.delta), 'norm_path': str(runtime.norm_path), 'norm_sha256': sha(runtime.norm_path),
            'contract_sha256': sha(OWN / 'contract.json'), 'base_conversion_receipt_sha256': sha(OWN / 'bf16_conversion_receipt.json'),
            'runtime_overlay_receipt_sha256': sha(OWN / 'runtime_overlay_receipt.json'),
            'inference_adapter': 'original_train_exact_constant_point_mass_inverse_v1',
            'constant_inverse_sha256': sha(Path(__file__).with_name('constant_inverse.py')),
            'constant_action_indices': np.flatnonzero(runtime.normalizer.constant).tolist(),
            'sampling_adapter': ADAPTER, 'common_noise_seed': SEED, 'predictions_per_query': 2,
            'common_noise_float32_sha256': noise_identity(),
            'coherent_sampling_sha256': sha(Path(__file__).with_name('coherent_sampling.py')),
            'coherent_service_sha256': sha(__file__)}
        evidence.write(json.dumps(loaded, allow_nan=False) + '\n')
        for line in sys.stdin:
            request = {}
            try:
                if len(line) > 4_000_000:
                    raise ValueError('Request exceeds bound')
                request = json.loads(line)
                rgb, state = observation(request)
                tick = time.monotonic()
                actions, predictions = infer(runtime, rgb, state)
                torch.cuda.synchronize()
                reply = {'episode_id': request['episode_id'], 'control_tick': request['control_tick'],
                         'actions': actions.tolist(), 'inference_seconds': time.monotonic() - tick,
                         'load_seconds': load, 'action_horizon': 50}
                evidence.write(json.dumps({'event': 'lingbot_decoded', 'actions32': actions.tolist(),
                    'model_predictions32': predictions.tolist(), 'sampling_adapter': ADAPTER,
                    'common_noise_float32_sha256': loaded['common_noise_float32_sha256'], **reply}, allow_nan=False) + '\n')
            except Exception as error:
                reply = {'episode_id': request.get('episode_id') if isinstance(request, dict) else None,
                         'control_tick': request.get('control_tick') if isinstance(request, dict) else None,
                         'error': repr(error)}
                evidence.write(json.dumps({'event': 'request_rejected', **reply}) + '\n')
            print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
