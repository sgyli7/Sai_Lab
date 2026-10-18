"""Owned GPU proof that the selected trained delta reloads and produces actions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from safetensors import safe_open

from experiments.pi05_g1.training import EpisodeDataset, memory_snapshot
from .data import OWN, SHARED, SPLIT, MeanStd, sha
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier trained-model restore proof')
    start = time.monotonic()
    report = {'schema': 'lingbot_vla2_g1_restore_probe_v1', 'status': 'running', 'test_opened': False, 'formal_outcomes_read': False}
    try:
        torch.set_num_threads(4)
        training = json.loads(args.training_result.read_text())
        if training.get('status') != 'completed' or training.get('optimizer_steps', 0) <= 0 or not training.get('selected_delta'):
            raise ValueError('A validation-selected real trained delta is required')
        delta = Path(training['selected_delta'])
        norm = Path(training['norm_path'])
        metadata = json.loads((delta / 'metadata.json').read_text())
        if sha(delta / 'trainable.safetensors') != metadata['weights_sha256'] or sha(norm) != training['norm_sha256']:
            raise ValueError('Selected model/normalization payload changed')
        dataset = EpisodeDataset(SHARED / 'data/lerobot', SPLIT, 'train', MeanStd(norm))
        rgb, state, _, _ = dataset.sample(0)
        runtime = Runtime(delta=delta, norm_stats=norm)
        parameters = dict(runtime.model.named_parameters())
        names = ['model.action_out_proj.weight', 'model.state_proj.weight', 'model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj.weight']
        restored = {}
        with safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as saved:
            for name in names:
                actual = parameters[name].detach().flatten()[:256].cpu()
                expected = saved.get_tensor(name).flatten()[:256]
                restored[name] = bool(torch.equal(actual, expected))
        if not all(restored.values()):
            raise ValueError('Actual CUDA model differs from saved selected delta')
        frozen_name = 'model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_proj.weight'
        index = json.loads((OWN / 'base_bf16/model.safetensors.index.json').read_text())['weight_map']
        with safe_open(str(OWN / 'base_bf16' / index[frozen_name]), framework='pt', device='cpu') as base:
            frozen_equal = bool(torch.equal(parameters[frozen_name].detach().flatten()[:256].cpu(), base.get_tensor(frozen_name).flatten()[:256]))
        if not frozen_equal:
            raise ValueError('Frozen VLM differs from the pretrained base')
        predictions, timings = [], []
        for repeat in range(2):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            tick = time.monotonic()
            actions = runtime.infer(rgb, state)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - tick)
            if actions.shape != (50, 32) or not np.isfinite(actions).all():
                raise ValueError('Reloaded policy did not produce finite50x32 actions')
            predictions.append(actions)
        report.update(status='completed', selected_delta=str(delta), selected_step=training['selected_step'], training_result_sha256=sha(args.training_result), weights_sha256=metadata['weights_sha256'], norm_sha256=sha(norm), actual_cuda_samples_equal_saved=restored, frozen_VLM_sample_equal_base=frozen_equal, action_shape=[50, 32], inference_seconds=timings, same_seed_max_abs_difference=float(np.max(np.abs(predictions[0] - predictions[1]))), model_load_seconds=runtime.load_seconds, **memory_snapshot(torch))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
