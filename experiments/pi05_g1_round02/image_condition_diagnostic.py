"""Train-only pixel audit and optional root-budgeted CUDA image conditioning test.

Default is CPU preparation. Pass --run-gpu only inside root's new round02 budget.
Replay RGB is decoded at the original 50Hz MP4 frame index, without interpolation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pi05_g1.contract import ACTION_NAMES, PROMPT, encode_row
from experiments.pi05_g1.model_runtime import resize_pad_rgb


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def video_frame_indices(path, ticks):
    import av
    requested=set(ticks);output={}
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if abs(float(stream.average_rate)-50)>1e-3:raise ValueError('50Hz source video required')
        for index,frame in enumerate(container.decode(stream)):
            if index in requested:
                timestamp=float(frame.pts*stream.time_base)
                if abs(timestamp-index/50)>.01001:raise ValueError(f'Video PTS not aligned to tick {index}')
                output[index]=frame.to_ndarray(format='rgb24')
            if len(output)==len(requested):break
    if set(output)!=requested:raise ValueError(f'Missing requested video ticks: {requested-set(output)}')
    return output


def pixel_summary(rgb):
    import torch
    from openpi.shared.image_tools import resize_with_pad_torch
    legacy=resize_pad_rgb(rgb)
    source=torch.from_numpy(rgb.astype(np.float32)/127.5-1.)
    official=resize_with_pad_torch(source,224,224).numpy()
    if official.ndim==4:official=official[0]
    difference=legacy-official
    return {'raw_shape':list(rgb.shape),'raw_dtype':str(rgb.dtype),'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'raw_channel_mean':rgb.mean(axis=(0,1)).tolist(),'model_shape_nchw':[1,3,224,224],'legacy_normalized_range':[float(legacy.min()),float(legacy.max())],'official_torch_resize_difference':{'mean_abs':float(np.abs(difference).mean()),'max_abs':float(np.abs(difference).max()),'rms':float(np.sqrt(np.square(difference).mean()))},'mask':{'base_0_rgb':True,'left_wrist_0_rgb':False,'right_wrist_0_rgb':False}}


def pair_actions(left,right):
    physical=left['actions']-right['actions'];normalized=left['normalized_actions']-right['normalized_actions']
    return {'physical_mean_abs_per_dim':np.abs(physical).mean(axis=0).tolist(),'physical_max_abs_per_dim':np.abs(physical).max(axis=0).tolist(),'normalized_mean_abs_per_dim':np.abs(normalized).mean(axis=0).tolist(),'normalized_max_abs_per_dim':np.abs(normalized).max(axis=0).tolist(),'normalized_rms_per_dim':np.sqrt(np.square(normalized).mean(axis=0)).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','norm-stats','dataset-root','manifest','replay-dir','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--ticks',nargs='+',type=int,default=[0,50,76,100])
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-gpu',action='store_true')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    split=json.loads(Path(args.manifest).read_text())
    if args.episode not in split['train']:raise ValueError('Image diagnostic may use training episodes only')
    if len(set(args.ticks))!=len(args.ticks) or min(args.ticks)<0:raise ValueError('Unique nonnegative ticks required')
    root=Path(args.dataset_root);info=json.loads((root/'meta/info.json').read_text())
    keys=[name for name,feature in info['features'].items() if feature.get('dtype')=='video' and ('ego' in name.lower() or 'head' in name.lower())]
    if len(keys)!=1:raise ValueError('Exactly one head RGB video feature required')
    fmt={'episode_index':args.episode,'episode_chunk':args.episode//info.get('chunks_size',1000),'video_key':keys[0]}
    import pyarrow.parquet as pq
    rows=pq.read_table(root/info['data_path'].format(**fmt)).to_pylist()
    if max(args.ticks)>=len(rows):raise ValueError('Requested tick outside training episode')
    dataset_video=root/info['video_path'].format(**fmt)
    replay_video=Path(args.replay_dir)/'head_50fps.mp4'
    dataset_rgb=video_frame_indices(dataset_video,args.ticks);replay_rgb=video_frame_indices(replay_video,args.ticks)
    report={'schema':'pi05_g1_round02_image_condition_diagnostic_v1','status':'cpu_prepared','training_episode':args.episode,'ticks':args.ticks,'same_state_prompt_noise_within_tick':True,'seed':args.seed,'prompt':PROMPT,'action_names':list(ACTION_NAMES),'dataset_video_sha256':digest(dataset_video),'replay_video_sha256':digest(replay_video),'replay_frame_source':'original MP4 decoded frame index; lossy encoded observation, not raw render','checkpoint':str(Path(args.checkpoint).resolve()),'norm_sha256':digest(args.norm_stats),'falsifiable_findings':{'range_layout_mask':'Current shared path is uint8 RGB -> PIL resizepad -> float[-1,1] -> NCHW; official eval preprocess neither rescales nor resizes an already224x224 input. No static evidence of train/infer double normalization or BGR conversion.','resize':'PIL antialiased bilinear differs from official torch bilinear without antialias; both current training and inference use the same PIL path. For480x640 both compute224x168 with28-row black borders. Other aspect ratios can differ by round-vs-floor.','inference_limit':'Image-conditioned action differences establish model sensitivity, not that a visual mismatch caused physical failure; q, prompt and noise held fixed only within each tick. Black-image sensitivity is a diagnostic, not task success.'},'samples':[]}
    from PIL import Image
    for tick in args.ticks:
        if abs(float(rows[tick]['timestamp'])-tick/50)>.01001:raise ValueError('Training row timestamp not aligned')
        state=encode_row(rows[tick])['state'];black=np.zeros_like(dataset_rgb[tick])
        images={'dataset':dataset_rgb[tick],'replay':replay_rgb[tick],'black':black}
        np.savez(output/f'input_tick{tick:04d}.npz',state=state,**images)
        sample={'tick':tick,'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'images':{name:pixel_summary(rgb) for name,rgb in images.items()}}
        for name,rgb in images.items():Image.fromarray(rgb).save(output/f'{name}_tick{tick:04d}.png')
        report['samples'].append(sample)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if args.run_gpu:
        from experiments.pi05_g1.model_runtime import ModelRuntime
        runtime=ModelRuntime(args.checkpoint,args.norm_stats,head_only=False)
        report['checkpoint_weights_sha256']=digest(Path(args.checkpoint)/'model.safetensors')
        span=np.asarray(runtime.norm.stats['actions']['q99'])-np.asarray(runtime.norm.stats['actions']['q01'])
        report['collapsed_normalization_dimensions']=np.flatnonzero(span<=1e-6).tolist()
        for sample in report['samples']:
            tick=sample['tick']
            with np.load(output/f'input_tick{tick:04d}.npz',allow_pickle=False) as source:
                predictions={name:runtime.infer(source[name],source['state'],PROMPT,seed=args.seed,num_steps=10) for name in ('dataset','replay','black')}
            np.savez(output/f'actions_tick{tick:04d}.npz',**{f'{name}_{key}':value for name,result in predictions.items() for key,value in result.items()})
            sample['actions']={name:{'inference_seconds':value['inference_seconds'],'first_action':value['actions'][0].tolist()} for name,value in predictions.items()}
            sample['paired_differences']={name:pair_actions(predictions[left],predictions[right]) for name,left,right in [('replay_minus_dataset','replay','dataset'),('black_minus_dataset','black','dataset'),('black_minus_replay','black','replay')]}
        report['status']='cuda_diagnostic_complete'
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':report['status'],'output':str(output),'ticks':args.ticks}),flush=True)


if __name__=='__main__':main()
