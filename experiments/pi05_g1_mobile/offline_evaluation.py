"""Auxiliary held-out action error, never checkpoint selection or physical success.

--plan-only reads manifest lengths only. Actual evaluation is root GPU supervised.
Raw targets use the existing episode encoder with an identity action transform,
so constant quantile dimensions cannot erase raw ground truth during evaluation.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
from .contract import ACTION_NAMES,PROMPT,SCHEMA as CONTRACT_SCHEMA

SCHEMA='pi05_g1_mobile_offline_error_result_v1'
GROUPS={'arms':list(range(14)),'hands':list(range(14,28)),'height':[28],
        'navigation_translation':[29,30],'navigation_yaw':[31]}
UNITS=['rad']*28+['m','m/s','m/s','rad/s']
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write_new(path,value):
    with Path(path).open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def plan(manifest):
    manifest=Path(manifest);data=json.loads(manifest.read_text());test=data['test']
    if len(test)!=10 or len(set(test))!=10:raise ValueError('Exactly10 unique held-out test episodes required')
    if set(test)&(set(data['train'])|set(data['validation'])):raise ValueError('Split overlap')
    files={int(x['episode_id']):x for x in data['files']};chunks=[];offset=0
    for episode in test:
        length=files[episode]['length']
        if not isinstance(length,int) or length<=0:raise ValueError('Invalid complete episode length')
        for frame in range(0,length,50):
            seed=int.from_bytes(hashlib.sha256(f'42:{episode}:{frame}'.encode()).digest()[:4],'little')%(2**31)
            chunks.append({'episode_id':episode,'frame_index':frame,'dataset_index':offset+frame,'valid_frames':min(50,length-frame),'seed':seed})
        offset+=length
    return {'schema':'pi05_g1_mobile_offline_chunk_plan_v1','manifest_sha256':sha(manifest),
            'test_episode_ids':test,'chunks':chunks,'chunk_count':len(chunks),'valid_frames':offset,
            'seed_rule':'sha256(42:episode_id:frame_index) first4 little-endian mod2^31',
            'checkpoint_selection':False,'test_observations_opened':False}


class RawTargets:
    def normalize(self,value,key):
        if key!='actions':raise ValueError('Only raw action targets are transformed')
        return np.asarray(value,np.float32)


def resolve_selected(training_result,manifest,norm,expected_validation_chunks):
    training_result=Path(training_result).resolve();result=json.loads(training_result.read_text())
    if result.get('schema')!='pi05_g1_mobile_training_result_v1' or result.get('status')!='completed' or not result.get('training_stopped'):
        raise ValueError('A genuine completed mobile training result with training_stopped is required')
    selected=Path(result['selected_checkpoint']).resolve()
    if selected.parent!=training_result.parent or not selected.is_dir():raise ValueError('Selected checkpoint must be direct child of completed training directory')
    metadata=json.loads((selected/'metadata.json').read_text());identity=result['identity']
    expected={'manifest_sha256':sha(manifest),'norm_sha256':sha(norm),'prompt':PROMPT,'contract_schema':CONTRACT_SCHEMA}
    if any(identity.get(k)!=v or metadata.get('identity',{}).get(k)!=v for k,v in expected.items()):raise ValueError('Selected training/data/norm/prompt identity differs')
    if metadata.get('schema')!='pi05_g1_mobile_training_checkpoint_v1' or not metadata.get('validation_complete') or not metadata.get('validation',{}).get('complete'):
        raise ValueError('Selected checkpoint lacks complete validation')
    validation=metadata['validation']
    if validation.get('chunks')!=expected_validation_chunks or validation.get('episodes')!=10:raise ValueError('Selected validation coverage differs')
    if metadata['validation_flow_loss']!=result['selected_validation_flow_loss']:raise ValueError('Selected validation loss mismatch')
    pointer=json.loads((training_result.parent/'best_checkpoint.json').read_text())
    if Path(pointer['path']).resolve()!=selected or pointer['metadata_sha256']!=sha(selected/'metadata.json'):raise ValueError('Completed selection pointer mismatch')
    # Selection is already frozen by validation. Only verify its receipts; never
    # inspect held-out losses to choose a different checkpoint.
    declared=metadata.get('weights_sha256',metadata.get('warmstart',{}).get('weights_sha256'))
    return selected,{'training_result_sha256':sha(training_result),'selected_metadata_sha256':sha(selected/'metadata.json'),'declared_weights_sha256':declared,'step':metadata['step'],'validation_flow_loss':metadata['validation_flow_loss'],'selection_used_test':False}


def bounds(path):
    value=json.loads(Path(path).read_text());lo=np.asarray(value['lower'],np.float64);hi=np.asarray(value['upper'],np.float64)
    if value.get('names')!=list(ACTION_NAMES) or lo.shape!=(32,) or hi.shape!=(32,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi):raise ValueError('Frozen common bounds names/shape/range invalid')
    return lo,hi


def metrics(absolute,square,outside,frames,any_outside):
    if frames<=0:raise ValueError('No valid test frames')
    mae=absolute/frames;rmse=np.sqrt(square/frames)
    return {'valid_frames':frames,'mae_per_channel':mae.tolist(),'rmse_per_channel':rmse.tolist(),
            'out_of_bounds_rate_per_channel':(outside/frames).tolist(),'any_channel_out_of_bounds_frame_rate':any_outside/frames,
            'groups':{name:{'channel_indices':indices,'unit':UNITS[indices[0]],'mae':float(mae[indices].mean()),'rmse':float(np.sqrt(square[indices].sum()/(frames*len(indices))))} for name,indices in GROUPS.items()},
            'navigation':{'mae_per_component':mae[29:32].tolist(),'rmse_per_component':rmse[29:32].tolist(),'units':UNITS[29:32]},
            'mixed_unit_aggregate_reported':False}


def compare(base,fine):
    def change(a,b):return None if a==0 else (b-a)/a
    return {'relative_change_sign':'negative is reduced action error; no physical ability claim',
            'mae_relative_change_per_channel':[change(a,b) for a,b in zip(base['mae_per_channel'],fine['mae_per_channel'])],
            'rmse_relative_change_per_channel':[change(a,b) for a,b in zip(base['rmse_per_channel'],fine['rmse_per_channel'])],
            'groups':{name:{key+'_relative_change':change(base['groups'][name][key],fine['groups'][name][key]) for key in ('mae','rmse')} for name in GROUPS},
            'zero_baseline_convention':'null; no division by zero'}


def release_runtime(runtime):
    # Called only by actual root-supervised worker after model load. CPU mock
    # tests supply a no-op release callback and never touch torch.cuda.
    torch=runtime.torch
    runtime.model=None
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()


def evaluate_policy(label,checkpoint,norm,dataset,chunk_plan,output,lo,hi,runtime_factory,release):
    folder=output/label;folder.mkdir();absolute=np.zeros(32,np.float64);square=absolute.copy();outside=absolute.copy();frames=0;any_outside=0;latencies=[];rows=[];target_outside=np.zeros(32,np.float64);target_any_outside=0
    runtime=None
    try:
        runtime=runtime_factory(checkpoint,norm,head_only=False)
        for number,item in enumerate(chunk_plan['chunks']):
            rgb,state,target,mask=dataset.sample(item['dataset_index'])
            mask=np.asarray(mask,bool);target=np.asarray(target,np.float32)
            expected=np.arange(50)<item['valid_frames']
            if mask.shape!=(50,) or not np.array_equal(mask,expected) or target.shape!=(50,32) or not np.isfinite(target).all():raise ValueError('Raw target/tail mask differs from frozen plan')
            rgb_hash=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
            prediction=runtime.infer(rgb,state,PROMPT,seed=item['seed'],num_steps=10)
            actions=np.asarray(prediction['actions'],np.float32)
            if actions.shape!=(50,32) or not np.isfinite(actions).all():raise ValueError('Nonfinite/incomplete prediction')
            delta=actions[mask].astype(np.float64)-target[mask].astype(np.float64)
            invalid=(actions[mask]<lo)|(actions[mask]>hi)
            target_invalid=(target[mask]<lo)|(target[mask]>hi)
            target_outside+=target_invalid.sum(0);target_any_outside+=int(target_invalid.any(1).sum())
            absolute+=np.abs(delta).sum(0);square+=np.square(delta).sum(0);outside+=invalid.sum(0)
            frames+=int(mask.sum());any_outside+=int(invalid.any(1).sum());latencies.append(float(prediction['inference_seconds']))
            path=folder/f'chunk_{number:05d}.npz'
            np.savez_compressed(path,predicted_actions=actions,raw_target_actions=target,mask=mask,rgb_sha256=np.asarray(rgb_hash))
            row={**item,'artifact':str(path),'artifact_sha256':sha(path),'rgb_sha256':rgb_hash,'inference_seconds':latencies[-1]};rows.append(row)
            with (folder/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        result=metrics(absolute,square,outside,frames,any_outside)
        result.update(raw_target_out_of_bounds_rate_per_channel=(target_outside/frames).tolist(),raw_target_any_channel_out_of_bounds_frame_rate=target_any_outside/frames,chunks=len(rows),episodes=len(chunk_plan['test_episode_ids']),latency_mean_seconds=float(np.mean(latencies)),latency_p95_seconds=float(np.quantile(latencies,.95)))
        write_new(folder/'metrics.json',result);return result,rows
    finally:
        if runtime is not None:release(runtime)


def evaluate(args,*,runtime_factory=None,dataset_factory=None,release=release_runtime):
    started=time.monotonic();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Fresh evidence output required')
    output.mkdir(parents=True);report={'schema':SCHEMA,'status':'failed','physical_success_evaluated':False,'test_used_for_checkpoint_selection':False}
    try:
        chunk_plan=plan(args.manifest);write_new(output/'chunk_plan.json',chunk_plan)
        manifest=json.loads(Path(args.manifest).read_text())
        if Path(manifest['dataset_root']).resolve()!=Path(args.dataset).resolve():raise ValueError('Dataset root mismatch')
        validation_ids=set(manifest['validation']);val_chunks=sum(math.ceil(x['length']/50) for x in manifest['files'] if x['episode_id'] in validation_ids)
        selected,receipt=resolve_selected(args.training_result,args.manifest,args.norm_stats,val_chunks)
        base=Path(args.base_checkpoint).resolve()
        official=(ROOT/'.scratch/pi05_g1/pi05_base_pytorch').resolve()
        if base!=official:raise ValueError('Official base checkpoint directory required')
        lo,hi=bounds(args.common_bounds)
        source_paths=[Path(__file__),ROOT/'experiments/pi05_g1_mobile/model_runtime.py',ROOT/'experiments/pi05_g1_mobile/contract.py',ROOT/'experiments/pi05_g1/training.py',ROOT/'experiments/pi05_g1/contract.py']
        identity={'manifest_sha256':sha(args.manifest),'norm_sha256':sha(args.norm_stats),'bounds_sha256':sha(args.common_bounds),'dataset_root':str(Path(args.dataset).resolve()),'source_sha256':{str(p):sha(p) for p in source_paths},'weights_sha256':{'baseline':sha(base/'model.safetensors'),'finetuned':sha(selected/'model.safetensors')},'selected':receipt,'test_episode_file_identity':[x for x in manifest['files'] if x['episode_id'] in manifest['test']]}
        if receipt['declared_weights_sha256']!=identity['weights_sha256']['finetuned']:raise ValueError('Selected weights bytes do not match frozen checkpoint receipt')
        write_new(output/'identity.json',identity)
        if runtime_factory is None:
            from .model_runtime import ModelRuntime
            runtime_factory=ModelRuntime
        if dataset_factory is None:
            from experiments.pi05_g1.training import EpisodeDataset
            dataset_factory=EpisodeDataset
        dataset=dataset_factory(args.dataset,args.manifest,'test',RawTargets())
        if list(dataset.validation_indices())!=[x['dataset_index'] for x in chunk_plan['chunks']]:raise ValueError('Dataset chunk coverage differs from metadata plan')
        baseline,base_rows=evaluate_policy('baseline',base,args.norm_stats,dataset,chunk_plan,output,lo,hi,runtime_factory,release)
        fine,fine_rows=evaluate_policy('finetuned',selected,args.norm_stats,dataset,chunk_plan,output,lo,hi,runtime_factory,release)
        if [x['rgb_sha256'] for x in base_rows]!=[x['rgb_sha256'] for x in fine_rows]:raise ValueError('Paired policies observed different raw RGB bytes')
        if baseline['valid_frames']!=chunk_plan['valid_frames'] or fine['valid_frames']!=chunk_plan['valid_frames']:raise ValueError('Incomplete heldout coverage')
        if any(sha(path)!=digest for path,digest in identity['source_sha256'].items()) or sha(args.manifest)!=identity['manifest_sha256'] or sha(args.norm_stats)!=identity['norm_sha256'] or sha(args.common_bounds)!=identity['bounds_sha256']:raise ValueError('Frozen evaluator inputs changed')
        report.update(status='completed',baseline=baseline,finetuned=fine,comparison=compare(baseline,fine),identity=identity,action_names=list(ACTION_NAMES),units=UNITS,prompt=PROMPT,head_only=False,flow_steps=10,raw_targets_unclipped=True,predictions_unclipped=True,full_test_episode_count=10,full_test_chunk_count=chunk_plan['chunk_count'])
    except BaseException as error:report['error']=repr(error)
    report['elapsed_seconds']=time.monotonic()-started;write_new(output/'offline_result.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--plan-only',action='store_true')
    for key in ('dataset','norm-stats','base-checkpoint','training-result','common-bounds'):p.add_argument('--'+key)
    args=p.parse_args()
    if args.plan_only:write_new(args.output,plan(args.manifest));return 0
    if any(getattr(args,key) is None for key in ('dataset','norm_stats','base_checkpoint','training_result','common_bounds')):p.error('Actual evaluation requires dataset/norm/base/training-result/common-bounds')
    report=evaluate(args);print(json.dumps({'schema':SCHEMA,'status':report['status'],'output':args.output}));return 0 if report['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
