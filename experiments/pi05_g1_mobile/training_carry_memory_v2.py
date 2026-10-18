"""Memory v2: explicit idle-cache boundaries for unchanged carry training.

Copied schedule from frozen training_carry.py; no monkeypatch of old modules.
Unused caching allocator blocks are released only with no live accumulated grads.

Scheduling is explicitly copied from training.py, never monkeypatched.
Single family201 augments training only; not a full-task success demonstration.

Only the supervising root may launch GPU training. No test split is opened.
Every model is retained. Final FP32 Adam storage is explicit opt-in only.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
from experiments.pi05_g1.training import MasterAdamW,EpisodeDataset,memory_snapshot
from experiments.pi05_g1.model_runtime import QuantileNormalizer
from experiments.pi05_g1_round02.training import weighted_flow_loss,disk_estimate as _disk_estimate,require_disk,release_cpu_staging,save_baseline_checkpoint
from .contract import PROMPT,SCHEMA

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'.scratch/pi05_g1'
OFFICIAL_BASE=SCRATCH/'pi05_base_pytorch'
DATA=SCRATCH/'round03_mobile'
CHECKPOINT_SCHEMA='pi05_g1_mobile_training_checkpoint_v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.incomplete')
    with tmp.open('w') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


class StopRequest:
    def __init__(self):self.requested=False;self.signal=None
    def handler(self,signum,frame):self.requested=True;self.signal=int(signum)


def training_configuration(args):
    return {'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            'batch_size':1,'gradient_accumulation':8,'learning_rate':1e-5,'betas':[.9,.95],
            'epsilon':1e-8,'weight_decay':.01,'gradient_clip_norm':1.,'loss_weights':[1.]*32,
            'model_precision':'BF16 with official FP32 stability parameters','optimizer_precision':'FP32 masters and moments in memory',
            'validation_noise_seed':42,'save_optimizer_requested':args.save_optimizer,
            'exact_optimizer_resume_supported':False,'memory_policy':{'version':'idle_cuda_cache_at_optimizer_validation_save_boundaries_v2','live_tensor_precision_changed':False,'guard_gib':16,'empty_cache_is_not_OOM_cure':True},'data_strategy':{'kind':'single_family_carry_prefix_train_only' if args.carry_prefix_manifest else 'original_only','train_sampling_weights':{'original':.6,'carry_prefix':.4} if args.carry_prefix_manifest else {'original':1.},'validation':'original10complete190chunks_only','normalization':'original80train_only_unchanged','task_success_teacher':False}}


def disk_estimate(warmstart,max_steps,interval,reserve_gib,*,save_optimizer=False):
    estimate=_disk_estimate(warmstart,max_steps,interval,reserve_gib)
    if not save_optimizer:
        omitted=estimate['final_optimizer_bytes_upper_bound']
        estimate['final_optimizer_bytes_upper_bound']=0
        estimate['all_payload_bytes_upper_bound']-=omitted
        estimate['required_free_bytes']-=omitted
    estimate['save_optimizer_requested']=bool(save_optimizer)
    estimate['checkpoint_scope']='Full-parameter inference model; no exact optimizer resume entrypoint'
    return estimate


def preflight(args):
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve existing/partial training output; new directory required')
    if Path(args.checkpoint).resolve()!=OFFICIAL_BASE.resolve():raise ValueError('This round initializes only official pi05_base_pytorch, never apple-tuned weights')
    if not 0<args.hours<=4.3 or not 0<args.max_steps<=1600 or args.checkpoint_every!=400:
        raise ValueError('Frozen schedule requires hours<=4.3, max_steps<=1600, checkpoint_every400')
    if args.save_reserve_seconds<450 or args.validation_reserve_seconds<240 or args.hours*3600<=args.save_reserve_seconds+args.validation_reserve_seconds:
        raise ValueError('Need final-save450s + validation240s reserves inside deadline')
    if args.disk_reserve_gib<32:raise ValueError('Disk reserve must be at least32GiB')
    manifest_path=Path(args.manifest).resolve();manifest=json.loads(manifest_path.read_text())
    if manifest['revision']!='97f7b5a4135e4e6a206d394c9b6ec0253f1558a7':raise ValueError('Wrong mobile dataset revision')
    if [len(manifest[k]) for k in ['train','validation','test']]!=[80,10,10]:raise ValueError('Frozen whole-episode80/10/10 required')
    if any(set(manifest[a])&set(manifest[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Split overlap')
    dataset=Path(args.dataset).resolve()
    if Path(manifest['dataset_root']).resolve()!=dataset:raise ValueError('Dataset root differs from prepared manifest')
    info=json.loads((dataset/'meta/info.json').read_text())
    from .contract import DATASET_NAMES
    for key in ['observation.state','action']:
        if tuple(info['features'][key]['names'])!=tuple(DATASET_NAMES):raise ValueError('Reused EpisodeDataset requires exact audited joint ordering')
    if info['fps']!=50:raise ValueError('50Hz data required')
    normalizer=QuantileNormalizer(args.norm_stats)
    reference=DATA/'training_cpu_preflight_no_optimizer_v2.json'
    reference_data=json.loads(reference.read_text())
    if sha(manifest_path)!=reference_data['manifest_sha256'] or sha(args.norm_stats)!=reference_data['norm_sha256']:
        raise ValueError('Carry augmentation must retain frozen original80train normalization and source split')
    carry=None
    if args.carry_prefix_manifest:
        from .carry_prefix_data import admission as carry_admission
        carry=carry_admission(args.carry_prefix_manifest,args.manifest,args.norm_stats)
    disk=disk_estimate(args.checkpoint,args.max_steps,args.checkpoint_every,args.disk_reserve_gib,save_optimizer=args.save_optimizer)
    parent=output.parent
    while not parent.exists():parent=parent.parent
    free=require_disk(parent,disk['all_payload_bytes_upper_bound'],disk['reserve_bytes'])
    if parent.stat().st_dev!=(Path(args.checkpoint)/'model.safetensors').stat().st_dev:raise ValueError('Step0 hardlink requires same filesystem')
    frozen={}
    for p in [manifest_path,Path(args.norm_stats).resolve(),dataset/'meta/info.json',dataset/'meta/modality.json',dataset/'meta/tasks.jsonl',OFFICIAL_BASE/'conversion_provenance.json',OFFICIAL_BASE/'config.json']:
        frozen[str(p)]=sha(p)
    for directory in [ROOT/'experiments/pi05_g1_mobile',ROOT/'experiments/pi05_g1',ROOT/'experiments/pi05_g1_round02']:
        for p in sorted(directory.glob('*.py')):frozen[str(p)]=sha(p)
    for directory in [SCRATCH/'openpi/src',SCRATCH/'openpi/packages/openpi-client/src']:
        paths=sorted(directory.rglob('*.py'))
        if not directory.is_dir() or not paths:raise ValueError('Missing pinned OpenPI Python source root: '+str(directory))
        for p in paths:frozen[str(p.resolve())]=sha(p)
    tokenizer=SCRATCH/'venv/openpi-cache/big_vision/paligemma_tokenizer.model'
    if sha(tokenizer)!='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6':raise ValueError('Tokenizer changed')
    frozen[str(tokenizer)]=sha(tokenizer)
    frozen[str(reference.resolve())]=sha(reference)
    if carry:frozen.update(carry['extra_frozen_files'])
    source=Path(args.checkpoint)/'model.safetensors'
    return {'schema':'pi05_g1_mobile_training_preflight_v1','training_configuration':training_configuration(args),'disk_estimate':disk,'initial_free_disk_bytes':free,'manifest_sha256':sha(manifest_path),'norm_sha256':sha(args.norm_stats),'official_base':{'path':str(source.resolve()),'sha256':sha(source),'bytes':source.stat().st_size},'frozen_files':frozen,'prompt':PROMPT,'contract_schema':SCHEMA,'test_observations_opened':False,'dataset_root':str(dataset),'train_episode_ids':manifest['train'],'validation_episode_ids':manifest['validation'],'carry_prefix':carry,'validation_expected_chunks':sum(math.ceil(r['length']/50) for r in manifest['files'] if r['episode_id'] in manifest['validation'])}


def precision(runtime):
    return runtime.torch.autocast('cuda',dtype=runtime.torch.bfloat16) if runtime.device.type=='cuda' else contextlib.nullcontext()


def validate(runtime,dataset,weights,deadline):
    torch=runtime.torch;runtime.model.eval();total=0.;frames=0;chunks=0
    try:
        devices=[torch.cuda.current_device()] if runtime.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(42)
            if devices:torch.cuda.manual_seed_all(42)
            with torch.no_grad(),precision(runtime):
                for index in dataset.validation_indices():
                    if time.monotonic()>=deadline:return None
                    rgb,state,actions,mask=dataset.sample(index)
                    losses=runtime.model(runtime.observation(rgb,state,PROMPT),torch.as_tensor(actions,device=runtime.device)[None])
                    loss=weighted_flow_loss(losses,torch.as_tensor(mask,device=runtime.device),weights)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation')
                    count=int(mask.sum());total+=float(loss)*count;frames+=count;chunks+=1
        if time.monotonic()>deadline:return None
        return {'flow_loss':total/frames,'frames':frames,'chunks':chunks,'episodes':len(dataset.episodes),'noise_seed':42,'complete':True}
    finally:runtime.model.train()


def update(runtime,dataset,optimizer,weights,deadline):
    """Exactly8 microbatches or no update; a signal is checked by caller at boundary."""
    torch=runtime.torch;optimizer.zero_grad(set_to_none=True);total=0.
    for _ in range(8):
        if time.monotonic()>=deadline:
            optimizer.zero_grad(set_to_none=True);return None
        rgb,state,actions,mask=dataset.sample(random.randrange(len(dataset.indices)))
        with precision(runtime):
            losses=runtime.model(runtime.observation(rgb,state,PROMPT),torch.as_tensor(actions,device=runtime.device)[None])
            loss=weighted_flow_loss(losses,torch.as_tensor(mask,device=runtime.device),weights)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        (loss/8).backward();total+=float(loss.detach())
    if time.monotonic()>=deadline:
        optimizer.zero_grad(set_to_none=True);return None
    norm=torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if runtime.device.type=='cuda':torch.cuda.synchronize()
    return {'flow_loss':total/8,'gradient_norm':float(norm),'microbatches':8}


def checkpoint(runtime,optimizer,path,metadata,disk,*,include_optimizer):
    from safetensors.torch import save_model
    if path.exists() or path.with_name(path.name+'.incomplete').exists():raise FileExistsError('Checkpoint exists, preserve evidence')
    require_disk(path.parent,disk['model_bytes']+(disk['final_optimizer_bytes_upper_bound'] if include_optimizer else 0),disk['reserve_bytes'])
    temp=path.with_name(path.name+'.incomplete');temp.mkdir();started=time.monotonic();torch=runtime.torch
    optimizer.zero_grad(set_to_none=True);save_model(runtime.model,str(temp/'model.safetensors'));cleanup=[release_cpu_staging()]
    if include_optimizer:
        torch.save({'schema':'pi05_g1_mobile_optimizer_state_v1','optimizer':optimizer.state_dict(),'step':metadata['step'],'identity':metadata['identity'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp/'training_state.pt');cleanup.append(release_cpu_staging())
    metadata={**metadata,'optimizer_payload_saved':include_optimizer,'exact_optimizer_resume_supported':False,'checkpoint_scope':'Full-parameter inference model; FP32 Adam/RNG payload included but no resume CLI' if include_optimizer else 'Full-parameter inference model and metadata only; not exact optimizer resume','weights_sha256':sha(temp/'model.safetensors'),'checkpoint_save_seconds':time.monotonic()-started,'cpu_staging_cleanup':cleanup}
    write_new(temp/'metadata.json',metadata);write_new(temp/'metadata_sha256.json',{'metadata_sha256':sha(temp/'metadata.json')})
    os.replace(temp,path);return metadata



class IdleMemoryBoundaries:
    """Explicit scheduling seam; never evict live tensors or alter training RNG."""
    def __init__(self,runtime,optimizer,event):
        self.runtime=runtime;self.optimizer=optimizer;self.event=event
    def run(self,phase):
        # Never discard partially accumulated gradients to obtain more memory.
        for parameter in self.optimizer.model_parameters+self.optimizer.master_parameters:
            if parameter.grad is not None:raise RuntimeError('Memory boundary requires cleared model AND master gradients')
        runtime=self.runtime;torch=runtime.torch;started=time.monotonic()
        before=memory_snapshot(torch)
        cpu=release_cpu_staging()
        cuda_released=False
        if runtime.device.type=='cuda':
            if not torch.cuda.is_initialized():raise RuntimeError('CUDA runtime unexpectedly uninitialized at memory boundary')
            torch.cuda.synchronize(runtime.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(runtime.device)
            cuda_released=True
        after=memory_snapshot(torch)
        record={'phase':phase,'before':before,'after':after,'cpu_staging':cpu,
                'cuda_empty_cache_called':cuda_released,'elapsed_seconds':time.monotonic()-started,
                'interpretation':'Unused caching blocks only; live models/masters/moments unchanged. No guarantee all reserved pages returned or peakmemoryfit.'}
        self.event('idle_memory_boundary',**record)
        return record


def train(args):
    started=time.monotonic();deadline=started+args.hours*3600
    admission=preflight(args)
    if args.preflight_only:return admission
    output=Path(args.output).resolve();output.mkdir(parents=True);write_new(output/'input_manifest.json',admission)
    request=StopRequest();previous={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGINT]}
    for s in previous:signal.signal(s,request.handler)
    step=0;best=float('inf');selected=None;baseline=None;stopped=None;final=None
    log=(output/'progress.jsonl').open('x',buffering=1)
    def event(name,**payload):
        row={'event':name,'step':step,'elapsed_seconds':time.monotonic()-started,**payload};line=json.dumps(row,allow_nan=False);log.write(line+'\n')
        if name!='optimizer_step' or step%10==0:print(line,flush=True)
    identity={'manifest_sha256':admission['manifest_sha256'],'norm_sha256':admission['norm_sha256'],'prompt':PROMPT,'contract_schema':SCHEMA,'input_manifest_sha256':sha(output/'input_manifest.json')}
    if admission.get('carry_prefix'):
        identity['carry_prefix_manifest_sha256']=admission['carry_prefix']['manifest_sha256']
        identity['carry_family_id']='native201'
        identity['carry_prefix_train_only']=True
    def metadata(value):return {'schema':CHECKPOINT_SCHEMA,'step':step,'identity':identity,'warmstart':{'checkpoint':str(OFFICIAL_BASE),'weights_sha256':admission['official_base']['sha256']},'training_configuration':training_configuration(args),'validation_complete':value is not None,'validation_flow_loss':value['flow_loss'] if value else None,'validation':value,'optimizer_master_dtype':'float32','full_parameter_training':True,'elapsed_seconds':time.monotonic()-started}
    def select(path,value):
        nonlocal best,selected
        if value is not None and value['flow_loss']<best:
            best=value['flow_loss'];selected=str(path);atomic(output/'best_checkpoint.json',{'path':selected,'validation_flow_loss':best,'metadata_sha256':sha(path/'metadata.json'),'step':step})
    try:
        from .model_runtime import ModelRuntime
        import torch
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        event('cpu_admission_complete',disk_estimate=admission['disk_estimate'],prompt=PROMPT)
        normalizer=QuantileNormalizer(args.norm_stats)
        trainset=EpisodeDataset(args.dataset,args.manifest,'train',normalizer);valset=EpisodeDataset(args.dataset,args.manifest,'validation',normalizer)
        if args.carry_prefix_manifest:
            from .carry_prefix_data import CarryPrefix,CarryMixture
            prefix=CarryPrefix(args.carry_prefix_manifest,normalizer)
            trainset=CarryMixture(trainset,prefix)
        runtime=ModelRuntime(args.checkpoint,args.norm_stats,training=True,head_only=False)
        optimizer=MasterAdamW(runtime.model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        weights=torch.ones(32,device=runtime.device)
        validation_deadline=deadline-args.save_reserve_seconds;training_deadline=validation_deadline-args.validation_reserve_seconds
        event('training_started',config={'batch_size':1,'gradient_accumulation':8,'learning_rate':1e-5,'max_steps':args.max_steps,'hours':args.hours,'checkpoint_every':400,'save_optimizer':args.save_optimizer},model_dtype_counts=runtime.dtype_summary)
        boundaries=IdleMemoryBoundaries(runtime,optimizer,event)
        def complete_validation():
            boundaries.run('before_validation')
            try:return validate(runtime,valset,weights,validation_deadline)
            finally:boundaries.run('after_validation')
        def save_model_checkpoint(path,value,include_optimizer):
            boundaries.run('before_checkpoint')
            try:return checkpoint(runtime,optimizer,path,metadata(value),admission['disk_estimate'],include_optimizer=include_optimizer)
            finally:boundaries.run('after_checkpoint')
        baseline=complete_validation()
        if baseline is None:raise RuntimeError('Baseline full validation incomplete; no updates permitted')
        path=output/'step_000000';saved=save_baseline_checkpoint(args.checkpoint,path,metadata(baseline));write_new(path/'metadata_sha256.json',{'metadata_sha256':sha(path/'metadata.json')});select(path,baseline)
        event('checkpoint_saved',path=str(path),validation=baseline,storage='official-base hardlink')
        last_value=baseline;last_value_step=0
        while step<args.max_steps and time.monotonic()<training_deadline and not request.requested:
            tick=time.monotonic();result=update(runtime,trainset,optimizer,weights,training_deadline)
            if result is None:break
            step+=1;boundaries.run('optimizer_boundary');event('optimizer_step',**result,source_draw_counts=getattr(trainset,'draw_counts',None),step_seconds=time.monotonic()-tick,**memory_snapshot(torch))
            if request.requested:break
            if step%args.checkpoint_every==0:
                value=complete_validation();last_value=value;last_value_step=step
                path=output/f'step_{step:06d}';save_model_checkpoint(path,value,False);select(path,value);event('checkpoint_saved',path=str(path),validation=value)
        stopped={'reason':'signal' if request.requested else 'max_steps' if step>=args.max_steps else 'time_budget','signal':request.signal,'step':step,'complete_optimizer_update_boundary':True,'elapsed_seconds':time.monotonic()-started}
        event('training_stopped',**{k:v for k,v in stopped.items() if k not in ['step','elapsed_seconds']})
        value=last_value if last_value_step==step else complete_validation()
        final=output/f'final_{step:06d}';saved=save_model_checkpoint(final,value,args.save_optimizer);select(final,value)
        atomic(output/'last_checkpoint.json',{'path':str(final),'step':step,'metadata_sha256':sha(final/'metadata.json')})
        for p,h in admission['frozen_files'].items():
            if sha(p)!=h:raise ValueError('Frozen training input changed: '+p)
        if sha(admission['official_base'].get('path',Path(args.checkpoint)/'model.safetensors'))!=admission['official_base']['sha256']:
            raise ValueError('Official base payload changed during training')
        if time.monotonic()>deadline:raise TimeoutError('Finalization exceeded absolute training deadline')
        if value is None:raise RuntimeError('Final validation incomplete; saved final cannot claim completed training')
        report={'schema':'pi05_g1_mobile_training_result_v1','status':'completed','training_stopped':stopped,'step':step,'selected_checkpoint':selected,'selected_validation_flow_loss':best,'baseline_validation':baseline,'final_validation':value,'final_checkpoint':str(final),'training_configuration':training_configuration(args),'optimizer_payload_saved':saved['optimizer_payload_saved'],'final_optimizer_includes_fp32_master':saved['optimizer_payload_saved'],'exact_optimizer_resume_supported':False,'checkpoint_scope':saved['checkpoint_scope'],'elapsed_seconds':time.monotonic()-started,'identity':identity,'test_loaded':False,'all_checkpoints_retained':True}
        write_new(output/'training_result.json',report);event('training_completed',selected_checkpoint=selected);return report
    except BaseException as error:
        report={'schema':'pi05_g1_mobile_training_result_v1','status':'failed','step':step,'training_stopped':stopped,'selected_checkpoint':selected,'baseline_validation':baseline,'final_checkpoint':str(final) if final else None,'error':repr(error),'elapsed_seconds':time.monotonic()-started,'test_loaded':False,'external_kill_may_prevent_this_receipt':True}
        if not (output/'training_result.json').exists():write_new(output/'training_result.json',report)
        event('training_failed',error=repr(error));raise
    finally:
        log.close()
        for s,handler in previous.items():signal.signal(s,handler)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(OFFICIAL_BASE));p.add_argument('--dataset',default=str(DATA/'data/lerobot'))
    p.add_argument('--manifest',default=str(DATA/'prepared/split_manifest.json'));p.add_argument('--norm-stats',default=str(DATA/'prepared/train_norm_stats.json'));p.add_argument('--output',required=True)
    p.add_argument('--carry-prefix-manifest',help='Optional independently audited family201 TRAIN-only physical carry prefix; never a full-success teacher')
    p.add_argument('--hours',type=float,default=4.3);p.add_argument('--max-steps',type=int,default=1600);p.add_argument('--checkpoint-every',type=int,default=400);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--disk-reserve-gib',type=float,default=32);p.add_argument('--save-reserve-seconds',type=float,default=450);p.add_argument('--validation-reserve-seconds',type=float,default=240);p.add_argument('--preflight-only',action='store_true');p.add_argument('--save-optimizer',action='store_true',help='Opt in to final FP32 master/Adam+RNG payload; default saves full inference model only. No optimizer resume entrypoint.')
    return p


def main():
    args=parser().parse_args();result=train(args)
    if args.preflight_only:print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
