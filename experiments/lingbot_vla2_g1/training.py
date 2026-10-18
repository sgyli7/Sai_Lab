"""Bounded expert-only fine-tuning; full validation, immutable selected/final deltas."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from experiments.pi05_g1.training import EpisodeDataset,MasterAdamW,memory_snapshot
from .data import SHARED,SPLIT,SPLIT_SHA,OWN,MeanStd,sha
from .runtime import Runtime
from .contract import PROMPT


def write_new(path,doc):
    with Path(path).open('x') as f:json.dump(doc,f,indent=2,allow_nan=False)


def validate(runtime,dataset,deadline,stop):
    if isinstance(dataset,dict):
        values={}
        for name,source in dataset.items():
            value=validate(runtime,source,deadline,stop)
            if value is None:return None
            values[name]=value
        if set(values)!={'original','supplementary'}:raise ValueError('Two complete validation sources required')
        return {'total_loss':sum(v['total_loss'] for v in values.values())*.5,'flow_loss':sum(v['flow_loss'] for v in values.values())*.5,'frames':sum(v['frames'] for v in values.values()),'chunks':sum(v['chunks'] for v in values.values()),'episodes':sum(v['episodes'] for v in values.values()),'source_metrics':values,'source_weights':{'original':.5,'supplementary':.5},'complete':True,'noise_seed':42}
    frames=0;chunks=0;total=0.;flow_total=0.
    runtime.model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            for index in dataset.validation_indices():
                if time.monotonic()>=deadline or stop():return None
                rgb,state,actions,mask=dataset.sample(index)
                loss,flow=runtime.loss(rgb,state,actions,mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation')
                count=int(mask.sum());total+=float(loss)*count;flow_total+=float(flow)*count;frames+=count;chunks+=1
        return {'total_loss':total/frames,'flow_loss':flow_total/frames,'frames':frames,'chunks':chunks,'episodes':len(dataset.episodes),'complete':True,'noise_seed':42}
    finally:runtime.model.train()


def snapshot_trainable(model,existing=None):
    current=existing if existing is not None else {}
    with torch.no_grad():
        for name,value in model.named_parameters():
            if not value.requires_grad:continue
            if name in current:current[name].copy_(value.detach())
            else:current[name]=value.detach().to(device='cpu',copy=True).contiguous()
    return current


def save_delta(path,snapshot,metadata):
    from safetensors.torch import save_file
    if path.exists():raise FileExistsError(path)
    path.mkdir();payload=path/'trainable.safetensors'
    save_file(snapshot,str(payload),metadata={'format':'pt','kind':'trainable_delta','base_receipt_sha256':metadata['base_receipt_sha256']})
    metadata={**metadata,'weights_sha256':sha(payload),'weights_bytes':payload.stat().st_size,'tensor_count':len(snapshot),'optimizer_saved':False}
    write_new(path/'metadata.json',metadata)
    with payload.open('rb') as f:
        import os
        os.fsync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return metadata


def train(args):
    if args.output.exists():raise FileExistsError('A new immutable output directory is required')
    if not 0<args.hours<=3.8 or not 0<args.max_steps<=800:raise ValueError('Bounded first-round recipe required')
    if sha(SPLIT)!=SPLIT_SHA:raise ValueError('Shared data split changed')
    args.output.mkdir(parents=True);started=time.monotonic();deadline=started+args.hours*3600
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    progress=(args.output/'progress.jsonl').open('x',buffering=1)
    step=0;best_step=0;best_snapshot=None;best=None;last=None;baseline=None
    identity={'base_receipt_sha256':sha(OWN/'bf16_conversion_receipt.json'),'norm_path':str(args.norm_stats.absolute()),'norm_sha256':sha(args.norm_stats),'supplementary_manifest':str(args.supplementary_manifest.absolute()) if args.supplementary_manifest else None,'supplementary_manifest_sha256':sha(args.supplementary_manifest) if args.supplementary_manifest else None,'contract_sha256':sha(OWN/'contract.json'),'split_sha256':SPLIT_SHA,'runtime_overlay_receipt_sha256':sha(OWN/'runtime_overlay_receipt.json'),'prompt':PROMPT,'test_opened':False}
    write_new(args.output/'recipe.json',{'hours':args.hours,'max_steps':args.max_steps,'microbatch':1,'accumulation':8,'lr':1e-5,'optimizer':'FP32-master AdamW','validation_every':100,'selection_loss':'complete validation flow_loss; total_loss includes training MoE regularizers','freeze_vlm':True,'auxiliary_teachers':False,'native_queries_preserved':True,'checkpoint_strategy':'Best candidate retained in bounded CPU RAM; final immutable selected and last trainable deltas only. All metric candidates retained.','optimizer_resume_saved':False,**identity})
    def event(name,**payload):
        doc={'event':name,'step':step,'elapsed_seconds':time.monotonic()-started,**payload}
        line=json.dumps(doc,allow_nan=False);progress.write(line+'\n')
        if name!='optimizer_step' or step%10==0:print(line,flush=True)
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
        normalizer=MeanStd(args.norm_stats)
        trainset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'train',normalizer)
        valset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'validation',normalizer)
        if args.supplementary_manifest:
            fresh=json.loads(args.supplementary_manifest.read_text())
            if fresh.get('status')!='completed' or not fresh.get('fresh_source_enabled') or fresh.get('source_manifest_sha256')!=SPLIT_SHA:raise ValueError('Fully qualified same-split supplementary source required')
            from experiments.pi05_g1_mobile.supplementary_data import SupplementaryEpisodes,MixedTraining
            trainset=MixedTraining(trainset,SupplementaryEpisodes(args.supplementary_manifest,'train',normalizer))
            valset={'original':valset,'supplementary':SupplementaryEpisodes(args.supplementary_manifest,'validation',normalizer)}
        runtime=Runtime(training=True,norm_stats=args.norm_stats);parameters=[p for p in runtime.model.parameters() if p.requires_grad]
        if runtime.model.config.bias_update_speed!=0:raise ValueError('This delta recipe requires the upstream real-robot fixed router bias')
        persistent=set(runtime.model.state_dict())
        fixed_buffers={name:value.detach().cpu().clone() for name,value in runtime.model.named_buffers() if name in persistent}
        optimizer=MasterAdamW(parameters,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
        event('training_started',parameters=runtime.parameter_elements,trainable=runtime.trainable_elements,**memory_snapshot(torch))
        baseline=validate(runtime,valset,deadline-240,lambda:stopping)
        if baseline is None:raise RuntimeError('Baseline full validation incomplete; no updates allowed')
        best=baseline;last=baseline;last_validated_step=0;event('validation_complete',validation=baseline,selected=True)
        while step<args.max_steps and time.monotonic()<deadline-300 and not stopping:
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True);loss_sum=0.;complete=True
            for micro in range(8):
                if stopping or time.monotonic()>=deadline-300:complete=False;break
                rgb,state,actions,mask=trainset.sample(random.randrange(len(trainset.indices)))
                loss,flow=runtime.loss(rgb,state,actions,mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/8).backward();loss_sum+=float(loss.detach())
            if not complete:optimizer.zero_grad(set_to_none=True);break
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();step+=1
            event('optimizer_step',loss=loss_sum/8,gradient_norm=float(norm),step_seconds=time.monotonic()-tick,**memory_snapshot(torch))
            if step%100==0:
                value=validate(runtime,valset,deadline-180,lambda:stopping)
                if value is not None:
                    last=value;last_validated_step=step;selected=value['flow_loss']<best['flow_loss']
                    if selected:best=value;best_step=step;best_snapshot=snapshot_trainable(runtime.model,best_snapshot)
                    event('validation_complete',validation=value,selected=selected)
                else:event('validation_incomplete')
        event('training_stopped',reason='signal' if stopping else 'step_limit' if step>=args.max_steps else 'time_budget')
        if last_validated_step!=step and not stopping:
            value=validate(runtime,valset,deadline-120,lambda:stopping)
            if value is not None:
                last=value;last_validated_step=step;selected=value['flow_loss']<best['flow_loss']
                if selected:best=value;best_step=step;best_snapshot=snapshot_trainable(runtime.model,best_snapshot)
                event('validation_complete',validation=value,selected=selected)
        for name,value in runtime.model.named_buffers():
            if name in fixed_buffers and not torch.equal(value.detach().cpu(),fixed_buffers[name]):raise ValueError('Unsaved persistent buffer changed: '+name)
        event('fixed_buffers_verified',persistent_buffer_count=len(fixed_buffers),elements=sum(value.numel() for value in fixed_buffers.values()))
        selected_path=None
        if best_snapshot is not None:
            selected_path=args.output/'selected'
            save_delta(selected_path,best_snapshot,{**identity,'step':best_step,'validation':best,'selected_by':'complete validation flow_loss only'})
        if best_step==step and best_snapshot is not None:final_path=selected_path
        else:
            # Two immutable payloads at most. The base remains the step0 candidate.
            final_path=args.output/'final'
            best_snapshot=snapshot_trainable(runtime.model,best_snapshot)
            save_delta(final_path,best_snapshot,{**identity,'step':step,'validation':last if last_validated_step==step else None,'selected_by':'final update boundary, not automatically selected'})
        result={'schema':'lingbot_vla2_g1_training_v1','status':'completed','optimizer_steps':step,'selected_step':best_step,'selected_delta':str(selected_path) if selected_path else None,'final_delta':str(final_path),'baseline_validation':baseline,'selected_validation':best,'last_validation':last,'final_validation_complete':last_validated_step==step,'stop_requested':stopping,'elapsed_seconds':time.monotonic()-started,**identity}
        write_new(args.output/'training_result.json',result);event('training_completed',selected_step=best_step,selected_delta=result['selected_delta']);return result
    except BaseException as error:
        if not (args.output/'training_result.json').exists():write_new(args.output/'training_result.json',{'schema':'lingbot_vla2_g1_training_v1','status':'failed','optimizer_steps':step,'error':repr(error),'elapsed_seconds':time.monotonic()-started,**identity})
        event('training_failed',error=repr(error));raise
    finally:
        progress.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--hours',type=float,default=3.5);p.add_argument('--max-steps',type=int,default=800);p.add_argument('--norm-stats',type=Path,default=OWN/'train_meanstd.json');p.add_argument('--supplementary-manifest',type=Path);args=p.parse_args();train(args)


if __name__=='__main__':main()
