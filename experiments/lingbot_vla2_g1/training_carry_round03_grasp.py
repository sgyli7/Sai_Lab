"""Round03 candidate: TRAIN60/40, prefix grasp phases50/50, LR5e-5 from R2."""
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
from .grasp_balanced_prefix import phase_identity


from .training import validate, snapshot_trainable, save_delta, write_new

PREFIX_SHA='3dab70a1de3b387197fabbbf73cb83af1db0eebe4e6c146f6d7577238d00b9df'


def admit(args):
    if args.supplementary_manifest is not None:
        raise ValueError('Only the independently audited native201 TRAIN-only prefix is admitted')
    prefix=Path(args.carry_prefix_manifest).absolute()
    initial=Path(args.initial_delta).absolute()
    if sha(prefix)!=PREFIX_SHA or initial!=OWN/'training_round02/selected':
        raise ValueError('Frozen prefix and selected round02 initialization required')
    if Path(args.norm_stats).absolute()!=OWN/'train_meanstd.json':
        raise ValueError('Original LingBot train-only mean/std must remain unchanged')
    norm=json.loads(args.norm_stats.read_text());split=json.loads(SPLIT.read_text())
    if sha(SPLIT)!=SPLIT_SHA or norm.get('split_manifest_sha256')!=SPLIT_SHA or norm.get('train_episode_ids')!=split['train'] or norm.get('fit_splits')!=['train']:
        raise ValueError('Original80 train-only normalization/split identity differs')
    proof_path=OWN/'round02_prefix_cpu_admission.json';proof=json.loads(proof_path.read_text())
    source=json.loads(prefix.read_text());peer_norm=Path(proof['peer_prefix_binding_norm'])
    if proof.get('status')!='passed' or proof.get('frames')!=1396 or proof.get('prefix_train_only') is not True or proof.get('source_full_task_success') is not False or proof.get('prefix_manifest_sha256')!=PREFIX_SHA or proof.get('LingBot_norm_sha256')!=sha(args.norm_stats) or source.get('norm_sha256')!=sha(peer_norm) or proof.get('peer_prefix_binding_norm_sha256')!=sha(peer_norm):
        raise ValueError('Explicit peer-norm/LingBot-norm bridge identity differs')
    metadata=json.loads((initial/'metadata.json').read_text())
    if metadata.get('step')!=200 or metadata.get('norm_sha256')!=sha(args.norm_stats) or metadata.get('base_receipt_sha256')!=sha(OWN/'bf16_conversion_receipt.json') or metadata.get('contract_sha256')!=sha(OWN/'contract.json') or metadata.get('weights_sha256')!=sha(initial/'trainable.safetensors'):
        raise ValueError('Initial selected parameter identity differs')
    return {'TRAIN_prefix_phase_sampling':phase_identity(prefix),'initial_delta':str(initial),'initial_weights_sha256':metadata['weights_sha256'],'initial_step':200,'initial_total_optimizer_updates':1000,'initial_metadata_sha256':sha(initial/'metadata.json'),'carry_prefix_manifest':str(prefix),'carry_prefix_manifest_sha256':PREFIX_SHA,'carry_prefix_train_only':True,'task_success_teacher':False,'peer_binding_norm_path':str(peer_norm),'peer_binding_norm_sha256':sha(peer_norm),'LingBot_norm_sha256':sha(args.norm_stats),'original80train_norm_unchanged':True,'prefix_CPU_admission_sha256':sha(proof_path),'optimizer_resume':False,'training_source_weights':{'original':.6,'carry_prefix':.4},'validation_scope':'original10_complete190chunks9240frames_only'}


def train(args):
    if args.output.exists():raise FileExistsError('A new immutable output directory is required')
    if not 0<args.hours<=.78 or not 0<args.max_steps<=600:raise ValueError('Bounded600-update third-round candidate required')
    if sha(SPLIT)!=SPLIT_SHA:raise ValueError('Shared data split changed')
    admission=admit(args)
    args.output.mkdir(parents=True);write_new(args.output/'carry_admission.json',admission)
    started=time.monotonic();deadline=started+args.hours*3600
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    progress=(args.output/'progress.jsonl').open('x',buffering=1)
    step=0;best_step=0;best_snapshot=None;best=None;last=None;baseline=None
    identity={'base_receipt_sha256':sha(OWN/'bf16_conversion_receipt.json'),'norm_path':str(args.norm_stats.absolute()),'norm_sha256':sha(args.norm_stats),'supplementary_manifest':str(args.supplementary_manifest.absolute()) if args.supplementary_manifest else None,'supplementary_manifest_sha256':sha(args.supplementary_manifest) if args.supplementary_manifest else None,'contract_sha256':sha(OWN/'contract.json'),'split_sha256':SPLIT_SHA,'runtime_overlay_receipt_sha256':sha(OWN/'runtime_overlay_receipt.json'),'prompt':PROMPT,'test_opened':False,**admission}
    write_new(args.output/'recipe.json',{'hours':args.hours,'max_steps':args.max_steps,'microbatch':1,'accumulation':8,'lr':5e-5,'optimizer':'FP32-master AdamW','validation_every':100,'selection_loss':'complete validation flow_loss; total_loss includes training MoE regularizers','freeze_vlm':True,'auxiliary_teachers':False,'native_queries_preserved':True,'checkpoint_strategy':'Best candidate retained in bounded CPU RAM; one immutable validation-selected payload only. All candidate/final metrics retained; a worse final payload is never materialized.','optimizer_resume_saved':False,**identity})
    def event(name,**payload):
        doc={'event':name,'step':step,'elapsed_seconds':time.monotonic()-started,**payload}
        line=json.dumps(doc,allow_nan=False);progress.write(line+'\n')
        if name!='optimizer_step' or step%10==0:print(line,flush=True)
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
        normalizer=MeanStd(args.norm_stats)
        trainset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'train',normalizer)
        valset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'validation',normalizer)
        from experiments.pi05_g1_mobile.carry_prefix_data import CarryMixture
        from .grasp_balanced_prefix import GraspBalancedPrefix as CarryPrefix
        prefix_data=CarryPrefix(args.carry_prefix_manifest,normalizer)
        trainset=CarryMixture(trainset,prefix_data)
        runtime=Runtime(training=True,delta=args.initial_delta,norm_stats=args.norm_stats);parameters=[p for p in runtime.model.parameters() if p.requires_grad]
        if runtime.model.config.bias_update_speed!=0:raise ValueError('This delta recipe requires the upstream real-robot fixed router bias')
        persistent=set(runtime.model.state_dict())
        fixed_buffers={name:value.detach().cpu().clone() for name,value in runtime.model.named_buffers() if name in persistent}
        optimizer=MasterAdamW(parameters,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
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
        if step==0:raise RuntimeError('No optimizer updates completed; no third-round training claim')
        selected_path=Path(args.initial_delta).absolute()
        if best_snapshot is not None:
            selected_path=args.output/'selected'
            save_delta(selected_path,best_snapshot,{**identity,'step':best_step,'validation':best,'selected_by':'complete validation flow_loss only'})
        final_path=selected_path if best_step==step and best_snapshot is not None else None
        result={'schema':'lingbot_vla2_g1_training_carry_round03_grasp_v1','status':'completed','optimizer_steps':step,'selected_step':best_step,'selected_delta':str(selected_path) if selected_path else None,'final_delta':str(final_path) if final_path else None,'final_weights_saved':final_path is not None,'baseline_validation':baseline,'selected_validation':best,'last_validation':last,'final_validation_complete':last_validated_step==step,'stop_requested':stopping,'elapsed_seconds':time.monotonic()-started,**identity}
        result['training_draw_counts']=dict(trainset.draw_counts)
        result['prefix_phase_draw_counts']=dict(getattr(prefix_data,'phase_draw_counts',{}))
        if result['prefix_phase_draw_counts'] and sum(result['prefix_phase_draw_counts'].values())!=trainset.draw_counts['carry_prefix']:raise ValueError('Actual phase draws differ from prefix draws')
        write_new(args.output/'training_result.json',result);event('training_completed',selected_step=best_step,selected_delta=result['selected_delta']);return result
    except BaseException as error:
        if not (args.output/'training_result.json').exists():write_new(args.output/'training_result.json',{'schema':'lingbot_vla2_g1_training_carry_round03_grasp_v1','status':'failed','optimizer_steps':step,'error':repr(error),'elapsed_seconds':time.monotonic()-started,**identity})
        event('training_failed',error=repr(error));raise
    finally:
        progress.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--hours',type=float,default=.78);p.add_argument('--max-steps',type=int,default=500);p.add_argument('--norm-stats',type=Path,default=OWN/'train_meanstd.json');p.add_argument('--supplementary-manifest',type=Path);p.add_argument('--initial-delta',type=Path,required=True);p.add_argument('--carry-prefix-manifest',type=Path,required=True);args=p.parse_args();train(args)


if __name__=='__main__':main()
