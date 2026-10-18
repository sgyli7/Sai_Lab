"""Round02 weights-only warmstart; explicit teacher mixture and loss weights.

All periodic checkpoints are model+metadata only; exactly one final checkpoint
also stores fresh FP32 master/Adam state. No checkpoint is deleted or overwritten.
Only root may launch this program inside the round02 GPU budget.
"""
from __future__ import annotations
import argparse
import datetime
import ctypes
import gc
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import struct
import time

import numpy as np

from experiments.pi05_g1.model_runtime import ModelRuntime, QuantileNormalizer
from experiments.pi05_g1.training import MasterAdamW, memory_snapshot
from .datasets import build_sources, sha256
from .checkpoint_lineage import (CHECKPOINT_SCHEMA, LINEAGE_SCHEMA, ROSTER_SCHEMA,
    collect_candidates, resolve_lineage, select_global_best, validate_failed_parent)


def load_loss_weights(path):
    data=json.loads(Path(path).read_text());data=data.get('actions') if isinstance(data,dict) else data
    weights=np.asarray(data,np.float32)
    if weights.shape!=(32,) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('Explicit32 finite nonnegative action loss weights, with positive sum, required')
    return weights


def weighted_flow_loss(losses,mask,weights):
    valid=mask.to(dtype=losses.dtype)[None,:,None]
    weight=weights.to(dtype=losses.dtype)[None,None,:]*valid
    return (losses*weight).sum()/weight.sum().clamp_min(1e-12)


def disk_estimate(warmstart,max_steps,interval,reserve_gib):
    weights=Path(warmstart)/'model.safetensors'
    with weights.open('rb') as handle:
        header_length=struct.unpack('<Q',handle.read(8))[0];header=json.loads(handle.read(header_length))
    numel=sum(math.prod(value['shape']) for key,value in header.items() if key!='__metadata__')
    model=weights.stat().st_size
    # All unique parameters get masters; conservatively reserve two moments even
    # for the small upstream parameters outside the action-loss forward graph.
    optimizer=numel*12+2*1024**2
    periodic=max_steps//interval
    # Include step0 as a full payload conservatively, although it is a hardlink.
    payload=(periodic+2)*(model+2*1024**2)+optimizer
    return {'model_bytes':model,'unique_parameter_elements':numel,'periodic_checkpoint_count_upper_bound':periodic,'baseline_checkpoint_count':1,'baseline_storage':'hardlink; admission conservatively reserves a full model','final_optimizer_bytes_upper_bound':optimizer,'all_payload_bytes_upper_bound':payload,'reserve_bytes':int(reserve_gib*1024**3),'required_free_bytes':payload+int(reserve_gib*1024**3)}


def require_disk(output,required,reserve):
    free=shutil.disk_usage(output).free
    if free<required+reserve:raise RuntimeError(f'Disk admission rejected: free={free}, next_payload={required}, reserve={reserve}; no archives removed')
    return free


def source_receipts(sources):
    return [{key:value for key,value in source.items() if key not in ('train','validation')} for source in sources]


def validation_improves(value,best):
    return value is not None and math.isfinite(value['combined_flow_loss']) and value['combined_flow_loss']<best


def release_cpu_staging():
    """Best-effort release of allocator-free host pages; never touch live tensors."""
    def snapshot():
        result={}
        for filename,keys in (('/proc/self/status',('VmRSS','RssAnon','RssFile')),('/proc/meminfo',('MemAvailable',))):
            try:
                for line in Path(filename).read_text().splitlines():
                    key,_,value=line.partition(':')
                    if key in keys:result[key+'_bytes']=int(value.split()[0])*1024
            except (OSError,ValueError):pass
        return result
    receipt={'before':snapshot(),'libc':list(platform.libc_ver()),'trim_available':False,'trim_return':None}
    try:
        receipt['gc_collected']=gc.collect()
        if platform.system()=='Linux' and receipt['libc'][0]=='glibc':
            trim=ctypes.CDLL(None).malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
            receipt['trim_available']=True;receipt['trim_return']=int(trim(0))
        else:receipt['diagnostic']='malloc_trim requires Linux glibc; no fallback memory mutation'
    except Exception as error:receipt['diagnostic']=repr(error)
    receipt['after']=snapshot()
    receipt['interpretation']='Best effort only; no claim that all staging/driver pages were reclaimed'
    return receipt


def save_baseline_checkpoint(source,path,metadata):
    """Keep the exact warmstart storage; never serialize model or optimizer here."""
    source=Path(source)/'model.safetensors';path=Path(path)
    if metadata.get('step')!=0 or metadata.get('validation_complete') is not True:
        raise ValueError('Step0 requires complete validation before any update')
    expected=metadata['warmstart']['weights_sha256']
    if sha256(source)!=expected:raise ValueError('Warmstart weights changed before baseline checkpoint')
    temporary=path.with_name(path.name+'.incomplete')
    if path.exists() or temporary.exists():raise FileExistsError('Baseline evidence already exists')
    temporary.mkdir();started=time.monotonic()
    # Do not fall back to copying or saving: a different filesystem is a clear
    # admission failure, rather than an unbudgeted allocation or dtype recast.
    os.link(source,temporary/'model.safetensors')
    saved={**metadata,'optimizer_payload_saved':False,'round_optimizer_updates':metadata.get('cumulative_round_updates',0),'local_optimizer_updates':0,
           'weights_sha256':expected,'weights_storage':'hardlink to frozen warmstart; no reserialization or dtype conversion',
           'checkpoint_save_seconds':time.monotonic()-started,
           'checkpoint_complete_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (temporary/'metadata.json').write_text(json.dumps(saved,indent=2,allow_nan=False))
    os.replace(temporary,path)
    return saved


def validate(runtime,sources,loss_weights,deadline):
    torch=runtime.torch;runtime.model.eval();per_source={}
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for source in sources:
                    total,frames,done=0.,0,0
                    for index in source['validation'].validation_indices():
                        if time.monotonic()>=deadline:return None
                        rgb,state,actions,mask=source['validation'].sample(index)
                        losses=runtime.model(runtime.observation(rgb,state),torch.as_tensor(actions,device=runtime.device)[None])
                        loss=weighted_flow_loss(losses,torch.as_tensor(mask,device=runtime.device),loss_weights)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
                        n=int(mask.sum());total+=float(loss)*n;frames+=n;done+=1
                    per_source[source['name']]={'flow_loss':total/frames,'frames':frames,'chunks':done,'episodes':source['validation_episodes'],'validation_weight':source['validation_weight']}
        denominator=sum(source['validation_weight'] for source in sources)
        combined=sum(source['validation_weight']*per_source[source['name']]['flow_loss'] for source in sources)/denominator
        return {'combined_flow_loss':combined,'sources':per_source,'selection_formula':'weighted mean of per-source mean masked flow losses; weights explicitly supplied by root'}
    finally:runtime.model.train()


def save_checkpoint(runtime,optimizer,path,metadata,*,include_optimizer,disk,reserve):
    from safetensors.torch import save_model
    torch=runtime.torch
    if path.exists() or path.with_name(path.name+'.incomplete').exists():raise FileExistsError('Checkpoint or partial evidence already exists; refusing overwrite')
    require_disk(path.parent,disk['model_bytes']+(disk['final_optimizer_bytes_upper_bound'] if include_optimizer else 0),reserve)
    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    temporary=path.with_name(path.name+'.incomplete');temporary.mkdir()
    started=time.monotonic()
    save_model(runtime.model,str(temporary/'model.safetensors'))
    cleanup=[{'phase':'after_model_serialize',**release_cpu_staging()}]
    if include_optimizer:
        torch.save({'optimizer':optimizer.state_dict(),'step':metadata['step'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'round02_data_identity':metadata['round02_data_identity']},temporary/'training_state.pt')
        cleanup.append({'phase':'after_optimizer_serialize',**release_cpu_staging()})
    metadata={**metadata,'optimizer_payload_saved':include_optimizer,'cpu_staging_cleanup':cleanup,'checkpoint_save_seconds':time.monotonic()-started,'checkpoint_complete_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (temporary/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False));os.replace(temporary,path)
    return metadata


def train(args):
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('New round02 output must not exist; do not append to a prior run')
    output.parent.mkdir(parents=True,exist_ok=True)
    weights_np=load_loss_weights(args.loss_weights)
    normalizer=QuantileNormalizer(args.norm_stats)
    disk=disk_estimate(args.warmstart,args.max_steps,args.checkpoint_every,args.disk_reserve_gib)
    initial_free=require_disk(output.parent,disk['all_payload_bytes_upper_bound'],disk['reserve_bytes'])
    sources=build_sources(args.data_config,normalizer)
    identity={'data_config_sha256':sha256(args.data_config),'norm_sha256':sha256(args.norm_stats),'loss_weights_sha256':sha256(args.loss_weights),'sources':source_receipts(sources)}
    parent=Path(args.warmstart).resolve();previous=json.loads((parent/'metadata.json').read_text())
    recovery=previous.get('schema')==CHECKPOINT_SCHEMA;failure_receipt=None
    if recovery:
        if not args.recovery_failed_budget:raise ValueError('Round02 recovery requires explicit closed failed-budget receipt')
        failure_receipt,parent_lineage=validate_failed_parent(parent,identity,args.recovery_failed_budget,args.max_steps,args.genesis_checkpoint)
        parent_weights=failure_receipt['source_weights_sha256'];inherited=parent_lineage['cumulative_round_updates']
        directories=[parent.parent,*map(lambda value:Path(value).resolve(),args.candidate_training_dir),output]
    else:
        if previous.get('step')!=1200 or parent!=Path(args.genesis_checkpoint).resolve():raise ValueError('Genesis warmstart must be the declared original step1200')
        if args.recovery_failed_budget or args.candidate_training_dir:raise ValueError('Extra runs/failure receipts require round02 recovery')
        parent_lineage=resolve_lineage(parent,verify_weights=True);parent_weights=parent_lineage['genesis']['weights_sha256'];inherited=0;directories=[output]
    directories=list(dict.fromkeys(directories))
    immediate={'path':str(parent),'metadata_sha256':sha256(parent/'metadata.json'),'weights_sha256':parent_weights,'local_step':previous['step'],'cumulative_round_updates':inherited}
    warmstart={'checkpoint':str(parent),'weights_sha256':parent_weights,'source_optimizer_steps':previous['step'],'source_validation_flow_loss':previous['validation_flow_loss'],'optimizer_reset':True,'source_optimizer_payload_loaded':False,'genesis_checkpoint':parent_lineage['genesis_checkpoint']}
    previous_collection=collect_candidates([directory for directory in directories if directory!=output],args.data_config,args.norm_stats,args.loss_weights) if recovery else None
    if previous_collection and previous_collection['identity']!=identity:raise ValueError('Prior candidate identity differs from current source receipts')
    original=next(source for source in sources if source['kind']=='lerobot')
    compatible_identity={'manifest_sha256':original['manifest_sha256'],'norm_sha256':identity['norm_sha256']}
    started=time.monotonic();deadline=started+args.hours*3600
    output.mkdir()
    roster={'schema':ROSTER_SCHEMA,'training_dirs':list(map(str,directories)),'current_training_dir':str(output)}
    (output/'candidate_training_dirs.json').write_text(json.dumps(roster,indent=2))
    failure_reference=None
    if failure_receipt:
        failure_path=output/'parent_failure_receipt.json';failure_path.write_text(json.dumps(failure_receipt,indent=2,allow_nan=False))
        failure_reference={'path':str(failure_path),'sha256':sha256(failure_path)}
    log=(output/'progress.jsonl').open('x',buffering=1)
    def event(**values):
        values['elapsed_seconds']=time.monotonic()-started
        line=json.dumps(values,allow_nan=False);log.write(line+'\n');print(line,flush=True)
    event(event='cpu_admission_complete',disk_estimate=disk,initial_free_disk_bytes=initial_free,round02_data_identity=identity,warmstart=warmstart,inherited_round_updates=inherited,candidate_training_dirs=roster,source_failure_receipt=failure_reference,loss_weights=weights_np.tolist())
    import torch
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    runtime=ModelRuntime(args.warmstart,args.norm_stats,training=True,head_only=False)
    optimizer=MasterAdamW(runtime.model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    loss_weights=torch.as_tensor(weights_np,device=runtime.device)
    event(event='training_started',step=0,inherited_round_updates=inherited,cumulative_round_updates=inherited,warmstart=warmstart,optimizer_master_dtype='float32',optimizer_state='fresh AdamW; no optimizer resume',model_dtype_counts=runtime.dtype_summary,source_counts=source_receipts(sources),config={'batch_size':1,'gradient_accumulation':8,'learning_rate':1e-5,'max_steps':args.max_steps,'hours':args.hours,'checkpoint_every':args.checkpoint_every,'head_only':False},**memory_snapshot(torch))
    step=0;prior_best=select_global_best(previous_collection) if previous_collection else None
    best=prior_best['validation_flow_loss'] if prior_best else float('inf');last_validation=None;last_validation_step=None
    best_path=prior_best['path'] if prior_best else None
    validation_deadline=deadline-args.save_reserve_seconds
    training_deadline=validation_deadline-args.validation_reserve_seconds
    def metadata(value):
        lineage={'schema':LINEAGE_SCHEMA,'immediate_parent':immediate,'genesis':parent_lineage['genesis'],'inherited_round_updates':inherited,'local_step':step,'cumulative_round_updates':inherited+step,'optimizer_reset':True,'source_failure_receipt':failure_reference}
        return {'schema':CHECKPOINT_SCHEMA,'step':step,'local_step':step,'inherited_round_updates':inherited,'cumulative_round_updates':inherited+step,'checkpoint_lineage':lineage,'warmstart':warmstart,'validation_flow_loss':value['combined_flow_loss'] if value else None,'validation_sources':value['sources'] if value else None,'validation_complete':value is not None,'best_validation_loss':best if np.isfinite(best) else None,'identity':compatible_identity,'round02_data_identity':identity,'loss_weights':weights_np.tolist(),'optimizer_master_dtype':'float32','model_dtype_counts':runtime.dtype_summary,'head_only':False,'elapsed_seconds':time.monotonic()-started,'selection_policy':'minimum completed weighted validation across all declared candidate runs; no test data used'}
    baseline=validate(runtime,sources,loss_weights,validation_deadline)
    event(event='validation',step=0,complete=baseline is not None,metrics=baseline,phase='warmstart_before_any_update')
    if baseline is None:raise RuntimeError('Step0 validation incomplete; refusing updates without a complete warmstart candidate')
    if validation_improves(baseline,best):best=baseline['combined_flow_loss'];best_path=str(output/'step_000000')
    last_validation=baseline;last_validation_step=0
    save_baseline_checkpoint(args.warmstart,output/'step_000000',metadata(baseline))
    (output/'best_checkpoint.json').write_text(json.dumps({'path':best_path,'validation_flow_loss':best}))
    event(event='checkpoint_saved',step=0,path='step_000000',optimizer_payload_saved=False,weights_sha256=warmstart['weights_sha256'],storage='hardlink')
    draw_counts={source['name']:{'prefix':0,'full':0} for source in sources}
    while step<args.max_steps and time.monotonic()<training_deadline:
        optimizer.zero_grad(set_to_none=True);loss_total=0.;complete=0;batch_sources=[]
        batch_draw_counts={source['name']:{'prefix':0,'full':0} for source in sources}
        torch.cuda.synchronize();tick=time.monotonic()
        for _ in range(8):
            if time.monotonic()>=training_deadline:break
            source=random.choices(sources,weights=[source['train_sampling_weight'] for source in sources],k=1)[0]
            dataset=source['train']
            index,branch=dataset.draw_index(random) if hasattr(dataset,'draw_index') else (random.randrange(len(dataset.indices)),'full')
            draw_counts[source['name']][branch]+=1;batch_draw_counts[source['name']][branch]+=1
            rgb,state,actions,mask=dataset.sample(index)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses=runtime.model(runtime.observation(rgb,state),torch.as_tensor(actions,device=runtime.device)[None])
                loss=weighted_flow_loss(losses,torch.as_tensor(mask,device=runtime.device),loss_weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            (loss/8).backward();loss_total+=float(loss.detach());complete+=1;batch_sources.append(source['name'])
        if complete!=8:optimizer.zero_grad(set_to_none=True);break
        norm=torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),1.,error_if_nonfinite=True)
        probes={}
        for name,parameter in runtime.model.named_parameters():
            if name in ('action_in_proj.weight','action_out_proj.weight') or name.endswith('layers.0.self_attn.q_proj.weight'):
                label='vision' if 'vision_tower' in name else 'VLM' if 'language_model' in name else 'expert' if 'gemma_expert' in name else name.split('.')[0]
                probes[label]=(parameter,parameter.detach().reshape(-1)[:128].float().clone())
        optimizer.step();step+=1;torch.cuda.synchronize()
        updates={name:{'sampled_elements':before.numel(),'changed_elements':int((parameter.detach().reshape(-1)[:128].float()!=before).sum()),'max_abs_delta':float((parameter.detach().reshape(-1)[:128].float()-before).abs().max())} for name,(parameter,before) in probes.items()}
        optimizer.zero_grad(set_to_none=True)
        event(event='optimizer_step',step=step,inherited_round_updates=inherited,cumulative_round_updates=inherited+step,flow_loss=loss_total/8,gradient_norm=float(norm),step_seconds=time.monotonic()-tick,batch_sources=batch_sources,time_sampling_draw_counts=batch_draw_counts,cumulative_time_sampling_draw_counts=draw_counts,parameter_update_probes=updates,**memory_snapshot(torch))
        if step%args.checkpoint_every==0:
            value=validate(runtime,sources,loss_weights,validation_deadline);last_validation=value;last_validation_step=step
            event(event='validation',step=step,complete=value is not None,metrics=value)
            improved=validation_improves(value,best)
            if improved:best=value['combined_flow_loss'];best_path=str(output/f'step_{step:06d}')
            name=f'step_{step:06d}';saved=save_checkpoint(runtime,optimizer,output/name,metadata(value),include_optimizer=False,disk=disk,reserve=disk['reserve_bytes'])
            if improved:(output/'best_checkpoint.json').write_text(json.dumps({'path':best_path,'validation_flow_loss':best}))
            event(event='checkpoint_saved',step=step,path=name,optimizer_payload_saved=False,save_seconds=saved['checkpoint_save_seconds'],cpu_staging_cleanup=saved['cpu_staging_cleanup'],free_disk_bytes=shutil.disk_usage(output).free)
    value=last_validation if last_validation_step==step else validate(runtime,sources,loss_weights,validation_deadline)
    improved=validation_improves(value,best)
    if improved:best=value['combined_flow_loss'];best_path=str(output/f'final_{step:06d}')
    final_name=f'final_{step:06d}'
    saved=save_checkpoint(runtime,optimizer,output/final_name,metadata(value),include_optimizer=True,disk=disk,reserve=disk['reserve_bytes'])
    if improved:(output/'best_checkpoint.json').write_text(json.dumps({'path':best_path,'validation_flow_loss':best}))
    (output/'last_checkpoint.json').write_text(json.dumps({'path':final_name}))
    selected=json.loads((output/'best_checkpoint.json').read_text())['path']
    event(event='training_stopped',step=step,inherited_round_updates=inherited,cumulative_round_updates=inherited+step,reason='max_steps' if step>=args.max_steps else 'time_budget',final_checkpoint=final_name,best_validation_loss=best,baseline_validation=baseline,validation_improved_over_warmstart=best<baseline['combined_flow_loss'],training_benefit_conclusion='validation improvement only; closed-loop benefit requires comparison' if best<baseline['combined_flow_loss'] else 'further recovery-update benefit unestablished; retained prior candidates remain eligible',selected_checkpoint=selected,cumulative_time_sampling_draw_counts=draw_counts,final_validation=value,final_save_seconds=saved['checkpoint_save_seconds'],all_checkpoints_retained=True,selected_checkpoint_exists=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('warmstart','norm-stats','data-config','loss-weights','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--hours',type=float,default=4.)
    parser.add_argument('--max-steps',type=int,default=2000)
    parser.add_argument('--checkpoint-every',type=int,default=200)
    parser.add_argument('--disk-reserve-gib',type=float,default=32.)
    parser.add_argument('--save-reserve-seconds',type=float,default=300.)
    parser.add_argument('--validation-reserve-seconds',type=float,default=180.)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--recovery-failed-budget',help='Closed ledger containing the interrupted parent run; never optimizer resume')
    parser.add_argument('--candidate-training-dir',action='append',default=[],help='Additional explicit candidate run; immediate parent run is included automatically')
    parser.add_argument('--genesis-checkpoint',default=str(Path(__file__).resolve().parents[2]/'.scratch/pi05_g1/training_run02/step_001200'))
    args=parser.parse_args()
    if not 0<args.hours<=8 or args.max_steps<1 or args.checkpoint_every<1 or min(args.disk_reserve_gib,args.save_reserve_seconds,args.validation_reserve_seconds)<0:parser.error('Positive bounded hours/steps/checkpoint interval and nonnegative reserves required')
    train(args)


if __name__=='__main__':main()
