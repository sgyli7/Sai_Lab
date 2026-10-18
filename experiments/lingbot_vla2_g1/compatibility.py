"""Bounded real-checkpoint CUDA forward/backward/update and inference probe."""
import argparse
import json
from pathlib import Path
import time
import torch
from experiments.pi05_g1.training import EpisodeDataset,MasterAdamW,memory_snapshot
from .data import SHARED,SPLIT,OWN,MeanStd
from .runtime import Runtime


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic();report={'schema':'lingbot_vla2_g1_compatibility_v1','status':'running','gpu_used':True,'test_opened':False}
    def event(name,**data):print(json.dumps({'event':name,'elapsed_seconds':time.monotonic()-started,**data},allow_nan=False),flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        normalizer=MeanStd(OWN/'train_meanstd.json')
        dataset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'train',normalizer)
        rgb,state,actions,mask=dataset.sample(0)
        runtime=Runtime(training=True);report['device_name']=runtime.device_name;event('model_loaded',device=runtime.device_name,parameters=runtime.parameter_elements,trainable=runtime.trainable_elements,**memory_snapshot(torch))
        parameters={name:value for name,value in runtime.model.named_parameters()}
        sampled={name:value.detach().flatten()[:256].clone() for name,value in parameters.items() if name in ['model.action_out_proj.weight','model.state_proj.weight','model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj.weight','model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_proj.weight']}
        optimizer=MasterAdamW([p for p in runtime.model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
        loss,flow=runtime.loss(rgb,state,actions,mask);event('forward_complete',loss=float(loss.detach()),flow_loss=float(flow.detach()),**memory_snapshot(torch))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forward loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in runtime.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        gradient_tensors=sum(p.grad is not None for p in runtime.model.parameters())
        torch.cuda.empty_cache()
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        changes={name:bool(torch.any(parameters[name].detach().flatten()[:256]!=value)) for name,value in sampled.items()}
        event('optimizer_step_complete',gradient_norm=float(norm),gradient_tensors=gradient_tensors,sampled_parameter_changed=changes,**memory_snapshot(torch))
        if not any(changes[name] for name in changes if 'qwenvl.model.language_model' not in name):raise ValueError('No sampled trainable BF16 parameter changed')
        if any(changes[name] for name in changes if 'qwenvl.model.language_model' in name):raise ValueError('Frozen VLM changed')
        del optimizer;runtime.model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        runtime.model.gradient_checkpointing_disable();runtime.model.eval();runtime.model.config.use_cache=True
        tick=time.monotonic();prediction=runtime.infer(rgb,state);torch.cuda.synchronize();elapsed=time.monotonic()-tick
        if prediction.shape!=(50,32) or not torch.isfinite(torch.as_tensor(prediction)).all():raise ValueError('Invalid action chunk')
        report.update(status='completed',load_seconds=runtime.load_seconds,forward_loss=float(loss.detach()),flow_loss=float(flow.detach()),gradient_norm=float(norm),gradient_tensors=gradient_tensors,sampled_parameter_changed=changes,inference_seconds=elapsed,actions_shape=list(prediction.shape),parameter_elements=runtime.parameter_elements,trainable_elements=runtime.trainable_elements,head_rgb_shape=list(rgb.shape),model_variant='native queries preserved, auxiliary teachers/loss disabled, VLM frozen, BF16, SDPA, dense torch stored MoE',**memory_snapshot(torch))
        event('inference_complete',inference_seconds=elapsed,actions_shape=list(prediction.shape))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
