"""Pinned pretrained LingBot V2, preserved native queries, local SDPA/MoE backends."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .patch_runtime import prepare,BASE
from .backends import dense_stored_experts
from .contract import pack_state,pack_actions,unpack_actions,ACTION_MASK,PROMPT
from .data import MeanStd,sha


def components():
    # Overlay creation is a CPU preparation step, never mutates raw upstream.
    overlay=BASE/'runtime_source'
    if not overlay.is_dir():prepare()
    sys.path.insert(0,str(overlay))
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert as experts
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    experts.apply_lingbot_qwen2_patch();apply_lingbot_qwen3_vl_patch()
    experts.Qwen2FusedExperts.forward=dense_stored_experts
    experts.robby_moe_forward=None
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAConfig,LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.module_utils import init_empty_weights,load_model_weights
    from transformers.modeling_utils import no_init_weights
    import yaml
    document=yaml.safe_load((BASE/'upstream/configs/vla/real_robot/real_robot.yaml').read_text())
    accepted=set(inspect.signature(LingbotVLAConfig.__init__).parameters)
    kwargs={k:v for k,v in document['train'].items() if k in accepted}
    # Plain PyYAML leaves e.g. "1e-4" as str, unlike the typed upstream parser.
    for name in ['sequence_wise_loss_coeff','router_z_loss_coeff','routed_scaling_factor','bias_update_speed']:
        if name in kwargs:kwargs[name]=float(kwargs[name])
    kwargs.update(tokenizer_path=str(BASE/'qwen_processor'),vlm_repo_id=str(BASE/'qwen_processor'),post_training=True,adanorm_time=True,moe_implementation='fused',attention_implementation='sdpa',vit_attn_implementation='sdpa',train_expert_only=True,freeze_vision_encoder=True,train_state_proj=True,use_compile=False)
    config=LingbotVLAV2Config(**kwargs)
    config.auxiliary_supervision=False
    return config,LingbotVlaV2Policy,init_empty_weights,load_model_weights,no_init_weights


def meta_model():
    config,policy,empty,loader,no_init=components()
    with empty(),no_init():model=policy(config).to(torch.bfloat16)
    return model,loader


def cpu_admission():
    model,_=meta_model()
    index=json.loads((BASE/'base_bf16/model.safetensors.index.json').read_text())['weight_map']
    names={k for k,_ in model.named_parameters()};buffers={k for k,_ in model.named_buffers()}
    missing=names-set(index);unexpected=set(index)-names-buffers
    receipt=json.loads((BASE/'bf16_conversion_receipt.json').read_text())
    shapes={name:shape for shard in receipt['shards'] for name,shape in shard['tensor_shapes'].items()}
    expected={**dict(model.named_parameters()),**dict(model.named_buffers())}
    shape_errors={name:{'model':list(expected[name].shape),'checkpoint':shape} for name,shape in shapes.items() if name in expected and list(expected[name].shape)!=shape}
    result={'schema':'lingbot_vla2_meta_admission_v1','gpu_used':False,'parameter_elements':sum(p.numel() for p in model.parameters()),'trainable_elements':sum(p.numel() for p in model.parameters() if p.requires_grad),'parameter_tensors':len(names),'checkpoint_entries':len(index),'missing':sorted(missing),'unexpected':sorted(unexpected),'native_depth_query_structure_preserved':model.model.use_depth_align,'future_video_query_structure_preserved':model.model.use_future_video,'auxiliary_supervision':False,'state_dim':model.config.max_state_dim,'action_dim':model.config.max_action_dim,'checked_tensor_shapes':len(shapes),'shape_errors':shape_errors,'assets_completed':receipt.get('status')=='completed'}
    (BASE/'meta_admission.json').write_text(json.dumps(result,indent=2)+'\n')
    if missing or unexpected or shape_errors:raise ValueError('Checkpoint layout mismatch; see meta_admission.json')
    return result


class Runtime:
    def __init__(self,training=False,delta=None,norm_stats=None):
        self.device=torch.device('cuda')
        if not torch.cuda.is_available():raise RuntimeError('GB10 CUDA required; no CPU model fallback')
        self.device_name=torch.cuda.get_device_name(0)
        if self.device_name!='NVIDIA GB10':raise RuntimeError('This experiment is pinned to the shared NVIDIA GB10')
        if json.loads((BASE/'bf16_conversion_receipt.json').read_text()).get('status')!='completed':raise RuntimeError('Pretrained asset conversion must complete before CUDA admission')
        start=time.monotonic();self.model,loader=meta_model()
        loader(self.model,str(BASE/'base_bf16'),init_device='cuda',post_training=True,adanorm_time=True)
        # UMA: release only our clean source-file cache after copying to CUDA.
        import os
        for shard in (BASE/'base_bf16').glob('*.safetensors'):
            with shard.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        self.norm_path=Path(norm_stats or BASE/'train_meanstd.json').absolute()
        self.normalizer=MeanStd(self.norm_path)
        if delta:
            from safetensors import safe_open
            metadata=json.loads((Path(delta)/'metadata.json').read_text())
            for key,path in [('base_receipt_sha256',BASE/'bf16_conversion_receipt.json'),('norm_sha256',self.norm_path),('contract_sha256',BASE/'contract.json')]:
                if metadata.get(key)!=sha(path):raise ValueError('Delta identity mismatch: '+key)
            if (Path(delta)/'trainable.safetensors').stat().st_size!=metadata['weights_bytes']:raise ValueError('Delta payload size changed')
            parameters=dict(self.model.named_parameters())
            with safe_open(str(Path(delta)/'trainable.safetensors'),framework='pt',device='cpu') as payload:
                keys=set(payload.keys());expected={n for n,p in parameters.items() if p.requires_grad}
                if keys!=expected:raise ValueError('Delta trainable parameter coverage mismatch')
                with torch.no_grad():
                    for name in payload.keys():parameters[name].copy_(payload.get_tensor(name).to(device=self.device,dtype=parameters[name].dtype))
            with (Path(delta)/'trainable.safetensors').open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        from transformers import AutoProcessor
        self.processor=AutoProcessor.from_pretrained(BASE/'qwen_processor',local_files_only=True)
        self.model.config.use_cache=not training
        if training:
            self.model.train()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:self.model.eval()
        self.load_seconds=time.monotonic()-start
        self.parameter_elements=sum(p.numel() for p in self.model.parameters())
        self.trainable_elements=sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def inputs(self,rgb,state32,actions32=None,frame_mask=None):
        from PIL import Image
        rgb=np.asarray(rgb,dtype=np.uint8)
        if rgb.shape!=(480,640,3):raise ValueError('Frozen head camera RGB must be480x640x3')
        image=Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR)
        processed=self.processor.image_processor(images=image,return_tensors='pt')
        pixels=processed['pixel_values'];grid=processed['image_grid_thw']
        # Existing camera only. Missing wrist images are masked, as upstream.
        images=torch.stack([pixels,torch.full_like(pixels,-1),torch.full_like(pixels,-1)])[None]
        state=self.normalizer.normalize(state32,'state');state[-1]=0
        prompt=self.processor.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],tokenize=False,add_generation_prompt=False)
        tokens=self.processor.tokenizer(prompt,padding='max_length',padding_side='right',max_length=72,truncation=True,return_tensors='pt')
        result={'images':images.to(self.device,dtype=torch.bfloat16),'img_masks':torch.tensor([[True,False,False]],device=self.device),'image_grid_thw':grid.repeat(3,1)[None].to(self.device),'lang_tokens':tokens['input_ids'].to(self.device),'lang_masks':tokens['attention_mask'].to(self.device,dtype=torch.bool),'state':torch.as_tensor(pack_state(state),device=self.device,dtype=torch.bfloat16)[None]}
        if actions32 is not None:
            result['actions']=torch.as_tensor(pack_actions(actions32),device=self.device,dtype=torch.bfloat16)[None]
            mask=np.asarray(frame_mask,dtype=bool)[:,None]&ACTION_MASK[None]
            result['joint_mask']=torch.as_tensor(mask,device=self.device,dtype=torch.float32)[None]
        return result

    def loss(self,rgb,state,normalized_actions,mask):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=self.model(**self.inputs(rgb,state,normalized_actions,mask))
        return result[0],result[1]

    def infer(self,rgb,state):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            value=self.model.sample_actions(**self.inputs(rgb,state))
        raw=unpack_actions(value[0].float().cpu().numpy())
        return self.normalizer.denormalize(raw,'actions')


if __name__=='__main__':
    print(json.dumps(cpu_admission(),indent=2))
