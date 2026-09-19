"""Exact frozen ONNX anchor plus a differentiable policy increment.

For full finetuning the increment is MLP(theta)-MLP(theta_initial). This retains
the original runtime's finite-precision forward at initialization, while giving
ordinary full-MLP gradients. It avoids changing the normalization epsilon or
loosening parity to conceal cross-library GEMM roundoff. Export retains the
original ONNX graph and adds the learned increment; no runtime Python is needed.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from sim2sim.train.onnx_import import parse_mlp_onnx, _sample_realistic_obs61

class NativeAnchor:
    def __init__(self, path):
        self.path=Path(path)
        self.raw=self.path.read_bytes()
        so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
        self.session=ort.InferenceSession(self.raw,sess_options=so,providers=["CPUExecutionProvider"])
        self.input=self.session.get_inputs()[0].name
        self.sha256=hashlib.sha256(self.raw).hexdigest()
        from sim2sim.policy_time import time_input_seconds,has_heading_input
        meta=self.session.get_modelmeta().custom_metadata_map
        self.time_input_s=time_input_seconds(meta);self.heading_input=has_heading_input(meta)
        from sim2sim.policy_memory import has_yaw_memory
        self.yaw_memory_input=has_yaw_memory(meta)
        from sim2sim.policy_state import state_input
        self.state_input=state_input(meta)
        from sim2sim.policy_task_state import task_input
        self.task_input=task_input(meta)
        self.obs_dim=61+(7 if self.task_input else 0)
        if self.session.get_inputs()[0].shape != [1,self.obs_dim]:
            raise ValueError('Policy observation shape disagrees with its declared contract')

    def __call__(self, obs):
        obs=np.asarray(obs,np.float32).reshape(-1,self.obs_dim)
        return np.concatenate([self.session.run(None,{self.input:o[None]})[0] for o in obs],axis=0)

class Increment(nn.Module):
    def __init__(self, source, variant="anchor", bound=.2,time_gate=None,command_gate='',input_dim=61,mask_task_state=False,action_basis='',mask_motion_state=False):
        super().__init__()
        rec=parse_mlp_onnx(source)
        self.variant=variant
        self.bound=float(bound)
        self.time_gate=time_gate
        if command_gate not in ('','negative_throttle'):raise ValueError('Unknown command gate')
        self.command_gate=command_gate
        self.input_dim=input_dim
        self.mask_motion_state=mask_motion_state
        if action_basis not in ('','brake_sagittal_v1'):
            raise ValueError('Unknown learned action basis: '+action_basis)
        if action_basis and (input_dim!=68 or variant!='residual' or command_gate!='negative_throttle'):
            raise ValueError('Brake action basis requires the 68D brake-gated residual')
        self.action_basis=action_basis
        action_mask=torch.ones(14)
        if action_basis:
            action_mask.zero_();action_mask[[2,3,11,12]]=1.
        self.register_buffer('action_mask',action_mask)
        if input_dim not in (61,68) or (input_dim!=61 and variant!='residual'):
            raise ValueError('The 68D contract currently supports a residual policy only')
        self.register_buffer("mean",torch.from_numpy(np.r_[rec.mean.copy(),np.zeros(input_dim-61)].astype(rec.mean.dtype)))
        self.register_buffer("denominator",torch.from_numpy(np.r_[rec.std.copy(),np.ones(input_dim-61)].astype(rec.std.dtype)))
        mask=torch.ones(input_dim)
        if mask_motion_state:
            if variant!='residual':raise ValueError('Motion-state ablation requires a residual actor')
            mask[58:61]=0.
        if mask_task_state:
            if input_dim!=68:raise ValueError('Task-state ablation requires a declared 68D actor')
            mask[61:]=0.
        self.register_buffer('observation_mask',mask)
        if variant=="residual":
            self.net=nn.Sequential(nn.Linear(input_dim,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,14))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
            self.initial=None
            if input_dim==68:self.double()
        else:
            layers=[]
            for i,(w,b) in enumerate(rec.layers):
                layer=nn.Linear(w.shape[1],w.shape[0])
                with torch.no_grad():layer.weight.copy_(torch.from_numpy(w.copy()));layer.bias.copy_(torch.from_numpy(b.copy()))
                layers.append(layer)
                if i+1<len(rec.layers):layers.append(nn.ELU())
            self.net=nn.Sequential(*layers)
            self.initial=copy.deepcopy(self.net).requires_grad_(False)
            # Subtract nearby full-network outputs in float64. Otherwise two
            # independent float32 GEMMs can swamp a small learned increment.
            self.double()

    def forward(self, obs):
        x=((obs-self.mean)/self.denominator)*self.observation_mask
        if self.variant=="residual":delta=self.bound*torch.tanh(self.net(x.clamp(-20,20)))
        else:delta=(self.net(x)-self.initial(x)).to(obs.dtype)
        if self.time_gate is not None:
            start,end,seconds=self.time_gate
            gate=((obs[:,48:49]*seconds-start)/(end-start)).clamp(0.,1.)
            delta=delta*gate
        if self.command_gate:delta=delta*self.command_weight(obs)
        delta=delta*self.action_mask
        return delta.to(obs.dtype)

    def command_weight(self,obs):
        return (-obs[:,48:49]/.05).clamp(0.,1.)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # Pre-task-state checkpoints had no identity observation mask. Preserve
        # their resume path; never infer missing learned parameters or 68D masks.
        key=prefix+'observation_mask'
        if self.input_dim==61 and not self.mask_motion_state and key not in state_dict:
            state_dict[key]=torch.ones_like(self.observation_mask)
        key=prefix+'action_mask'
        if not self.action_basis and key not in state_dict:
            state_dict[key]=torch.ones_like(self.action_mask)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

class Policy(nn.Module):
    def __init__(self, source, variant="anchor", std=.03, bound=.2, template=None,time_gate=None,command_gate='',mask_task_state=False,action_basis='',mask_motion_state=False):
        super().__init__()
        self.anchor=NativeAnchor(source)
        if time_gate is not None:
            start,end=map(float,time_gate)
            if not 0<=start<end<=self.anchor.time_input_s:raise ValueError("Time gate requires a declared time-input actor")
            time_gate=(start,end,self.anchor.time_input_s)
        if mask_motion_state and not self.anchor.state_input:
            raise ValueError('Motion-state ablation requires a declared state-input anchor')
        self.delta=Increment(template or source,variant,bound,time_gate,command_gate,self.anchor.obs_dim,mask_task_state,action_basis,mask_motion_state)
        self.mask_task_state=mask_task_state
        self.mask_motion_state=mask_motion_state
        if self.anchor.time_input_s:
            self.delta.mean[48]=0.;self.delta.denominator[48]=1.
        if self.anchor.heading_input:self.delta.mean[49:51]=0.;self.delta.denominator[49:51]=1.
        if self.anchor.yaw_memory_input:self.delta.mean[55]=0.;self.delta.denominator[55]=1.
        if self.anchor.state_input:
            if variant!='residual':raise ValueError('State input requires a residual increment')
            self.delta.mean[58:61]=torch.tensor([0.,0.,.115])
            self.delta.denominator[58:61]=torch.tensor([.6,.6,.05])
        if self.anchor.task_input:
            self.delta.denominator[61:]=torch.tensor([2.,1.,.3,.3,.3,1.,1.])
        self.log_std=nn.Parameter(torch.full((14,),float(np.log(std))))
        self.variant=variant

    def anchor_values(self, obs):
        if isinstance(obs,torch.Tensor):obs=obs.detach().cpu().numpy()
        return torch.from_numpy(self.anchor(obs))

    def forward(self,obs,anchor=None):
        if anchor is None:anchor=self.anchor_values(obs).to(obs.device)
        return anchor+self.delta(obs)

    def distribution(self,obs,anchor=None):
        mean=self(obs,anchor)
        std=self.log_std.clamp(np.log(.005),np.log(.3)).exp()
        if self.delta.command_gate:std=std*self.delta.command_weight(obs)*self.delta.action_mask.to(std.dtype)+1e-5
        return torch.distributions.Normal(mean,std)

    @torch.no_grad()
    def predict(self,obs):
        x=torch.as_tensor(np.asarray(obs,np.float32).reshape(-1,self.anchor.obs_dim),device=self.log_std.device)
        return self(x).cpu().numpy()

class Critic(nn.Module):
    def __init__(self,source,extra_dim,time_input_s=0.,heading_input=False,yaw_memory_input=False,state_input='',input_dim=61):
        super().__init__()
        rec=parse_mlp_onnx(source)
        self.input_dim=input_dim
        self.register_buffer("mean",torch.from_numpy(np.r_[rec.mean.copy(),np.zeros(input_dim-61)].astype(rec.mean.dtype)))
        self.register_buffer("denominator",torch.from_numpy(np.r_[rec.std.copy(),np.ones(input_dim-61)].astype(rec.std.dtype)))
        if time_input_s:self.mean[48]=0.;self.denominator[48]=1.
        if heading_input:self.mean[49:51]=0.;self.denominator[49:51]=1.
        if yaw_memory_input:self.mean[55]=0.;self.denominator[55]=1.
        if state_input:
            self.mean[58:61]=torch.tensor([0.,0.,.115])
            self.denominator[58:61]=torch.tensor([.6,.6,.05])
        if input_dim==68:self.denominator[61:]=torch.tensor([2.,1.,.3,.3,.3,1.,1.])
        self.net=nn.Sequential(nn.Linear(input_dim+extra_dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1))

    def forward(self,obs):
        x=torch.cat([((obs[:,:self.input_dim]-self.mean)/self.denominator).clamp(-20,20),obs[:,self.input_dim:].clamp(-20,20)],dim=-1)
        return self.net(x).squeeze(-1)

def export_policy(policy,path):
    import onnx
    from onnx import helper,compose
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    delta_path=path.with_suffix(".delta.onnx")
    delta_actor=policy.delta if policy.log_std.device.type=='cpu' else copy.deepcopy(policy.delta).cpu()
    torch.onnx.export(delta_actor.eval(),(torch.zeros(1,policy.anchor.obs_dim),),str(delta_path),input_names=["obs"],output_names=["increment"],opset_version=18,dynamo=False)
    original=compose.add_prefix(onnx.load_model_from_string(policy.anchor.raw),"factory/")
    delta=compose.add_prefix(onnx.load(str(delta_path)),"adapt/")
    for model in (original,delta):
        old=model.graph.input[0].name
        for node in model.graph.node:
            for i,name in enumerate(node.input):
                if name==old:node.input[i]="obs"
    nodes=list(original.graph.node)+list(delta.graph.node)
    nodes.append(helper.make_node("Add",[original.graph.output[0].name,delta.graph.output[0].name],["actions"]))
    graph=helper.make_graph(nodes,"factory_plus_adaptation",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,policy.anchor.obs_dim])],[helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],initializer=list(original.graph.initializer)+list(delta.graph.initializer))
    result=helper.make_model(graph,opset_imports=[helper.make_opsetid("",18)])
    result.ir_version=min(original.ir_version,10)
    # Native anchors may themselves be previously adapted policies.
    # Replace current-stage fields, preserving unique inherited metadata.
    metadata={prop.key:prop.value for prop in original.metadata_props}
    metadata.update(sim2sim_factory_sha256=policy.anchor.sha256,sim2sim_adaptation=policy.variant)
    if policy.anchor.task_input:metadata['sim2sim_task_state_mask']=str(bool(policy.mask_task_state)).lower()
    if policy.anchor.state_input:metadata['sim2sim_motion_state_mask']=str(bool(policy.mask_motion_state)).lower()
    if policy.delta.time_gate is not None:
        import json
        metadata["sim2sim_increment_time_gate_s"]=json.dumps(policy.delta.time_gate)
    if policy.delta.command_gate:metadata['sim2sim_increment_command_gate']=policy.delta.command_gate
    metadata['sim2sim_increment_action_basis']=policy.delta.action_basis or 'all_actuators'
    if getattr(policy,"task_name",None):
        from .tasks import TASKS
        task=TASKS[policy.task_name]
        metadata.update(sim2sim_task=task.name,sim2sim_command_mode=task.mode,
                        sim2sim_period_s=str(task.period if task.mode=="phase" else 0))
        if policy.anchor.time_input_s:metadata["sim2sim_command_mode"]="one_shot_time"
        if getattr(policy,'roller_contract',False):metadata['sim2sim_command_mode']='roller_throttle_heading_error'
    for key,value in metadata.items():result.metadata_props.add(key=key,value=value)
    onnx.checker.check_model(result)
    onnx.save(result,str(path));delta_path.unlink()
    if getattr(policy,'task_name',None)=='walking':
        import json
        # This research actor controls whole trajectories including idle.
        # Make every intermediate export playable under the same contract.
        side=dict(action_scale=1.,sim2sim=dict(use_stand_policy=False),
                  research=dict(skill='walking',source_sha256=policy.anchor.sha256))
        path.with_suffix('.manifest.json').write_text(json.dumps(side,indent=2))
    return path

def parity(policy,exported=None,n=10000):
    rng=np.random.default_rng(0)
    reference=policy.anchor if exported is None else NativeAnchor(exported)
    result={}
    realistic=_sample_realistic_obs61(n,rng)
    if policy.anchor.obs_dim>61:realistic=np.pad(realistic,((0,0),(0,policy.anchor.obs_dim-61)))
    for label,obs in [("random",rng.standard_normal((n,policy.anchor.obs_dim),dtype=np.float32)),("realistic",realistic)]:
        with torch.no_grad():
            # The deployment contract is batch=1. Batched torch training can
            # have different summation roundoff; report it separately.
            batch=policy.predict(obs)
            single=np.concatenate([policy.predict(x[None]) for x in obs])
        ref=reference(obs)
        result[label+"_max_abs"]=float(np.max(np.abs(single-ref)))
        result[label+"_batch_max_abs"]=float(np.max(np.abs(batch-ref)))
    result["threshold"]=1e-5
    result["passed"]=all(result[k]<1e-5 for k in ["random_max_abs","realistic_max_abs"])
    return result
