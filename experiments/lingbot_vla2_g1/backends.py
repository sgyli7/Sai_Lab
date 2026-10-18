"""Structure-preserving PyTorch backends for the single GB10 adaptation."""
import torch
from torch.nn import functional as F


def sdpa_attention(query,key,value,mask,compute_dtype=None):
    """Original [B,L,H,D] API and true=allowed mask, including empty rows."""
    original_dtype=query.dtype
    if compute_dtype is None:
        compute_dtype=torch.bfloat16 if query.is_cuda else query.dtype
    groups=query.shape[2]//key.shape[2]
    if groups*key.shape[2]!=query.shape[2]:raise ValueError('Nonintegral GQA')
    q=query.transpose(1,2).to(compute_dtype)
    k=key.repeat_interleave(groups,dim=2).transpose(1,2).to(compute_dtype)
    v=value.repeat_interleave(groups,dim=2).transpose(1,2).to(compute_dtype)
    empty=~mask.any(dim=-1)
    # Upstream eager assigns a constant finite negative logit to every key on
    # empty query rows, producing the uniform mean of V. Preserve that behavior.
    q=q.masked_fill(empty[:,None,:,None],0)
    allowed=(mask|empty[:,:,None])[:,None]
    result=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.,is_causal=False)
    return result.transpose(1,2).reshape(query.shape[0],query.shape[1],-1).to(original_dtype)


def dense_stored_experts(self,module,num_experts,routing_weights,selected_experts,hidden_states):
    """Same routed computation with original fused 3D parameter layout.

    Evaluating each bank densely avoids custom group-GEMM/EP kernels while
    retaining all checkpoint keys and gradients through expert/routing weights.
    """
    gate=F.silu(torch.matmul(hidden_states,self.gate_proj.transpose(1,2)))
    up=torch.matmul(hidden_states,self.up_proj.transpose(1,2))
    outputs=torch.bmm(gate*up,self.down_proj.transpose(1,2))
    weights=hidden_states.new_zeros((hidden_states.shape[0],num_experts))
    weights=weights.scatter_add(1,selected_experts,routing_weights.to(weights.dtype))
    return torch.einsum('etd,te->td',outputs,weights)
