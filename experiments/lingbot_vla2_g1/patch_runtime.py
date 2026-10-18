"""Build an auditable isolated source overlay; upstream checkout stays pristine."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.scratch/lingbot_vla2_g1'
COMMIT='be969b8fd117fb70550c5d4bf4bc328211b5b1b6'


def prepare():
    source=BASE/'upstream';target=BASE/'runtime_source'
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMIT:raise ValueError('Unexpected upstream commit '+actual)
    if not target.exists():
        shutil.copytree(source/'lingbotvla',target/'lingbotvla',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path='lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
    original=(source/path).read_text();patched=original
    if original.count('= "flash_attention_2"')!=3:raise ValueError('Attention constructor changed')
    patched=patched.replace('= "flash_attention_2"','= "sdpa"')
    old='        align_metrics = {}\n        if self.config.align_params != {}:'
    new='        align_metrics = {}\n        if self.config.align_params != {} and getattr(self.config, "auxiliary_supervision", True):'
    if patched.count(old)!=1:raise ValueError('Auxiliary gate anchor changed')
    patched=patched.replace(old,new)
    anchor='    def get_attention_interface(self):\n'
    patched=patched.replace(anchor,anchor+'        if self.config.attention_implementation == "sdpa":\n            from experiments.lingbot_vla2_g1.backends import sdpa_attention\n            return sdpa_attention\n')
    (target/path).write_text(patched)
    # Avoid eager import of unrelated language-loss and sequence-parallel
    # infrastructure; the action runner imports its kernels explicitly.
    op='lingbotvla/ops/__init__.py';ops=(source/op).read_text()
    for line in ['from .attention import flash_attention_forward','from .fused_moe import fused_moe_forward','from .loss import causallm_loss_function']:
        ops=ops.replace(line,'# Local single-rank runner loads needed operators explicitly.')
    (target/op).write_text(ops)
    records=[]
    for rel in [path,op]:
        a=(source/rel).read_bytes();b=(target/rel).read_bytes()
        records.append({'path':rel,'upstream_sha256':hashlib.sha256(a).hexdigest(),'runtime_sha256':hashlib.sha256(b).hexdigest()})
    receipt={'schema':'lingbot_vla2_local_runtime_overlay_v1','upstream_commit':actual,'patches':records,'learnable_parameter_layout_changed':False,'auxiliary_supervision_gate':'disabled only by explicit runtime config; native query/head construction unchanged','attention':'SDPA, BF16 on CUDA; original boolean mask and GQA retained','moe':'3D stored experts with dense torch matmul; no EP/custom group GEMM in local runner'}
    (BASE/'runtime_overlay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return target


if __name__=='__main__':print(prepare())
