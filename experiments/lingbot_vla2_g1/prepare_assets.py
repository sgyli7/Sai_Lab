"""Download pinned public assets; sequential F32->BF16 conversion on CPU only."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import requests

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'.scratch/lingbot_vla2_g1'
MODEL='robbyant/lingbot-vla-v2-6b'
REVISION='11c703bf6a5c1f45b3b69168482da11fdbba53d7'
QWEN='Qwen/Qwen3-VL-4B-Instruct'
QREV='ebb281ec70b05090aa6165b016eac8ec08e71b17'
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.incomplete')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def guard():
    pre=json.loads((ROOT/'.scratch/pi05_g1/round03_mobile/training_cpu_preflight_no_optimizer_v2.json').read_text())
    peer=pre['disk_estimate']['required_free_bytes']
    used=sum(p.stat().st_size for p in SCRATCH.rglob('*') if p.is_file() and not p.is_symlink() and '/venv/' not in str(p))
    remaining=max(0,25*GIB-used)
    required=peer+8*GIB+remaining
    free=shutil.disk_usage(SCRATCH).free
    if free<required:raise RuntimeError(f'Disk reservation: free={free}, required={required}; no artifact removed')
    return {'free_bytes':free,'required_bytes':required,'own_non_environment_bytes':used,'own_total_cap_gib':25,'peer_teacher_video_reserve_gib':8}


def download(repo,revision,name,dest,expected_size=None,expected_sha=None):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if expected_size is not None and dest.stat().st_size!=expected_size:raise ValueError('Existing file size mismatch: '+str(dest))
        if expected_sha and sha(dest)!=expected_sha:raise ValueError('Existing hash mismatch: '+str(dest))
        return
    if expected_size is not None and expected_size>1024**3:
        os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
        os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS']='4'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        from huggingface_hub import hf_hub_download
        fetched=Path(hf_hub_download(repo_id=repo,revision=revision,filename=name,local_dir=dest.parent))
        if fetched.resolve()!=dest.resolve():raise ValueError('Unexpected local HF destination')
        if fetched.stat().st_size!=expected_size or (expected_sha and sha(fetched)!=expected_sha):raise ValueError('HF download identity mismatch')
        prior=dest.with_suffix(dest.suffix+'.download')
        if prior.exists():prior.unlink()  # Superseded partial created by this downloader.
        return
    temp=dest.with_suffix(dest.suffix+'.download')
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    offset=temp.stat().st_size if temp.exists() else 0
    response=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120))
    response.raise_for_status()
    if offset and response.status_code!=206:offset=0
    with temp.open('ab' if offset else 'wb') as output:
        for chunk in response.iter_content(4*1024**2):
            if chunk:output.write(chunk)
    if expected_size is not None and temp.stat().st_size!=expected_size:raise ValueError('Incomplete download: '+name)
    if expected_sha and sha(temp)!=expected_sha:raise ValueError('Download SHA mismatch: '+name)
    os.replace(temp,dest)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    metadata=requests.get(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}',params={'blobs':'true'},timeout=60).json()
    if metadata.get('sha')!=REVISION:raise ValueError('Pinned model revision mismatch')
    write(SCRATCH/'remote_model_manifest.json',metadata)
    members={r['rfilename']:r for r in metadata['siblings']}
    model=SCRATCH/'base_bf16';model.mkdir(exist_ok=True)
    for name in ['config.json','model.safetensors.index.json']:
        destination=(SCRATCH/'remote_weight_index.json') if name.endswith('index.json') else model/name
        item=members[name];download(MODEL,REVISION,name,destination,item.get('size'),item.get('lfs',{}).get('sha256'))
    index=json.loads((SCRATCH/'remote_weight_index.json').read_text())
    if not (model/'model.safetensors.index.json').exists():write(model/'model.safetensors.index.json',index)
    shards=sorted(set(index['weight_map'].values()))
    receipt_path=SCRATCH/'bf16_conversion_receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'schema':'lingbot_vla2_bf16_conversion_v1','status':'in_progress','model':MODEL,'revision':REVISION,'source_dtype':'F32','target_dtype':'BF16','rounding':'torch.to(bfloat16)','shards':[],'gpu_used':False}
    for name in shards:
        existing=next((s for s in receipt['shards'] if s['name']==name),None)
        dest=model/name
        if existing:
            if sha(dest)!=existing['output_sha256']:raise ValueError('Converted shard changed: '+name)
            continue
        probe=guard();print(json.dumps({'event':'download_shard_start','name':name,'disk':probe}),flush=True)
        item=members[name];raw=SCRATCH/'download_staging'/name
        download(MODEL,REVISION,name,raw,item['size'],item.get('lfs',{}).get('sha256'))
        input_sha=sha(raw);tensors={};counts={}
        with safe_open(raw,framework='pt',device='cpu') as source:
            for key in source.keys():
                value=source.get_tensor(key)
                counts[str(value.dtype)]=counts.get(str(value.dtype),0)+value.numel()
                tensors[key]=value.to(torch.bfloat16) if value.is_floating_point() else value.clone()
        temporary=dest.with_suffix('.safetensors.incomplete')
        save_file(tensors,str(temporary),metadata={'format':'pt','source_revision':REVISION,'conversion':'F32_to_BF16'})
        shapes={key:list(t.shape) for key,t in tensors.items()}
        del tensors,value;gc.collect()
        os.replace(temporary,dest)
        record={'name':name,'input_bytes':raw.stat().st_size,'input_sha256':input_sha,'output_bytes':dest.stat().st_size,'output_sha256':sha(dest),'input_dtype_elements':counts,'tensor_shapes':shapes,'raw_staging_retention':'removed only after verified converted output; public source is pinned and redownloadable'}
        receipt['shards'].append(record);write(receipt_path,receipt)
        for clean in (raw,dest):
            with clean.open('rb') as stream:
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        raw.unlink()  # Only this downloader's reproducible staging shard.
        print(json.dumps({'event':'shard_converted','name':name,'output_bytes':record['output_bytes']}),flush=True)
    index['metadata']['total_size']=sum(s['output_bytes'] for s in receipt['shards'])
    write(model/'model.safetensors.index.json',index)
    qwen=SCRATCH/'qwen_processor';qwen.mkdir(exist_ok=True)
    for name in ['config.json','tokenizer_config.json','tokenizer.json','preprocessor_config.json','video_preprocessor_config.json','chat_template.json','vocab.json','merges.txt']:
        download(QWEN,QREV,name,qwen/name)
    receipt.update(status='completed',total_output_bytes=sum(s['output_bytes'] for s in receipt['shards']),qwen_revision=QREV,processor_files=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(qwen.iterdir()) if p.is_file()])
    write(receipt_path,receipt)
    source=SCRATCH/'upstream'
    identity={'remote':subprocess.check_output(['git','-C',str(source),'remote','get-url','origin'],text=True).strip(),'commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'gpu_used':False}
    write(SCRATCH/'upstream_identity.json',identity)
    print(json.dumps({'event':'assets_completed','bytes':receipt['total_output_bytes'],'revision':REVISION}),flush=True)


if __name__=='__main__':main()
