"""Version5 candidate: independently restorable gzip text; NOT adopted by default.

All source bytes, full60/audit gates and v3 link/copy rules remain unchanged.

Version2 readiness/audit gates and streaming content/inode checks are retained.

Plan schema pi05_g1_mobile_archive_plan_v1: sources[{category,path,destination,
immutable:false,reason}], required_files[{path,sha256}], reconstruction_commands
(nonempty strings), docker_image_id, quiescence, and, for final only,
training_result, benchmark, episode_audit. Destination must be a NEW directory
below /home/ethan/ProjectBackups/2026-10-17. No source changes/deletions/chmod,
Docker export, environment/cache traversal, inference, or fabricated artifacts.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import re
import gzip
import sys


from .audit_delivery_guard_v2 import validate as validate_v2_audit
LEGACY_SOURCE_PATH='/home/ethan/Projects/Sai_Lab/experiments/pi05_g1_mobile/archive_mobile.py'
LEGACY_SOURCE_SHA256='107612bc0e6ef803aa4cce583d68a7c2e4f053dec63026b61d044937fb9b73e3'

BACKUP_ROOT=Path('/home/ethan/ProjectBackups/2026-10-17')
CATEGORIES={'round03','source_snapshots','pi05_source','round02_source','mobile_source','queue_source',
            'round02_interrupted_history','official_base','native_n16','homie50','assets',
            'pi_tokenizer','native_tokenizer','openpi_source','native_source','provenance'}
EXCLUDED={'.venv','.git','__pycache__','.pytest_cache','venv','plot_venv','node_modules','.cache','cache','caches'}
PAYLOAD={'.safetensors','.pt','.onnx','.data','.bin','.usd','.usdc','.usda','.npy','.npz','.mp4','.png','.jpg','.stl','.obj','.model'}
RESERVE=24*1024**3
ARCHIVE_RESERVE_POLICY={
    'version':'pi05_g1_mobile_archive_reserve_v4_24gib',
    'authority':'explicit root decision',
    'root_decision_path':'/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round03_mobile/root_archive_reserve_policy_v4_decision.json',
    'root_decision_sha256':'224a86ec9fdce1b013ecd9eefe06990abea7e0d413b74e4f08509b0c5afd3ddb',
    'decision':'Lower internal archive reserve to preserve all evidence bytes without deletion or permission changes',
    'previous_reserve_bytes':32*1024**3,'archive_reserve_bytes':RESERVE,
    'scope':'Archive only; training32GiB and GPU host protection16GiB remain unchanged',
}

ORBAX_ROOT=Path('/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/pi05_base/params/ocdbt.process_0/d')
ORBAX_SHARDS={str(ORBAX_ROOT/name):size for name,size in {
    '8c5d7070ea57bdce2f0a19f95b8a21b4':3077149148,
    'deefd3c43390a50472cbcd317b0fff58':2393670849,
    '0832cad6c37f82d4eedd897dcbb8da9d':2240590484,
    '7bc9d3296d23a6fb83a6b3778ac6e964':2240315383,
    'ec484cf8f02dcf59e1892180f0862e40':1234292390,
    'b4349aaadb7dfa45c3a53fc67c04b8f6':1120156687,
}.items()}
ORBAX_REASON='Orbax模型数据'


def explicit_immutable_payloads(plan,destination_device):
    """Metadata-only, exact known data shards; never generic cache/metadata."""
    entries=plan.get('immutable_payload_files',[])
    if not isinstance(entries,list):raise ValueError('immutable_payload_files must be an explicit list')
    approved={}
    for item in entries:
        if set(item)!={'path','bytes','reason'}:raise ValueError('Payload entry requires exact path/bytes/reason')
        path=Path(item['path'])
        if not path.is_absolute() or str(path) not in ORBAX_SHARDS:raise ValueError('Unknown explicit Orbax payload')
        if any(part in EXCLUDED|{'conda_cache','usd_env','tools_env'} for part in path.parts):raise ValueError('Cache/environment cannot be an explicit payload')
        if not re.fullmatch('[0-9a-f]{32}',path.name) or path.parent.name!='d' or path.parent.parent.name!='ocdbt.process_0':raise ValueError('Only exact Orbax data shard layout allowed')
        if path.is_symlink():raise ValueError('Explicit shard cannot be a symlink')
        if str(path) in approved:raise ValueError('Duplicate explicit payload')
        info=path.stat()
        if type(item['bytes']) is not int or item['bytes']!=info.st_size or item['bytes']!=ORBAX_SHARDS[str(path)]:raise ValueError('Explicit payload size differs')
        if item['reason']!=ORBAX_REASON:raise ValueError('Explicit Orbax reason missing')
        if not stat.S_ISREG(info.st_mode) or info.st_uid==0 or info.st_uid!=os.getuid() or info.st_dev!=destination_device:raise ValueError('Explicit payload requires regular same-user nonroot same-filesystem source')
        covering=[source for source in plan.get('sources',[]) if path==Path(source['path']) or path.is_relative_to(Path(source['path']))]
        if not covering or any(source.get('immutable') is not True for source in covering):raise ValueError('Explicit payload source must be declared immutable')
        approved[str(path)]=item
    return approved


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def stable(st):return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns

def write_new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_hashes(mapping):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty input SHA binding required')
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('Evidence SHA changed: '+path)


def readiness(plan,phase):
    qpath=Path(plan['quiescence']).resolve();q=read(qpath)
    if q.get('schema')!='pi05_g1_mobile_owned_quiescence_v1' or q.get('passed') is not True or q.get('owned_closed') is not True:
        raise ValueError('Owned resources/accounting must be closed')
    if any(q.get(key) for key in ['active_owned_units','active_owned_container_ids','accounting_open_tokens']):raise ValueError('Owned work remains active')
    checked=dt.datetime.fromisoformat(q['completed_utc'])
    age=(dt.datetime.now(dt.timezone.utc)-checked).total_seconds()
    if not 0<=age<=300:raise ValueError('Fresh owned quiescence receipt (<=300s) required')
    check_hashes(q['evidence_files']);ledger=Path(q['ledger']).resolve()
    if sha(ledger)!=q['ledger_sha256']:raise ValueError('Ledger changed after quiescence')
    bindings={str(qpath):sha(qpath),str(ledger):sha(ledger),**q['evidence_files']}
    state={'owned_closed':True,'global_idle':q.get('global_idle'),'foreign_compute':q.get('non_display_compute_processes',[]),
           'scope':'Owned work only; external game/foreign workloads do not become owned busy. Point-in-time receipt, not a lock against future launches.'}
    if phase=='partial':return bindings,state
    training=read(plan['training_result']);summary=read(plan['benchmark']);audit=read(plan['episode_audit'])
    if training.get('schema')!='pi05_g1_mobile_training_result_v1' or training.get('status')!='completed' or training.get('training_stopped',{}).get('complete_optimizer_update_boundary') is not True:raise ValueError('Final archive requires truly completed training')
    if summary.get('schema')!='pi05_g1_mobile_benchmark_summary_v1' or not all(summary.get(k) is True for k in ['complete','evidence_qualified','formal_verdict_available']) or summary.get('errors'):raise ValueError('Final archive requires complete qualified benchmark')
    if len(summary.get('policies',{}))!=3 or len(summary.get('case_ids',[]))!=20 or len(set(summary['case_ids']))!=20:raise ValueError('Final archive requires primary3 x20')
    if any(p.get('completed_evidence_cases')!=20 for p in summary['policies'].values()):raise ValueError('Incomplete policy evidence')
    episodes=summary.get('episodes',[])
    pairs={(r['policy'],r['case_id']) for r in episodes}
    expected={(p,c) for p in summary['policies'] for c in summary['case_ids']}
    if len(episodes)!=60 or pairs!=expected:raise ValueError('Benchmark episode roster is not exact60')
    if audit.get('schema')!='pi05_g1_mobile_episode_audits_v2' or audit.get('passed') is not True:raise ValueError('Independent episode audit failed')
    audited=audit.get('episodes',[]);paths={str(Path(r['result_path']).resolve()) for r in episodes}
    if len(audited)!=60 or {str(Path(r['result_path']).resolve()) for r in audited}!=paths:raise ValueError('Independent audit roster differs from benchmark60')
    check_hashes(summary['inputs_sha256']);bindings.update(summary['inputs_sha256'])
    for row in audited:
        validate_v2_audit(row)
        if row.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v2' or row.get('passed') is not True or row.get('anomalies') or row.get('autonomous_policy_trial') is not True:raise ValueError('Invalid independent episode audit')
        check_hashes(row['inputs_sha256']);bindings.update(row['inputs_sha256'])
        result_path=str(Path(row['result_path']).resolve())
        if result_path not in row['inputs_sha256']:raise ValueError('Audit does not bind actual result')
        unsigned={k:v for k,v in row.items() if k!='audit_sha256'}
        expected_sha=hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if row.get('audit_sha256')!=expected_sha:raise ValueError('Independent audit self SHA mismatch')
    for key in ['training_result','benchmark','episode_audit']:bindings[str(Path(plan[key]).resolve())]=sha(plan[key])
    state['final_evidence_qualified']=True
    return bindings,state


def enumerate_sources(plan,destination_device):
    sources=plan.get('sources',[])
    if not CATEGORIES<={s.get('category') for s in sources}:raise ValueError('Missing required archive coverage categories')
    if not plan.get('reconstruction_commands') or not all(isinstance(c,str) and c.strip() for c in plan['reconstruction_commands']):raise ValueError('Explicit Docker/environment reconstruction commands required')
    if not str(plan.get('docker_image_id','')).startswith('sha256:'):raise ValueError('Frozen Docker image ID required; no image export performed')
    explicit_payloads=explicit_immutable_payloads(plan,destination_device)
    records=[];omitted=[];destinations=set();roots=[]
    for source in sources:
        path=Path(source['path'])
        if not path.is_absolute() or not path.exists():raise ValueError('Missing/nonabsolute requested artifact: '+str(path))
        rel=Path(source['destination'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Unsafe archive destination')
        if any(rel==old or rel.is_relative_to(old) or old.is_relative_to(rel) for old in roots):raise ValueError('Overlapping archive destinations')
        roots.append(rel)
        if type(source.get('immutable',False)) is not bool:raise ValueError('immutable must be explicit boolean')
        if source.get('immutable') and not source.get('reason'):raise ValueError('Permanent immutable declaration needs reason')
        entries=[path]
        while entries:
            current=entries.pop();relative=current.relative_to(path) if current!=path else Path()
            target=rel/relative
            if any(part in EXCLUDED for part in relative.parts):omitted.append({'source':str(current),'reason':'environment/cache/git excluded'});continue
            if current.is_symlink() and current.is_dir():raise ValueError('Directory symlink requires explicit resolved source entry: '+str(current))
            if current.is_dir():entries.extend(sorted(current.iterdir(),reverse=True));records.append({'source':current,'target':target,'kind':'directory','category':source['category']});continue
            info=current.stat()
            if not stat.S_ISREG(info.st_mode):raise ValueError('Only regular files/directories supported: '+str(current))
            if str(target) in destinations:raise ValueError('Duplicate destination')
            destinations.add(str(target))
            eligible=bool(source.get('immutable') and (current.suffix.lower() in PAYLOAD or str(current) in explicit_payloads) and info.st_uid!=0 and info.st_uid==os.getuid() and info.st_dev==destination_device)
            records.append({'source':current,'target':target,'kind':'file','category':source['category'],'info':info,'link_eligible':eligible,'declared_immutable':source.get('immutable',False),'explicit_immutable_payload':explicit_payloads.get(str(current)),'symlink_target':str(current.resolve()) if current.is_symlink() else None})
    populated={r['category'] for r in records if r['kind']=='file'}
    if not CATEGORIES<=populated:raise ValueError('Coverage category contains no retained files')
    return records,omitted



TEXT_SUFFIXES={'.json','.jsonl','.log','.txt','.md','.csv'}
TEXT_MIN_BYTES=65536
CHUNK=1024**2
CONTROL_HEADROOM=1024**2
TEXT_GZIP_POLICY={'version':'pi05_g1_mobile_text_gzip_v5','adopted':False,'external_root_decision_required_for_final':True,'level':6,'mtime':0,'source_bytes_modified':False,'ratio_guaranteed':False}


def strict_text_stat(st):
    return stable(st)+(st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode),st.st_nlink,st.st_ctime_ns)


def compressed_text(row):
    return row['kind']=='file' and not row['link_eligible'] and row['symlink_target'] is None and not any(p.is_symlink() for p in [row['source'],*row['source'].parents]) and row['source'].suffix.lower() in TEXT_SUFFIXES and row['info'].st_size>=TEXT_MIN_BYTES


class FloorWriter:
    """Every actual header/body/footer write checks floor and future plaincopies.

    Checks are point in time; foreign filesystem consumption cannot be locked.
    Small headroom is retained for a typed receipt after a normal floor refusal.
    """
    def __init__(self,stream,root,remaining,headroom=CONTROL_HEADROOM):self.stream=stream;self.root=root;self.remaining=remaining;self.headroom=headroom
    def write(self,data):
        required=RESERVE+self.remaining()+len(data)+4096+self.headroom
        if shutil.disk_usage(self.root).free<required:raise OSError('24GiB floor plus remaining mandatory plain copies would be consumed')
        result=self.stream.write(data)
        if result!=len(data):raise OSError('Short archive write')
        return result
    def flush(self):
        if shutil.disk_usage(self.root).free<RESERVE+self.remaining()+self.headroom:raise OSError('24GiB floor during flush')
        return self.stream.flush()
    def tell(self):return self.stream.tell()


def guarded_json(path,value,root,remaining=lambda:0,headroom=CONTROL_HEADROOM):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with Path(path).open('xb',buffering=0) as stream:
        dst=FloorWriter(stream,root,remaining,headroom);dst.write(data);dst.flush();os.fsync(stream.fileno())


def roundtrip_sha(path,expected_bytes):
    h=hashlib.sha256();count=0
    with gzip.open(path,'rb') as stream:
        while True:
            block=stream.read(min(CHUNK,expected_bytes-count+1))
            if not block:break
            count+=len(block)
            if count>expected_bytes:raise ValueError('Gzip exceeds original logical byte bound')
            h.update(block)
    if count!=expected_bytes:raise ValueError('Gzip original size mismatch')
    return h.hexdigest()


def gzip_text(source,target,info,root,remaining):
    if source.is_symlink() or strict_text_stat(source.stat())!=strict_text_stat(info):raise ValueError('Compression source identity changed')
    logical=hashlib.sha256();count=0
    with source.open('rb') as src,target.open('xb',buffering=0) as raw:
        if strict_text_stat(os.fstat(src.fileno()))!=strict_text_stat(info):raise ValueError('Opened compression source differs')
        sink=FloorWriter(raw,root,remaining)
        with gzip.GzipFile(filename='',mode='wb',compresslevel=6,fileobj=sink,mtime=0) as zipped:
            while True:
                block=src.read(CHUNK)
                if not block:break
                logical.update(block);count+=len(block);zipped.write(block)
        sink.flush();os.fsync(raw.fileno())
        if strict_text_stat(os.fstat(src.fileno()))!=strict_text_stat(info):raise ValueError('Compression source mutated during read')
    if count!=info.st_size or strict_text_stat(source.stat())!=strict_text_stat(info):raise ValueError('Compression source mutated')
    original_sha=logical.hexdigest();restored_sha=roundtrip_sha(target,info.st_size)
    if restored_sha!=original_sha:raise ValueError('Gzip roundtrip logical SHA mismatch')
    return original_sha,sha(target),restored_sha


def final_gzip_adoption(plan,phase):
    if phase!='final':return {'adopted':False,'scope':'partial candidate archive only'}
    item=plan.get('text_gzip_decision',{})
    if set(item)!={'path','sha256'}:raise ValueError('Final V5 needs explicit Root text gzip decision path/SHA')
    path=Path(item['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2 or sha(path)!=item['sha256']:raise ValueError('Root text gzip decision changed')
    decision=read(path)
    if decision.get('schema')!='pi05_g1_mobile_archive_text_gzip_v5_decision_v1' or decision.get('root_reviewed') is not True or decision.get('adopted') is not True or decision.get('tool_sha256')!=sha(__file__) or decision.get('reserve_bytes')!=RESERVE:raise ValueError('V5 policy not adopted for this exact source/floor')
    return {'adopted':True,'path':str(path),'sha256':item['sha256']}


def archive(plan_path,output,phase):
    if phase not in ['partial','final']:raise ValueError('Explicit partial/final phase required')
    output=Path(output).resolve();plan_path=Path(plan_path).resolve()
    if output.exists() or output.is_symlink():raise FileExistsError('Archive destination must be NEW; never overwrite')
    if not output.is_relative_to(BACKUP_ROOT.resolve()) or output==BACKUP_ROOT.resolve():raise ValueError('Archive must be a new child of dated ProjectBackups root')
    plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_mobile_archive_plan_v1':raise ValueError('Unsupported archive plan')
    adoption=final_gzip_adoption(plan,phase)
    bindings,state=readiness(plan,phase);bindings[str(plan_path)]=sha(plan_path)
    if adoption.get('adopted'):bindings[adoption['path']]=adoption['sha256']
    if not plan.get('required_files'):raise ValueError('Explicit admitted dependency/retention closure is required')
    parent=output.parent
    while not parent.exists():parent=parent.parent
    records,omitted=enumerate_sources(plan,parent.stat().st_dev)
    for row in records:
        if output.is_relative_to(row['source'].resolve()):raise ValueError('Destination cannot be inside a source tree')
    available=shutil.disk_usage(parent).free
    plain_bytes=sum(r['info'].st_size for r in records if r['kind']=='file' and not r['link_eligible'] and not compressed_text(r))
    text_bytes=sum(r['info'].st_size for r in records if compressed_text(r));all_bytes=sum(r['info'].st_size for r in records if r['kind']=='file')
    control_extra=len(json.dumps({'input_bindings':bindings,'readiness':state,'omitted':omitted},indent=2).encode())+65536
    if available<plain_bytes+RESERVE+CONTROL_HEADROOM+control_extra:raise OSError('Insufficient disk for mandatory plain copies, typedreceipt metadata and24GiB reserve')
    resolved_files={str(r['source'].resolve()) for r in records if r['kind']=='file'}
    for path in bindings:
        if path!=str(plan_path) and str(Path(path).resolve()) not in resolved_files:raise ValueError('Gate/evidence input missing from retained source roster: '+path)
    for item in plan['required_files']:
        if str(Path(item['path']).resolve()) not in resolved_files:raise ValueError('Dependency closure file absent from planned trees: '+item['path'])
    output.mkdir(parents=True,exist_ok=False);remaining=[plain_bytes+control_extra]
    receipt={'schema':'pi05_g1_mobile_archive_receipt_v5','phase':phase,'status':'in_progress','archive_complete':False,'experiment_complete_claim':phase=='final','started_utc':now(),'plan_path':str(plan_path),'input_bindings':bindings,'readiness':state,'tool_sha256':sha(__file__),'legacy_source_path':LEGACY_SOURCE_PATH,'legacy_source_sha256':LEGACY_SOURCE_SHA256,'audit_schema':'pi05_g1_mobile_episode_audits_v2','archive_reserve_policy':dict(ARCHIVE_RESERVE_POLICY),'text_gzip_policy':dict(TEXT_GZIP_POLICY),'text_gzip_adoption':adoption,'disk_preflight':{'available_bytes':available,'all_logical_bytes':all_bytes,'mandatory_plain_copy_bytes':plain_bytes,'uncompressed_eligible_text_bytes':text_bytes,'compressed_bytes_unknown_before_streaming':True,'reserve_bytes':RESERVE,'control_headroom_bytes':CONTROL_HEADROOM,'additional_typed_receipt_reserved_bytes':control_extra,'ratio_guaranteed':False},'omitted':omitted,'errors':[]}
    inventory=output/'inventory.jsonl';copied={};count=0;error=None
    try:
        guarded_json(output/'archive_plan.json',plan,output,lambda:remaining[0])
        for item in plan['required_files']:
            if sha(item['path'])!=item['sha256']:raise ValueError('Pinned closure SHA mismatch: '+item['path'])
        with inventory.open('xb',buffering=0) as raw_index:
            index=FloorWriter(raw_index,output,lambda:remaining[0])
            for row in records:
                source=row['source'];logical=Path('retained')/row['target'];target=output/logical
                if row['kind']=='directory':
                    target.mkdir(parents=True,exist_ok=True);entry={'source_path':str(source),'relative_path':str(logical),'logical_relative_path':str(logical),'type':'directory','category':row['category']}
                else:
                    info=row['info']
                    if stable(source.stat())!=stable(info):raise ValueError('Source changed after preflight: '+str(source))
                    target.parent.mkdir(parents=True,exist_ok=True);method='copy';stored={}
                    if compressed_text(row):
                        target=output/'text_gzip'/(hashlib.sha256(str(logical).encode()).hexdigest()+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
                        digest,stored_sha,roundtrip=gzip_text(source,target,info,output,lambda:remaining[0]);method='gzip'
                        stored={'stored_relative_path':str(target.relative_to(output)),'stored_bytes':target.stat().st_size,'stored_sha256':stored_sha,'roundtrip_original_sha256':roundtrip,'gzip_level':6,'gzip_mtime':0,'source_strict_stat_before':list(strict_text_stat(info)),'source_strict_stat_after':list(strict_text_stat(source.stat()))}
                    else:
                        digest=sha(source)
                        if row['link_eligible']:
                            try:os.link(source,target);method='hardlink'
                            except OSError:pass
                        if method=='copy':
                            if not row['link_eligible']:remaining[0]-=info.st_size
                            if shutil.disk_usage(target.parent).free<info.st_size+remaining[0]+RESERVE+CONTROL_HEADROOM:raise OSError('Reserve would be consumed during copy')
                            with source.open('rb') as src,target.open('xb',buffering=0) as dst:
                                sink=FloorWriter(dst,output,lambda:remaining[0])
                                for block in iter(lambda:src.read(CHUNK),b''):sink.write(block)
                                sink.flush();os.fsync(dst.fileno())
                        if stable(source.stat())!=stable(info) or sha(target)!=digest:raise ValueError('Source mutation/archive SHA mismatch: '+str(source))
                    dest=target.stat();entry={'source_path':str(source),'source_resolved_path':str(source.resolve()),'relative_path':str(target.relative_to(output)),'logical_relative_path':str(logical),'type':method,'category':row['category'],'bytes':info.st_size,'sha256':digest,'source_uid':info.st_uid,'source_device':info.st_dev,'source_inode':info.st_ino,'source_mtime_ns':info.st_mtime_ns,'destination_inode':dest.st_ino,'declared_immutable':row['declared_immutable'],'explicit_immutable_payload':row['explicit_immutable_payload'],'symlink_target':row['symlink_target'],**stored}
                    copied[str(source.resolve())]=digest;count+=1
                index.write((json.dumps(entry,allow_nan=False)+'\n').encode());index.flush()
        for item in plan['required_files']:
            if copied.get(str(Path(item['path']).resolve()))!=item['sha256']:raise ValueError('Retained closure differs from pinned SHA')
        check_hashes(bindings)
        for row in records:
            if row['kind']=='file':
                if stable(row['source'].stat())!=stable(row['info']):raise ValueError('Source changed before archive completion')
                if compressed_text(row) and strict_text_stat(row['source'].stat())!=strict_text_stat(row['info']):raise ValueError('Compressed source strict stat changed before completion')
        receipt.update(status='retained_final' if phase=='final' else 'retained_partial',archive_complete=True,files=count)
    except BaseException as caught:
        error=caught;receipt.update(status='archive_failed_partial_retained',archive_complete=False,experiment_complete_claim=False,files=count);receipt['errors'].append(type(caught).__name__+': '+str(caught))
    finally:
        receipt['ended_utc']=now()
        if inventory.exists():receipt['inventory_sha256']=sha(inventory)
        try:guarded_json(output/'archive_receipt.json',receipt,output,headroom=0)
        except BaseException as receipt_error:
            receipt['errors'].append('receipt_write_error: '+repr(receipt_error));print(json.dumps(receipt,allow_nan=False),file=sys.stderr)
            if error is None:error=receipt_error
    if error is not None:raise error
    return receipt


def safe_child(root,relative):
    rel=Path(relative)
    if not isinstance(relative,str) or rel.is_absolute() or not rel.parts or any(p in ('.','..') for p in rel.parts) or str(rel)!=relative:raise ValueError('Unsafe/noncanonical archive path')
    result=root/rel
    if not result.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [result,*result.parents]):raise ValueError('Path escape/symlink refused')
    return result


def restore(archive_root,output):
    archive_root=Path(archive_root).absolute();output=Path(output).absolute()
    if any(p.is_symlink() for p in [archive_root,*archive_root.parents]):raise ValueError('Archive symlink root refused')
    if output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents):raise FileExistsError('Restore requires entirely fresh nonsymlink directory')
    if output.resolve().is_relative_to(archive_root.resolve()) or archive_root.resolve().is_relative_to(output.resolve()):raise ValueError('Restore/archive roots cannot overlap')
    receipt_path=archive_root/'archive_receipt.json';inventory=archive_root/'inventory.jsonl'
    safe_child(archive_root,'archive_receipt.json');safe_child(archive_root,'inventory.jsonl');source_receipt_sha=sha(receipt_path);source_receipt=read(receipt_path)
    if source_receipt.get('schema')!='pi05_g1_mobile_archive_receipt_v5' or source_receipt.get('archive_complete') is not True or source_receipt.get('status') not in ('retained_partial','retained_final') or sha(inventory)!=source_receipt.get('inventory_sha256'):raise ValueError('Complete V5 archive/inventory identity required')
    output.mkdir(parents=True,exist_ok=False);receipt={'schema':'pi05_g1_mobile_archive_restore_v5','status':'in_progress','started_utc':now(),'source_archive':str(archive_root),'source_inventory_sha256':sha(inventory),'files':0,'errors':[]};error=None
    try:
        rows=[];logical={};stored_paths=set()
        with inventory.open() as stream:
            for line in stream:
                row=json.loads(line);target=safe_child(output,row['logical_relative_path']);stored=safe_child(archive_root,row['relative_path'])
                if Path(row['logical_relative_path']).parts[0]!='retained':raise ValueError('Logical output must remain below retained namespace')
                if row['logical_relative_path'] in logical or row['relative_path'] in stored_paths:raise ValueError('Duplicate logical/stored path')
                if row['type'] not in ('directory','copy','hardlink','gzip'):raise ValueError('Unknown retained type')
                if row['type']!='gzip' and row['relative_path']!=row['logical_relative_path']:raise ValueError('Plain/directory logical and stored paths differ')
                if row['type']=='directory' and not stored.is_dir():raise ValueError('Stored directory missing')
                logical[row['logical_relative_path']]=row['type'];stored_paths.add(row['relative_path'])
                if row['type']!='directory':
                    if type(row.get('bytes')) is not int or row['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',str(row.get('sha256'))):raise ValueError('Invalid logical size/SHA bound')
                    if not stored.is_file() or stored.is_symlink():raise ValueError('Missing stored regular file')
                    if row['type']=='gzip' and (row.get('stored_relative_path')!=row['relative_path'] or type(row.get('stored_bytes')) is not int or row['stored_bytes']!=stored.stat().st_size or sha(stored)!=row.get('stored_sha256') or row.get('roundtrip_original_sha256')!=row['sha256']):raise ValueError('Stored gzip identity differs')
                    if row['type']!='gzip' and stored.stat().st_size!=row['bytes']:raise ValueError('Plain stored size differs')
                rows.append((row,target,stored))
        for name,kind in logical.items():
            for parent in Path(name).parents:
                if str(parent) in logical and logical[str(parent)]!='directory':raise ValueError('File ancestor overlaps logical destination')
        remaining=[sum(row['bytes'] for row,_,_ in rows if row['type']!='directory')]
        if shutil.disk_usage(output).free<remaining[0]+RESERVE+CONTROL_HEADROOM:raise OSError('Restore uncompressed bytes plus24GiB reserve unavailable')
        for row,target,stored in rows:
            if row['type']=='directory':target.mkdir(parents=True,exist_ok=True);continue
            remaining[0]-=row['bytes'];target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0;before=stable(stored.stat());opener=gzip.open if row['type']=='gzip' else open
            with opener(stored,'rb') as src,target.open('xb',buffering=0) as dst:
                sink=FloorWriter(dst,output,lambda:remaining[0])
                while True:
                    block=src.read(min(CHUNK,row['bytes']-count+1))
                    if not block:break
                    count+=len(block)
                    if count>row['bytes']:raise ValueError('Restored bytes exceed declared original bound')
                    h.update(block);sink.write(block)
                sink.flush();os.fsync(dst.fileno())
            if count!=row['bytes'] or h.hexdigest()!=row['sha256'] or stable(stored.stat())!=before:raise ValueError('Restore size/SHA/source identity changed')
            receipt['files']+=1
        if sha(inventory)!=source_receipt['inventory_sha256'] or sha(receipt_path)!=source_receipt_sha:raise ValueError('Archive metadata changed during restore')
        receipt.update(status='completed',all_logical_bytes_SHA_verified=True)
    except BaseException as caught:
        error=caught;receipt['status']='restore_failed_partial_retained';receipt['errors'].append(type(caught).__name__+': '+str(caught))
    finally:
        receipt['ended_utc']=now()
        try:guarded_json(output/'restore_receipt.json',receipt,output,headroom=0)
        except BaseException as receipt_error:
            receipt['errors'].append('receipt_write_error: '+repr(receipt_error));print(json.dumps(receipt),file=sys.stderr)
            if error is None:error=receipt_error
    if error is not None:raise error
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',type=Path);mode.add_argument('--restore-from',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['partial','final']);a=p.parse_args()
    if a.restore_from:receipt=restore(a.restore_from,a.output)
    else:
        if not a.phase:p.error('--phase required for archive')
        receipt=archive(a.plan,a.output,a.phase)
    print(json.dumps({'status':receipt['status'],'output':str(a.output)}))

if __name__=='__main__':main()
