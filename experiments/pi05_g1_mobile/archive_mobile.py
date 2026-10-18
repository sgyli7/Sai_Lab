"""Explicit CPU permanent retention, with distinct partial/final evidence gates.

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

BACKUP_ROOT=Path('/home/ethan/ProjectBackups/2026-10-17')
CATEGORIES={'round03','source_snapshots','pi05_source','round02_source','mobile_source','queue_source',
            'round02_interrupted_history','official_base','native_n16','homie50','assets',
            'pi_tokenizer','native_tokenizer','openpi_source','native_source','provenance'}
EXCLUDED={'.git','__pycache__','.pytest_cache','venv','plot_venv','node_modules','.cache','cache','caches'}
PAYLOAD={'.safetensors','.pt','.onnx','.data','.bin','.usd','.usdc','.usda','.npy','.npz','.mp4','.png','.jpg','.stl','.obj','.model'}
RESERVE=32*1024**3


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
    if audit.get('schema')!='pi05_g1_mobile_episode_audits_v1' or audit.get('passed') is not True:raise ValueError('Independent episode audit failed')
    audited=audit.get('episodes',[]);paths={str(Path(r['result_path']).resolve()) for r in episodes}
    if len(audited)!=60 or {str(Path(r['result_path']).resolve()) for r in audited}!=paths:raise ValueError('Independent audit roster differs from benchmark60')
    check_hashes(summary['inputs_sha256']);bindings.update(summary['inputs_sha256'])
    for row in audited:
        if row.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v1' or row.get('passed') is not True or row.get('anomalies') or row.get('autonomous_policy_trial') is not True:raise ValueError('Invalid independent episode audit')
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
            eligible=bool(source.get('immutable') and current.suffix.lower() in PAYLOAD and info.st_uid!=0 and info.st_uid==os.getuid() and info.st_dev==destination_device)
            records.append({'source':current,'target':target,'kind':'file','category':source['category'],'info':info,'link_eligible':eligible,'declared_immutable':source.get('immutable',False),'symlink_target':str(current.resolve()) if current.is_symlink() else None})
    populated={r['category'] for r in records if r['kind']=='file'}
    if not CATEGORIES<=populated:raise ValueError('Coverage category contains no retained files')
    return records,omitted


def archive(plan_path,output,phase):
    if phase not in ['partial','final']:raise ValueError('Explicit partial/final phase required')
    output=Path(output).resolve();plan_path=Path(plan_path).resolve()
    if output.exists() or output.is_symlink():raise FileExistsError('Archive destination must be NEW; never overwrite')
    if not output.is_relative_to(BACKUP_ROOT.resolve()) or output==BACKUP_ROOT.resolve():raise ValueError('Archive must be a new child of dated ProjectBackups root')
    plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_mobile_archive_plan_v1':raise ValueError('Unsupported archive plan')
    bindings,state=readiness(plan,phase);bindings[str(plan_path)]=sha(plan_path)
    if not plan.get('required_files'):raise ValueError('Explicit admitted dependency/retention closure is required')
    # Metadata-only preflight: no large payload hashes before disk admission.
    parent=output.parent
    while not parent.exists():parent=parent.parent
    records,omitted=enumerate_sources(plan,parent.stat().st_dev)
    for row in records:
        if output.is_relative_to(row['source'].resolve()):raise ValueError('Destination cannot be inside a source tree')
    available=shutil.disk_usage(parent).free
    copy_bytes=sum(r['info'].st_size for r in records if r['kind']=='file' and not r['link_eligible'])
    all_bytes=sum(r['info'].st_size for r in records if r['kind']=='file')
    # Eligible links need no payload allocation. A failed link falls back only
    # after the per-file reserve check; failures preserve a typed partial archive.
    if available<copy_bytes+RESERVE:raise OSError('Insufficient disk for additional copies plus32GiB reserve')
    resolved_files={str(r['source'].resolve()) for r in records if r['kind']=='file'}
    for path in bindings:
        if path!=str(plan_path) and str(Path(path).resolve()) not in resolved_files:raise ValueError('Gate/evidence input missing from retained source roster: '+path)
    for item in plan['required_files']:
        if str(Path(item['path']).resolve()) not in resolved_files:raise ValueError('Dependency closure file absent from planned trees: '+item['path'])
    output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'pi05_g1_mobile_archive_receipt_v1','phase':phase,'status':'in_progress','archive_complete':False,
             'experiment_complete_claim':phase=='final','started_utc':now(),'plan_path':str(plan_path),'input_bindings':bindings,'readiness':state,
             'disk_preflight':{'available_bytes':available,'copy_bytes_without_links':all_bytes,'expected_copy_bytes':copy_bytes,'reserve_bytes':RESERVE},'omitted':omitted,'errors':[]}
    inventory=output/'inventory.jsonl';copied={};count=0
    try:
        with plan_path.open('rb') as src,(output/'archive_plan.json').open('xb') as dst:shutil.copyfileobj(src,dst)
        for item in plan['required_files']:
            if sha(item['path'])!=item['sha256']:raise ValueError('Pinned closure SHA mismatch: '+item['path'])
        with inventory.open('x') as index:
            for row in records:
                source=row['source'];target=output/'retained'/row['target']
                if row['kind']=='directory':target.mkdir(parents=True,exist_ok=True);entry={'source_path':str(source),'relative_path':str(target.relative_to(output)),'type':'directory','category':row['category']}
                else:
                    info=row['info']
                    if stable(source.stat())!=stable(info):raise ValueError('Source changed after preflight: '+str(source))
                    digest=sha(source);target.parent.mkdir(parents=True,exist_ok=True);method='copy'
                    if row['link_eligible']:
                        try:os.link(source,target);method='hardlink'
                        except OSError:pass
                    if method=='copy':
                        if shutil.disk_usage(target.parent).free<info.st_size+RESERVE:raise OSError('Reserve would be consumed during copy')
                        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
                    if stable(source.stat())!=stable(info) or sha(target)!=digest:raise ValueError('Source mutation/archive SHA mismatch: '+str(source))
                    dest=target.stat();entry={'source_path':str(source),'source_resolved_path':str(source.resolve()),'relative_path':str(target.relative_to(output)),
                           'type':method,'category':row['category'],'bytes':info.st_size,'sha256':digest,'source_uid':info.st_uid,
                           'source_device':info.st_dev,'source_inode':info.st_ino,'source_mtime_ns':info.st_mtime_ns,'destination_inode':dest.st_ino,
                           'declared_immutable':row['declared_immutable'],'symlink_target':row['symlink_target']}
                    copied[str(source.resolve())]=digest;count+=1
                index.write(json.dumps(entry,allow_nan=False)+'\n');index.flush()
        for item in plan['required_files']:
            if copied.get(str(Path(item['path']).resolve()))!=item['sha256']:raise ValueError('Retained closure differs from pinned SHA')
        check_hashes(bindings)
        for row in records:
            if row['kind']=='file' and stable(row['source'].stat())!=stable(row['info']):raise ValueError('Source changed before archive completion')
        receipt.update(status='retained_final' if phase=='final' else 'retained_partial',archive_complete=True,files=count)
    except BaseException as error:
        receipt.update(status='archive_failed_partial_retained',archive_complete=False,experiment_complete_claim=False,files=count)
        receipt['errors'].append(type(error).__name__+': '+str(error));raise
    finally:
        receipt['ended_utc']=now()
        if inventory.exists():receipt['inventory_sha256']=sha(inventory)
        write_new(output/'archive_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['partial','final'],required=True);a=p.parse_args()
    receipt=archive(a.plan,a.output,a.phase);print(json.dumps({'status':receipt['status'],'output':str(a.output)}))

if __name__=='__main__':main()
