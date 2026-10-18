"""Optional POSIX clean-page hint for phase-verified, root-approved OWNED models.

Never rehash/read model payloads. No CUDA, global cache drop, unlink, chmod or
queue calls. The existing factory phase stat lacks mode; a separate explicit
root approval binds mode/stat after verified SHA. A hint does not prove eviction.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,re,stat,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MOBILE=ROOT/'.scratch/pi05_g1/round03_mobile'
BASE=ROOT/'.scratch/pi05_g1/pi05_base_pytorch/model.safetensors'
TRAIN=MOBILE/'training_carry_run01'
NATIVE=Path('/home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93')
NATIVE_RECEIPT=ROOT/'.scratch/unitree_g1/policy/mobile_box_files.json'
BASE_SHA='c1af1d96ce7d40f1f66fafd9adc658c39c88581a017ea5283b1eaa8dffac565d'
SCHEMA='pi05_g1_mobile_owned_model_cache_approval_v1'
STAT_KEYS=('bytes','device','inode','mtime_ns','mode','uid')


def read_small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2:raise ValueError('Small regular metadata file required:'+str(path))
    payload=path.read_bytes();return json.loads(payload),hashlib.sha256(payload).hexdigest()


def metadata(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Model symlink forbidden')
    value=path.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid():raise ValueError('Only current-user regular model files allowed')
    return {'bytes':value.st_size,'device':value.st_dev,'inode':value.st_ino,'mtime_ns':value.st_mtime_ns,'mode':stat.S_IMODE(value.st_mode),'uid':value.st_uid}


def mem():
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemFree','MemAvailable','Cached','Dirty'):values[key+'_bytes']=int(value.split()[0])*1024
    return values


def context(plan_path,phase_path,quiescence_path,*,now=None):
    plan_path=Path(plan_path).resolve();phase_path=Path(phase_path).resolve();quiescence_path=Path(quiescence_path).resolve()
    plan,plan_sha=read_small(plan_path);phase,phase_sha=read_small(phase_path);q,q_sha=read_small(quiescence_path)
    if plan.get('schema')!='pi05_g1_mobile_benchmark_plan_v1' or phase.get('schema')!='pi05_g1_mobile_execution_phase_sha_v1' or phase.get('status')!='verified' or Path(phase.get('plan','')).resolve()!=plan_path or phase.get('plan_sha256')!=plan_sha or phase.get('frozen_files')!=plan.get('frozen_files'):raise ValueError('Actual factory plan/phase verified identity mismatch')
    ledger=MOBILE/'gpu_budget.json';_,ledger_sha=read_small(ledger)
    if q.get('schema')!='pi05_g1_mobile_owned_quiescence_v1' or q.get('passed') is not True or q.get('owned_closed') is not True or Path(q.get('ledger','')).resolve()!=ledger or q.get('ledger_sha256')!=ledger_sha or any(q.get(key) for key in ('active_owned_units','active_owned_container_ids','accounting_open_tokens')):raise ValueError('Fresh owned-quiescence guard failed')
    now=time.time() if now is None else now
    checked=datetime.datetime.fromisoformat(q['checked_utc']).timestamp()
    if not 0<=now-checked<=60:raise ValueError('Owned quiescence must be at most60s old; no future timestamp')
    selection_path=Path(plan['selection_receipt']).resolve();selection,selection_sha=read_small(selection_path)
    if plan['frozen_files'].get(str(selection_path))!=selection_sha:raise ValueError('Factory selection receipt not frozen by verified plan')
    chosen=Path(selection['recorded_selected']['path']).resolve()
    if chosen.parent!=TRAIN.resolve() or not re.fullmatch(r'(?:step|final)_\d{6}',chosen.name):raise ValueError('Only actual selected direct child of training_carry_run01 allowed')
    meta,meta_sha=read_small(chosen/'metadata.json')
    if selection['recorded_selected']['metadata_sha256']!=meta_sha:raise ValueError('Selected metadata differs from validated factory selection')
    native,native_sha=read_small(NATIVE_RECEIPT)
    if native.get('profile')!='mobile_box' or native.get('revision')!=NATIVE.name or Path(native['model_root']).resolve()!=NATIVE.resolve():raise ValueError('Fixed native model receipt differs')
    if BASE.is_symlink() or (chosen/'model.safetensors').is_symlink():raise ValueError('Model path symlink forbidden')
    if plan['frozen_files'].get(str(NATIVE_RECEIPT.resolve()))!=native_sha:raise ValueError('Native receipt not bound to phase-verified plan')
    allowed={str(BASE.resolve()):BASE_SHA,str((chosen/'model.safetensors').resolve()):meta.get('weights_sha256')}
    for item in native['files']:
        if item['path'].endswith('.safetensors'):
            child=(NATIVE/item['path']).resolve()
            if child.parent!=NATIVE.resolve():raise ValueError('Native weight escaped pinned revision')
            allowed[str(child)]=item['sha256']
    entries=[]
    for policy,item in plan['model_artifacts'].items():
        if policy not in ('official_pi05_base','pi05_mobile','gr00t_n16'):raise ValueError('Unexpected/peer policy artifact')
        entries.extend(item if isinstance(item,list) else [item])
    if {str(Path(x['path']).resolve()) for x in entries}!=set(allowed):raise ValueError('Plan weights not exactly owned allowed set')
    observed=phase['model_artifacts_observed'];artifacts=[]
    for entry in entries:
        path=str(Path(entry['path']).resolve());verified=observed.get(path)
        if not verified or entry['sha256']!=allowed[path] or verified.get('sha256')!=allowed[path]:raise ValueError('Prior phase SHA not bound to approved artifact')
        current=metadata(path)
        for key in ('bytes','device','inode','mtime_ns'):
            if entry.get(key)!=verified.get(key) or current[key]!=verified.get(key):raise ValueError('Model stat changed since streaming phase verification:'+path)
        artifacts.append({'path':path,'sha256':allowed[path],**current})
    return {'plan_path':str(plan_path),'plan_sha256':plan_sha,'phase_path':str(phase_path),'phase_sha256':phase_sha,'quiescence_path':str(quiescence_path),'quiescence_sha256':q_sha,'selection_sha256':selection_sha,'native_receipt_sha256':native_sha,'ledger_sha256':ledger_sha,'artifacts':artifacts,'scope':'Own resources closed only; global idle is not required or claimed.'}


def proposal(plan,phase,quiescence,output):
    ctx=context(plan,phase,quiescence)
    value={'schema':SCHEMA,'root_approved':False,'mode_binding':'explicit_root_metadata_after_streaming_phase_verification','context':ctx,'note':'Review and explicitly approve a fresh manifest. Factory phase records size/device/inode/mtime but not mode; this proposal binds mode/uid without reading weights. No fadvise executed.'}
    with Path(output).open('x') as f:json.dump(value,f,indent=2)
    return value


def hint(approval_path,output,*,fadvise=None,now=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh cache-hint receipt required')
    report={'schema':'pi05_g1_mobile_owned_model_cache_hint_v1','status':'failed','model_payload_read_or_hashed':False,'global_cache_operation':False,'eviction_guaranteed':False,'gpu_started':False,'hints':[],'memory_before':mem()}
    try:
        approval,approval_sha=read_small(approval_path)
        if approval.get('schema')!=SCHEMA or approval.get('root_approved') is not True or approval.get('mode_binding')!='explicit_root_metadata_after_streaming_phase_verification':raise ValueError('Explicit root-approved metadata supplement required; phase alone lacks mode')
        old=approval['context'];ctx=context(old['plan_path'],old['phase_path'],old['quiescence_path'],now=now)
        if ctx!=old:raise ValueError('Approved context/stat/mode/mtime changed;never refresh approval')
        report['approval_sha256']=approval_sha;report['context']=ctx
        if not hasattr(os,'posix_fadvise') or not hasattr(os,'POSIX_FADV_DONTNEED'):raise NotImplementedError('POSIX_FADV_DONTNEED unavailable')
        fadvise=fadvise or os.posix_fadvise;seen=set()
        for entry in ctx['artifacts']:
            key=(entry['device'],entry['inode'])
            if key in seen:report['hints'].append({'path':entry['path'],'status':'deduplicated_same_inode'});continue
            seen.add(key);path=Path(entry['path']);before=metadata(path)
            if any(before[k]!=entry[k] for k in STAT_KEYS):raise ValueError('Approved metadata changed immediately before hint')
            fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            result={'path':str(path),'inode':entry['inode'],'device':entry['device'],'status':'failed'}
            try:
                handle=os.fstat(fd)
                if (handle.st_dev,handle.st_ino,handle.st_size,handle.st_mtime_ns,stat.S_IMODE(handle.st_mode),handle.st_uid)!=(entry['device'],entry['inode'],entry['bytes'],entry['mtime_ns'],entry['mode'],entry['uid']):raise ValueError('Opened file differs from approved model')
                fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);result['status']='hint_accepted_not_eviction_proof'
            except OSError as error:result.update(error=repr(error),errno=error.errno)
            finally:os.close(fd)
            after=metadata(path);result.update(metadata_before=before,metadata_after=after,metadata_unchanged=before==after);report['hints'].append(result)
            if before!=after:raise ValueError('Model metadata changed during hint')
            if result['status']=='failed':raise RuntimeError('Owned cache hint unsupported/failed;see per-file errno')
        # Guard lifetime/races without reading model payloads again.
        if context(old['plan_path'],old['phase_path'],old['quiescence_path'],now=now)!=old:raise ValueError('Owned context changed during hints')
        report['status']='completed'
    except Exception as error:report['error']=repr(error)
    report['memory_after']=mem();report['note']='DONTNEED is advisory for clean file-backed pages;busy/dirty pages mayremain. MemAvailable is not CUDA allocatable memory. This is not an OOM fix proof.'
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--approval');p.add_argument('--propose',action='store_true')
    for name in ('plan','phase','quiescence'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.propose:
        if not all((a.plan,a.phase,a.quiescence)):p.error('Proposal requires plan/phase/quiescence')
        proposal(a.plan,a.phase,a.quiescence,a.output);return 0
    if not a.approval:p.error('Hint requires explicit root approval')
    result=hint(a.approval,a.output);print(json.dumps({'status':result['status'],'output':a.output}));return 0 if result['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
