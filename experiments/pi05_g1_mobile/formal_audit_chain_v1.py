"""Stdlib small-JSON verification of independently audited formal prefix chains.

No raw trajectory/video/model payload hashing. Producer-prefix raw inventory
verification remains the responsibility of existing batch/v2 preparation.
"""
import hashlib,json,re
from pathlib import Path

POINTER='pi05_g1_mobile_formal_segment_pointer_v1'
CHECKS={'physical_counter_timing','initial_kinematic_sha','actual_runtime_identity','actions','policy_feedback','videos','acceptance'}
AUDITOR=Path(__file__).with_name('audit_episodes_v3.py').resolve()
LIMIT=16*1024**2

def check(value,message):
    if not value:raise ValueError(message)
def small(path):
    p=Path(path);check(p.is_absolute(),'Absolute small JSON required')
    check(not any(x.is_symlink() for x in [p,*p.parents]),'Symlink evidence forbidden')
    check(p.is_file() and p.stat().st_size<=LIMIT,'Missing/oversized small JSON')
    return p

def sha(path):
    p=small(path)
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(small(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def verify_prior_chain(latest_pointer,expected_end,plan_path):
    """Return verified nodes latest→genesis and latest three CPU evidence paths.

    Reads only small JSON and the auditor source for SHA; never follows audit
    input maps to image/video/rawtrace/model data. All pointer bindings themselves
    must be small JSON. The caller retains producer-only raw-prefix verification.
    """
    check(type(expected_end) is int and 3<=expected_end<=60 and expected_end%3==0,'Expected audited end must be3..60 multiple3')
    plan_path=small(plan_path).resolve();plan=read(plan_path);plan_sha=sha(plan_path)
    check(plan.get('schema')=='pi05_g1_mobile_benchmark_plan_v1' and len(plan.get('jobs',[]))==60,'Frozen60job plan required')
    tool_sha=hashlib.sha256(AUDITOR.read_bytes()).hexdigest();seen=set();nodes=[];bindings_all={};latest=small(latest_pointer).resolve();current=latest;end=expected_end
    while True:
        check(str(current) not in seen,'Pointer cycle');seen.add(str(current));pointer=read(current);start=end-3
        check(pointer.get('schema')==POINTER and pointer.get('status')=='segment_cpu_audit_passed' and pointer.get('passed') is True,'Independent CPU pointer must pass')
        check(pointer.get('plan_sha256')==plan_sha and type(pointer.get('segment_start_index')) is int and pointer['segment_start_index']==start and type(pointer.get('next_start_index')) is int and pointer['next_start_index']==end and type(pointer.get('audited_this_segment')) is int and pointer['audited_this_segment']==3,'Pointer gap/count/mixedplan')
        bound=pointer.get('bindings_sha256');check(isinstance(bound,dict) and bool(bound),'Pointer bindings missing');docs={}
        for name,h in bound.items():
            p=small(name).resolve();check(str(p)==name and isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) and sha(p)==h,'Changed pointer binding')
            docs[name]=read(p);bindings_all[name]=h
        check(bound.get(str(plan_path))==plan_sha,'Pointer lacks exact plan binding')
        def unique(schema):
            matches=[(Path(p),v) for p,v in docs.items() if v.get('schema')==schema]
            check(len(matches)==1,'Exactly one '+schema+' required');return matches[0]
        producer_path,producer=unique('pi05_g1_mobile_batch_segment_result_v1');guard_path,guard=unique('pi05_g1_mobile_formal_segment_guard_v1');audit_path,audit=unique('pi05_g1_mobile_episode_audits_v2')
        check(producer.get('status')=='completed' and producer.get('plan_sha256')==plan_sha and producer.get('segment_start_index')==start and producer.get('segment_stop_index_exclusive')==end and producer.get('previous_completed_jobs')==start and producer.get('completed_total_jobs')==end and producer.get('expected_segment_jobs')==3 and producer.get('expected_total_jobs')==60 and len(producer.get('jobs',[]))==3,'Producer segment mismatch')
        check(guard.get('status')=='passed' and guard.get('passed') is True and guard.get('mode')=='complete' and guard.get('actual_initial_state_match') is True and guard.get('plan_sha256')==plan_sha and guard.get('expected_start_index')==start and guard.get('verified_completed_prefix')==end and guard.get('checked_completed_segment_jobs')==3,'Complete guard/initial identity false')
        check(audit.get('passed') is True and audit.get('tool_sha256')==tool_sha and len(audit.get('episodes',[]))==3,'Independent v3 audit incomplete/changed tool')
        expected=plan['jobs'][start:end];expected_paths={str(Path(j['result_path']).resolve()) for j in expected}
        check(len(expected_paths)==3 and set(guard.get('case_dirs',[]))=={str(Path(p).parent) for p in expected_paths},'Guard actual case roster differs')
        rows={r.get('result_path'):r for r in audit['episodes']};check(len(rows)==3 and set(rows)==expected_paths,'Audit result roster differs')
        for index,job,record in zip(range(start,end),expected,producer['jobs']):
            rp=small(job['result_path']).resolve();raw=read(rp);row=rows[str(rp)];trace=str(rp.parent/'trajectory.jsonl');rh=sha(rp)
            check(record.get('index')==index and record.get('policy')==job['policy'] and record.get('case_id')==job['case_id'] and record.get('result_path')==str(rp) and record.get('status')=='completed' and record.get('source_interface_passed') is True and record.get('config_sha256')==job['config_sha256'],'Producer job identity mismatch')
            check(raw.get('status')=='completed' and raw.get('source_interface_passed') is True and raw.get('case_id')==job['case_id'] and raw.get('policy')==job['policy'] and type(raw.get('success')) is bool and raw['success']==record.get('success'),'Recorded result mismatch')
            check(raw.get('initial_physical_state_sha256')==guard.get('initial_physical_state_sha256'),'Initial guard/result hash mismatch')
            check(row.get('schema')=='pi05_g1_mobile_episode_cpu_audit_v2' and row.get('passed') is True and row.get('anomalies')==[] and set(row.get('checks',{}))==CHECKS and all(v.get('passed') is True for v in row['checks'].values()) and row.get('tool_sha256')==tool_sha and row.get('case_id')==job['case_id'],'Seven independent checks/selfSHA required')
            unsigned={k:v for k,v in row.items() if k!='audit_sha256'};check(row.get('audit_sha256')==digest(unsigned),'Audit selfSHA changed')
            check(record.get('result_sha256')==rh==row.get('inputs_sha256',{}).get(str(rp)),'Result recorded SHA differs')
            th=raw.get('trace_sha256');check(isinstance(th,str) and re.fullmatch('[0-9a-f]{64}',th) and record.get('case_files_sha256',{}).get(trace)==th==row.get('inputs_sha256',{}).get(trace),'Trace recorded SHA differs')
        prevs=[Path(p) for p,v in docs.items() if v.get('schema')==POINTER]
        check(len(prevs)==(1 if start else 0),'Unique previous independent pointer iff nonzero start')
        if start:
            prev=prevs[0];check(str(prev) not in seen,'Pointer cycle')
            if 'previous_pointer_path' in pointer:check(pointer['previous_pointer_path']==str(prev) and pointer.get('previous_pointer_sha256')==bound[str(prev)],'Explicit previous pointer binding mismatch')
            if 'audited_completed_prefix' in pointer:check(pointer['audited_completed_prefix']==end,'Audited prefix mismatch')
        else:prev=None
        nodes.append({'pointer_path':str(current),'pointer_sha256':sha(current),'start_index':start,'end_index':end,'producer_path':str(producer_path),'guard_path':str(guard_path),'audit_path':str(audit_path)})
        bindings_all[str(current)]=sha(current)
        if not start:break
        current=prev;end=start
    return {'schema':'pi05_g1_mobile_formal_audit_chain_verified_v1','passed':True,'plan_sha256':plan_sha,'audited_completed_prefix':expected_end,'nodes':nodes,'all_small_binding_sha256':bindings_all,'latest_direct_CPU_paths':[str(latest),nodes[0]['guard_path'],nodes[0]['audit_path']],'raw_payloads_rehashed':False}
