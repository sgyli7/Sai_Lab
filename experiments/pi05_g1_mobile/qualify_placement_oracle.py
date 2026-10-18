"""CPU gate for completed, independently audited nonformal placement oracle only.

--episode ABS_CASE_DIR --audit ABS_AUDIT_JSON --output NEW_JSON.
Never upgrades failed producers, calls a policy, or equates readyspan with stable
placement. Requires a real101-sample ready window AND frozen all-pairs pose rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import acceptance,acceptance_pose_speed as measurement,audit_episodes_v3 as independent

SCHEMA='pi05_g1_mobile_placement_primitive_qualification_v1'
FLAGS=('native_policy_trial','full_task_trial','teacher_trial','formal_benchmark_trial')
REQUIRED_CHECKS={'physical_counter_timing','initial_kinematic_sha','actual_runtime_identity','actions','policy_feedback','videos','acceptance'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def validate(episode,audit_path):
    folder=Path(episode).resolve();result_path=folder/'result.json';config_path=folder/'config.json';trace_path=folder/'trajectory.jsonl'
    result=read(result_path);config=read(config_path)
    if (result.get('schema')!='pi05_g1_mobile_case_result_v1' or result.get('status')!='completed' or
        result.get('source_interface_passed') is not True or result.get('success') is not False or result.get('cleanup_errors')):
        raise ValueError('Original oracle producer must truly complete, source-pass and successfalse; failed remains rejected')
    if config.get('mode')!='replay' or config.get('ticks')!=150 or config.get('evaluate_acceptance') is not True:
        raise ValueError('Explicit150-control replay oracle required')
    if any(config.get(flag) is not False for flag in FLAGS):raise ValueError('All four explicit nonformal/nonpolicy flags must befalse')
    if result.get('case_id')!=config['case']['case_id'] or result.get('policy')!=config['policy_id']:raise ValueError('Oracle case/policy identity mismatch')
    if any(flag in result and result[flag] is not False for flag in FLAGS):raise ValueError('Result contradicts explicit nonformal flags')
    if result.get('completed_control_ticks')!=150 or result.get('actual_sdk_integrations')!=600 or result.get('control_hz')!=50 or result.get('physics_hz')!=200:
        raise ValueError('Actual150controls/600physics/50Hz/200Hz required')
    doc=read(audit_path)
    if doc.get('passed') is not True:raise ValueError('Independent audit must pass')
    if doc.get('schema')=='pi05_g1_mobile_episode_audits_v2':
        matches=[a for a in doc['episodes'] if Path(a['result_path']).resolve()==result_path]
        if len(matches)!=1:raise ValueError('Independent audit must uniquely bind this case')
        audit=matches[0]
        if doc.get('tool_sha256')!=sha(independent.__file__):raise ValueError('Independent audit must use current frozen v3 tool')
    else:audit=doc
    if (audit.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v2' or audit.get('passed') is not True or audit.get('anomalies') or
        audit.get('autonomous_policy_trial') is not False or Path(audit.get('result_path','')).resolve()!=result_path or audit.get('tool_sha256')!=sha(independent.__file__)):
        raise ValueError('Exact nonautonomous independent v3 episode pass required')
    unsigned={k:v for k,v in audit.items() if k!='audit_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()!=audit.get('audit_sha256'):raise ValueError('Audit self SHA mismatch')
    checks=audit.get('checks',{})
    if not REQUIRED_CHECKS<=checks.keys() or any(c.get('passed') is not True for c in checks.values()):raise ValueError('Missing/failed independent physical/action/input/video checks')
    if checks['actual_runtime_identity'].get('full_roster_verified') is not True:raise ValueError('Frozen dependency roster not independently verified')
    counter=checks['physical_counter_timing']
    if (counter.get('controls'),counter.get('observations'),counter.get('physics_integrations'))!=(150,151,600):
        raise ValueError('Independent audit counters do not prove150/151/600')
    if checks['actual_runtime_identity'].get('frozen_dependency_count')!=len(config.get('frozen_files',{})) or not config.get('frozen_files'):
        raise ValueError('Independent frozen dependency count differs from actual config')
    bindings=audit['inputs_sha256']
    for p in (result_path,config_path,trace_path,folder/'runtime_identity.json'):
        if str(p) not in bindings:raise ValueError('Independent audit missing case input: '+str(p))
    for name,digest in bindings.items():
        if not Path(name).is_absolute() or sha(name)!=digest:raise ValueError('Current source/evidence differs from independent audit: '+name)
    if result.get('trace_sha256')!=sha(trace_path):raise ValueError('Original trace SHA mismatch')
    for source in (measurement.__file__,measurement.body_ids.__file__):
        if config['frozen_files'].get(str(Path(source).resolve()))!=sha(source):raise ValueError('Measurement/bodyadapter must be explicitly frozen in actual config')
    rows=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if len(rows)!=151:raise ValueError('Exactly151 complete raw observations required')
    measurement.validate_raw_rows(rows)
    context={**config,'audit_body_names':result['body_names'],'audit_contact_metadata':result['contact_view']}
    judgment=measurement.evaluate_episode(rows,context)
    independent.compare(result['acceptance'],judgment)
    independent.compare(audit['recomputed_acceptance'],{k:v for k,v in judgment.items() if k!='samples'})
    if (judgment.get('measurement_source',{}).get('schema')!=measurement.SCHEMA or judgment['measurement_source'].get('raw_SDK_velocity6_retained_unchanged') is not True):
        raise ValueError('Explicit new measurement identity missing')
    if any(judgment[k] is not False for k in ('success','grasped','fell','dropped')) or judgment['max_ready_span_seconds']<2:
        raise ValueError('Preplaced oracle requires no grasp/fall/drop/full-task success and at least2s readyspan')
    rule=config['acceptance']
    if rule['stable_seconds']!=2. or rule['stable_position_m']!=.005 or rule['stable_angle_degrees']!=5.:
        raise ValueError('Frozen2s/5mm/5degree placement thresholds required')
    windows=[];samples=judgment['samples']
    if len(samples)!=151:raise ValueError('Missing recomputed per-row readiness')
    for start in range(51):
        if all(s['tick']==start+i and s['placement_ready'] is True for i,s in enumerate(samples[start:start+101])):
            if acceptance.stable_pose_window(rows[start:start+101],rule['stable_position_m'],rule['stable_angle_degrees']):
                windows.append({'start_tick':start,'end_tick':start+100,'samples':101,'seconds':2.})
    if not windows:raise ValueError('No actual contiguous2s ready AND all-pairs stable box/bin window')
    streams=checks['videos']['streams']
    if set(streams)!={'head_video_path','observer_video_path'}:raise ValueError('Two independently decoded raw streams required')
    videos={}
    for field in ('head_video_path','observer_video_path'):
        item=streams[field];path=Path(item['path']).resolve()
        if path!=Path(result[field]).resolve() or item.get('sha256')!=sha(path) or bindings.get(str(path))!=item['sha256']:
            raise ValueError('Raw video path/SHA differs from independent audit')
        actual=independent.video_probe(path,151)
        for key,value in actual.items():
            if item.get(key)!=value:raise ValueError('Independent full video decode metadata changed: '+key)
        videos[field]={'path':str(path),'sha256':sha(path),**actual}
    independent.check_video_pair(videos)
    for p,digest in bindings.items():
        if sha(p)!=digest:raise ValueError('Input changed during qualification: '+p)
    return {'case_id':config['case']['case_id'],'original_producer_status':'completed','original_result_success':False,
        'qualified_stable_2s_windows':len(windows),'windows':windows,'max_ready_span_seconds':judgment['max_ready_span_seconds'],
        'measurement_source':judgment['measurement_source'],'trace_sha256':sha(trace_path),'independent_audit_sha256':audit['audit_sha256'],
        'independent_audit_file_sha256':sha(audit_path),'inputs_sha256':bindings,'videos':videos,'controls':150,'physics_integrations':600}


def qualify(episode,audit_path,output):
    for p in (episode,audit_path,output):
        if not Path(p).is_absolute():raise ValueError('Explicit absolute paths required')
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite prior qualification')
    report={'schema':SCHEMA,'status':'failed','passed':False,'full_task_success':False,'autonomous':False,'teacher':False,
        'native_policy_trial':False,'full_task_trial':False,'teacher_trial':False,'formal_benchmark_trial':False,
        'episode':str(Path(episode).resolve()),'audit':str(Path(audit_path).resolve()),'tool_sha256':sha(__file__),
        'scope':'Physical placement primitive qualification only; never autonomous/full-task/teacher evidence. Original results remain unchanged.'}
    try:report.update(validate(episode,audit_path));report.update(status='completed',passed=True)
    except Exception as error:report['error']=type(error).__name__+': '+str(error)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True,type=Path);parser.add_argument('--audit',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=qualify(args.episode,args.audit,args.output)
    print(json.dumps({'status':result['status'],'passed':result['passed'],'qualified_stable_2s_windows':result.get('qualified_stable_2s_windows',0)}))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
