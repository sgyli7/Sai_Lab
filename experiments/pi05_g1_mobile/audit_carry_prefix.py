"""CPU admission of the explicitly authorized native201 carry prefix only.

Requires a full v2 independent audit, preserves the legacy v1 video-origin fail,
and hashes every lossless input PNG through the boundary. Does not convert data,
train, relabel the failed full task, change normalization, or score final cases.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import PROMPT


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def admit(folder,v2_path,legacy_path,diagnostic_path,output):
    folder=Path(folder).resolve();v2_path=Path(v2_path).resolve();legacy_path=Path(legacy_path).resolve();diagnostic_path=Path(diagnostic_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior prefix admission receipt')
    paths={k:folder/v for k,v in [('result','result.json'),('trace','trajectory.jsonl'),('config','config.json'),('runtime_identity','runtime_identity.json')]}
    result=read(paths['result']);config=read(paths['config']);runtime=read(paths['runtime_identity']);v2=read(v2_path);legacy=read(legacy_path);diagnostic=read(diagnostic_path)
    if v2.get('schema')!='pi05_g1_mobile_episode_audits_v2' or v2.get('passed') is not True:raise ValueError('Full independent v2 audit must pass')
    cases=[e for e in v2['episodes'] if Path(e['result_path']).resolve()==paths['result']]
    if len(cases)!=1:raise ValueError('Exactly one bound v2 episode required')
    audit=cases[0]
    if audit.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v2' or audit.get('passed') is not True or audit.get('anomalies') or audit.get('autonomous_policy_trial') is not True or not audit.get('checks') or not all(c.get('passed') is True for c in audit['checks'].values()):raise ValueError('V2 episode must pass every independent check')
    unsigned={k:v for k,v in audit.items() if k!='audit_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()!=audit['audit_sha256']:raise ValueError('V2 episode self SHA mismatch')
    for name,digest in audit['inputs_sha256'].items():
        if sha(name)!=digest:raise ValueError('V2 bound input changed: '+name)
    binding={k:sha(path) for k,path in paths.items()}
    if any(audit['inputs_sha256'].get(str(path))!=binding[k] for k,path in paths.items()):raise ValueError('V2 audit lacks required raw evidence binding')
    oldcases=[e for e in legacy.get('episodes',[]) if Path(e['result_path']).resolve()==paths['result']]
    if legacy.get('schema')!='pi05_g1_mobile_episode_audits_v1' or legacy.get('passed') is not False or len(oldcases)!=1 or oldcases[0].get('passed') is not False or oldcases[0].get('anomalies')!=['ValueError: Video PTS/time origin mismatch']:raise ValueError('Legacy audit failure must remain precisely recorded')
    case=result['case'];score=audit['recomputed_acceptance'];T=score['carry_qualified_control_tick']
    if case.get('seed')!=201 or 'formal' in case['case_id'].lower() or config.get('case')!=case:raise ValueError('Only declared development family201 allowed')
    if config.get('mode')!='policy' or config.get('policy_id')!='gr00t_n16' or result.get('policy')!='gr00t_n16' or config.get('prompt')!=PROMPT:raise ValueError('Native policy-only task prompt required')
    if type(T) is not int or T<=0 or result.get('status')!='completed' or result.get('success') is not False or score.get('success') is not False or score.get('fell') or score.get('dropped') or result['acceptance']['carry_qualified_control_tick']!=T:raise ValueError('Carry qualification in failed complete task required')
    if score['max_carried_box_displacement_m']<4 or score['carry_turn_degrees']<60 or score['source_target_distance_m']<5:raise ValueError('Frozen carry conditions not met')
    if runtime['wbc_config'].get('enable_waist') is not False:raise ValueError('Native Homie waist ownership differs')
    if diagnostic.get('schema')!='pi05_g1_mobile_native_dev04_physical_diagnostic_v1' or diagnostic.get('acceptance_recomputed_exact_all_fields_with_1e6_float_tolerance') is not True:raise ValueError('Independent origin/physical diagnostic required')
    for name in ['head_50fps.mp4','observer_50fps.mp4']:
        evidence=diagnostic['videos'][name];actual=audit['checks']['videos']['streams']['head_video_path' if name.startswith('head') else 'observer_video_path']
        if evidence['sha256']!=actual['sha256'] or evidence['frames']!=actual['frames'] or evidence['first_pts_seconds']!=actual['first_pts_seconds'] or evidence.get('relative_50hz_contiguous') is not True:raise ValueError('Video diagnostic differs from v2 full audit')
    rows=[json.loads(line) for line in paths['trace'].open()]
    if T>=len(rows):raise ValueError('Boundary not covered')
    if any(s['fell'] or s['ground_drop'] for s in result['acceptance']['samples'][:T+1]):raise ValueError('Prefix includes fall/drop')
    entries=[];raw_inputs={}
    for tick,row in enumerate(rows[:T+1]):
        if row['control_tick']!=tick or row['physics_steps_since_reset']!=4*tick or abs(row['sim_time_seconds']-tick/50)>1e-9:raise ValueError('Prefix trace clock discontinuity')
        image_path=folder/'head_rgb'/f'{tick:05d}.png';byte_sha=sha(image_path)
        with Image.open(image_path) as image:
            if image.format!='PNG' or image.mode!='RGB' or image.size!=(640,480):raise ValueError('Lossless raw prefix image type/shape differs')
            rgb_sha=hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()
        if rgb_sha!=row['head_rgb_sha256']:raise ValueError('Prefix raw RGB pixels differ from synchronized trace')
        entry={'path':str(image_path),'byte_sha256':byte_sha,'rgb_sha256':rgb_sha,'control_tick':tick,'timestamp_seconds':tick/50}
        entries.append(entry);raw_inputs[str(image_path)]=byte_sha
    # Seal only after both original evidence and every individual PNG remain unchanged.
    for key,path in paths.items():
        if sha(path)!=binding[key]:raise ValueError('Raw evidence changed during admission')
    for name,digest in raw_inputs.items():
        if sha(name)!=digest:raise ValueError('PNG changed during admission')
    report={'schema':'pi05_g1_mobile_carry_prefix_audit_v1','status':'passed','case_id':case['case_id'],'seed':201,'family_id':'native201','split':'train_only',
            'carry_qualified_control_tick':T,'carry_qualified_sim_seconds':T/50,'prefix_no_fall_drop':True,'native_policy_only':True,
            'scope_partial_carry_not_task_success':True,'source_global_audit_passed':False,'legacy_v1_audit_passed':False,'independent_v2_audit_passed':True,
            'independent_v2_audit':{'path':str(v2_path),'sha256':sha(v2_path)},'legacy_v1_audit':{'path':str(legacy_path),'sha256':sha(legacy_path)},
            'video_origin_diagnostic':{'path':str(diagnostic_path),'sha256':sha(diagnostic_path)},'bind_sha256':binding,
            'source_task_success':False,'source_task_termination_reasons':score['termination_reasons'],
            'supervised_observation_ticks':[0,T-1],'supervised_applied_action_rows':[1,T],'raw_rgb_frames':entries[:-1],
            'boundary_frame':entries[-1],'audited_raw_png_count':len(entries),'full_source_frozen_dependency_count':audit['checks']['actual_runtime_identity']['frozen_dependency_count'],
            'limitation':'Train-only single development family carry prefix. No release/stable-placement success labels, no validation/test inclusion, no change to original train-only normalization, no formal benchmark score.',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tool_sha256':sha(__file__),'data_conversion_executed':False,'gpu_started':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['case-dir','v2-audit','legacy-audit','diagnostic','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=admit(a.case_dir,a.v2_audit,a.legacy_audit,a.diagnostic,a.output);print(json.dumps({'status':r['status'],'T':r['carry_qualified_control_tick'],'audited_raw_png_count':r['audited_raw_png_count'],'task_success_teacher':False}))

if __name__=='__main__':main()
