"""Train-only single-family physical carry prefix; never a successful task teacher.

No preparation occurs implicitly. A separate passed prefix audit plus explicit
root authorization is required to run the converter on actual case evidence.
Original80/10/10 data and original80-train normalization remain unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import PROMPT,encode_state,decode_to_source

SCHEMA='pi05_g1_mobile_carry_prefix_v1'
AUDIT_SCHEMA='pi05_g1_mobile_carry_prefix_audit_v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def original_identity(manifest,norm):
    document=json.loads(Path(manifest).read_text());splits=[document[k] for k in ('train','validation','test')]
    if [len(x) for x in splits]!=[80,10,10] or any(len(set(x))!=len(x) for x in splits) or any(set(splits[a])&set(splits[b]) for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Original source80/10/10 must remain disjoint and unchanged')
    return {'source_manifest_sha256':sha(manifest),'norm_sha256':sha(norm),'normalization_policy':'original80train_only_unchanged;carry_prefix_excluded_from_norm','source_test_payloads_opened':False}


def raw_png(entry):
    path=Path(entry['path'])
    if sha(path)!=entry['byte_sha256']:raise ValueError('Carry raw PNG bytes changed')
    with Image.open(path) as image:
        if image.format!='PNG' or image.mode!='RGB':raise ValueError('Raw lossless RGB PNG required')
        rgb=np.asarray(image).copy()
    if rgb.shape!=(480,640,3) or hashlib.sha256(rgb.tobytes()).hexdigest()!=entry['rgb_sha256']:raise ValueError('Carry synchronized raw pixel identity differs')
    return rgb



def verify_v2(audit,paths):
    if audit.get('legacy_v1_audit_passed') is not False or audit.get('independent_v2_audit_passed') is not True:raise ValueError('Explicit legacy-v1 failure and independent-v2 pass required')
    if audit.get('source_global_audit_passed') is not False:raise ValueError('Compatibility global audit flag refers to legacy-v1 only')
    receipt=audit['independent_v2_audit'];path=Path(receipt['path']).resolve()
    if sha(path)!=receipt['sha256']:raise ValueError('Independent v2 audit receipt changed')
    document=json.loads(path.read_text())
    if document.get('passed') is not True:raise ValueError('Independent v2 full audit is not passed')
    if document.get('schema')=='pi05_g1_mobile_episode_audits_v2':episodes=document['episodes']
    elif document.get('schema')=='pi05_g1_mobile_episode_cpu_audit_v2':episodes=[document]
    else:raise ValueError('Independent v2 audit schema differs')
    matches=[row for row in episodes if Path(row.get('result_path','')).resolve()==paths['result'].resolve()]
    if len(matches)!=1:raise ValueError('Independent v2 audit case identity mismatch')
    row=matches[0]
    if row.get('passed') is not True or row.get('autonomous_policy_trial') is not True or not row.get('checks') or any(value.get('passed') is not True for value in row['checks'].values()):raise ValueError('Independent v2 case does not pass every check')
    for key,evidence_path in paths.items():
        if row.get('inputs_sha256',{}).get(str(evidence_path.resolve()))!=sha(evidence_path):raise ValueError('Independent v2 evidence identity mismatch:'+key)
    acceptance=row.get('recomputed_acceptance',{})
    if acceptance.get('carry_qualified_control_tick')!=audit['carry_qualified_control_tick'] or acceptance.get('success') is not False or acceptance.get('fell') or acceptance.get('dropped'):raise ValueError('Independent v2 recomputed carry boundary/task identity mismatch')
    return {'path':str(path),'sha256':receipt['sha256'],'inputs_sha256':row['inputs_sha256']}


def prepare(case_dir,audit_path,source_manifest,norm,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New prefix output required')
    folder=Path(case_dir).resolve();audit_path=Path(audit_path).resolve();audit=json.loads(audit_path.read_text())
    if audit.get('schema')!=AUDIT_SCHEMA or audit.get('status')!='passed' or audit.get('prefix_no_fall_drop') is not True or audit.get('native_policy_only') is not True:raise ValueError('Independent passed carry-prefix audit required')
    if audit.get('scope_partial_carry_not_task_success') is not True or audit.get('source_global_audit_passed') is not False:raise ValueError('Explicit partial-carry vs failed global audit scope required')
    diagnostic=audit['video_origin_diagnostic']
    if sha(diagnostic['path'])!=diagnostic['sha256']:raise ValueError('Video origin diagnostic binding differs')
    paths={'result':folder/'result.json','trace':folder/'trajectory.jsonl','config':folder/'config.json','runtime_identity':folder/'runtime_identity.json'}
    for key,path in paths.items():
        if audit.get('bind_sha256',{}).get(key)!=sha(path):raise ValueError('Prefix audit binding differs:'+key)
    v2=verify_v2(audit,paths)
    result=json.loads(paths['result'].read_text());config=json.loads(paths['config'].read_text());runtime=json.loads(paths['runtime_identity'].read_text())
    case=result['case'];tick=audit['carry_qualified_control_tick']
    if type(tick) is not int or tick<=0:raise ValueError('Positive first carry qualification tick required')
    if case.get('seed')!=201 or audit.get('seed')!=201 or audit.get('case_id')!=case['case_id'] or 'formal' in case['case_id'].lower():raise ValueError('Only declared train-only case/family201 allowed')
    if result.get('status')!='completed' or result.get('success') is not False:raise ValueError('This adapter specifically records completed taskFalse carry evidence')
    if config.get('mode')!='policy' or config.get('policy_id')!='gr00t_n16' or result.get('policy')!='gr00t_n16' or config.get('case')!=case or config.get('prompt')!=PROMPT:raise ValueError('Native policy-only mobile case identity differs')
    acceptance=result['acceptance']
    if acceptance.get('carry_qualified_control_tick')!=tick or not acceptance.get('grasped') or acceptance.get('fell') or acceptance.get('dropped'):raise ValueError('First qualified carry/no drop/fall evidence differs')
    if result.get('trace_sha256')!=sha(paths['trace']) or result.get('runtime_identity_sha256')!=sha(paths['runtime_identity']):raise ValueError('Producer result binding differs')
    if runtime.get('source_joint_names')!=result['source_joint_names'] or runtime.get('wbc_config')!=result['wbc_config'] or runtime.get('frozen_interface_sha256')!=result['frozen_interface_sha256']:raise ValueError('Runtime controller/interface identity differs')
    rows=[json.loads(line) for line in paths['trace'].read_text().splitlines()]
    if len(rows)<=tick or result.get('completed_control_ticks')!=len(rows)-1:raise ValueError('Complete trace does not cover prefix boundary')
    names=result['source_joint_names'];states=[];actions=[];frames=[];q43=[]
    for i in range(tick+1):
        row=rows[i]
        if row['control_tick']!=i or row['physics_steps_since_reset']!=i*4 or row['source_joint_names']!=names:raise ValueError('Prefix state/physical timeline mismatch')
        q=np.asarray(row['joint_positions'],np.float32);state=encode_state(q,names)
        if i==tick:break  # No state T input; no action T+1 or later labels.
        action=np.asarray(rows[i+1]['applied_action32'],np.float32)
        if action.shape!=(32,) or not np.isfinite(action).all():raise ValueError('Invalid applied action32')
        source_action=np.asarray(rows[i+1]['applied_source_action50'],np.float32)
        if source_action.shape!=(50,) or not np.array_equal(source_action,decode_to_source(action,q,names)):raise ValueError('Next-row applied32/source50 mapping differs')
        path=folder/'head_rgb'/f'{i:05d}.png';entry={'path':str(path),'byte_sha256':sha(path),'rgb_sha256':row['head_rgb_sha256'],'control_tick':i,'timestamp_seconds':i/50}
        raw_png(entry);states.append(state);actions.append(action);frames.append(entry);q43.append(q)
    audited_frames=audit.get('raw_rgb_frames')
    if not isinstance(audited_frames,list) or len(audited_frames)!=tick:raise ValueError('Independent audit must bind every supervised raw PNG')
    for entry,declared in zip(frames,audited_frames):
        if any(entry[key]!=declared.get(key) for key in ('path','byte_sha256','rgb_sha256','control_tick','timestamp_seconds')):raise ValueError('Independent audit raw PNG list differs')
    identity=original_identity(source_manifest,norm);output.mkdir(parents=True)
    arrays={'state':np.asarray(states,np.float32),'actions':np.asarray(actions,np.float32),'timestamps':np.arange(tick,dtype=np.float64)/50,'measured_joint_positions43':np.asarray(q43,np.float32)}
    with (output/'prefix.npz').open('xb') as f:np.savez(f,**arrays)
    document={'schema':SCHEMA,'status':'completed','kind':'single_family_train_only_carry_prefix','task_success_teacher':False,'source_task_success':False,'case_id':case['case_id'],'seed':201,'family_id':'native201','train':[0],'validation':[],'test':[],'length':tick,'first_carry_qualified_control_tick':tick,'input_ticks':[0,tick-1],'target_applied_action_rows':[1,tick],'excluded_after_tick':tick,'npz':'prefix.npz','npz_sha256':sha(output/'prefix.npz'),'raw_rgb_frames':frames,'source_joint_names':names,'audit_path':str(audit_path),'audit_sha256':sha(audit_path),'video_origin_diagnostic':diagnostic,'independent_v2_audit':v2,'legacy_v1_audit_passed':False,'independent_v2_audit_passed':True,'source_global_audit_passed':False,'source_global_audit_flag_scope':'legacy_v1_only;independent_v2_all_checks_passed','evidence':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},**identity,'limitation':'One development family trains carry only; no approach/release success labels. Validation and model selection remain original10episodes only; physical task benefit unproved.'}
    write_new(output/'manifest.json',document);return document


def admission(manifest,source_manifest,norm):
    path=Path(manifest).resolve();data=json.loads(path.read_text());identity=original_identity(source_manifest,norm)
    if data.get('schema')!=SCHEMA or data.get('status')!='completed' or data.get('train')!=[0] or data.get('validation') or data.get('test') or data.get('family_id')!='native201':raise ValueError('Completed train-only family201 prefix required')
    if any(data.get(k)!=v for k,v in identity.items()):raise ValueError('Original manifest/norm changed')
    audit_path=Path(data['audit_path'])
    if sha(audit_path)!=data['audit_sha256']:raise ValueError('Independent prefix audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('schema')!=AUDIT_SCHEMA or audit.get('status')!='passed' or audit.get('carry_qualified_control_tick')!=data['length']:raise ValueError('Passed audit boundary mismatch')
    diagnostic=data['video_origin_diagnostic']
    if sha(diagnostic['path'])!=diagnostic['sha256']:raise ValueError('Video origin diagnostic changed')
    for key,item in data['evidence'].items():
        if sha(item['path'])!=item['sha256'] or audit['bind_sha256'].get(key)!=item['sha256']:raise ValueError('Original carry evidence changed:'+key)
    v2=verify_v2(audit,{key:Path(item['path']) for key,item in data['evidence'].items()})
    if data.get('case_id')!=audit.get('case_id') or data.get('seed')!=201 or audit.get('seed')!=201 or data.get('raw_rgb_frames')!=[{key:entry[key] for key in ('path','byte_sha256','rgb_sha256','control_tick','timestamp_seconds')} for entry in audit.get('raw_rgb_frames',[])]:raise ValueError('Manifest case/PNG evidence differs from passed prefix audit')
    for evidence_path,digest in v2['inputs_sha256'].items():
        if sha(evidence_path)!=digest:raise ValueError('Full v2 evidence chain changed:'+evidence_path)
    if data.get('independent_v2_audit')!=v2 or data.get('legacy_v1_audit_passed') is not False or data.get('independent_v2_audit_passed') is not True:raise ValueError('Manifest v2 audit chain differs')
    npz=path.parent/data['npz']
    if sha(npz)!=data['npz_sha256']:raise ValueError('Carry array identity changed')
    return {'manifest_path':str(path),'manifest_sha256':sha(path),'family_id':'native201','train_only':True,'frames':data['length'],'source_task_success':False,**identity,'extra_frozen_files':{**v2['inputs_sha256'],v2['path']:v2['sha256'],diagnostic['path']:diagnostic['sha256'],str(path):sha(path),str(audit_path):sha(audit_path),str(npz):sha(npz),**{v['path']:v['sha256'] for v in data['evidence'].values()},**{entry['path']:entry['byte_sha256'] for entry in data['raw_rgb_frames']}}}


class CarryPrefix:
    def __init__(self,manifest,normalizer):
        path=Path(manifest);data=json.loads(path.read_text());npz=path.parent/data['npz']
        if sha(npz)!=data['npz_sha256']:raise ValueError('Carry array changed')
        with np.load(npz,allow_pickle=False) as values:
            states=values['state'];actions=values['actions'];times=values['timestamps'];q=values['measured_joint_positions43']
        n=data['length']
        if states.shape!=(n,32) or actions.shape!=(n,32) or q.shape!=(n,43) or times.shape!=(n,) or not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(q).all() or not np.allclose(times,np.arange(n)/50,rtol=0,atol=1e-7):raise ValueError('Prefix tensor/timing mismatch')
        if len(data['raw_rgb_frames'])!=n or not np.array_equal(states,np.asarray([encode_state(row,data['source_joint_names']) for row in q])):raise ValueError('Prefix measured state alignment differs')
        self.frames=data['raw_rgb_frames'];self.states=states;self.actions=normalizer.normalize(actions,'actions');self.indices=list(range(n));self.episodes=[{'id':'native201'}]
        for i,entry in enumerate(self.frames):
            if entry['control_tick']!=i or abs(entry['timestamp_seconds']-times[i])>1e-7:raise ValueError('Prefix RGB clock differs')
            raw_png(entry)
    def sample(self,index):
        stop=min(index+50,len(self.actions));count=stop-index
        targets=np.repeat(self.actions[stop-1:stop],50,axis=0);targets[:count]=self.actions[index:stop]
        return raw_png(self.frames[index]),self.states[index],targets,np.arange(50)<count


class CarryMixture:
    def __init__(self,original,prefix):
        self.sources=[original,prefix];self.indices=range(len(original.indices)+len(prefix.indices));self.draw_counts={'original':0,'carry_prefix':0}
    def sample(self,index):
        source=0 if random.random()<.6 else 1;name=('original','carry_prefix')[source];self.draw_counts[name]+=1
        data=self.sources[source];return data.sample(random.randrange(len(data.indices)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case-dir','audit','source-manifest','norm-stats','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();result=prepare(a.case_dir,a.audit,a.source_manifest,a.norm_stats,a.output);print(json.dumps({'status':result['status'],'length':result['length'],'task_success_teacher':False}))

if __name__=='__main__':main()
