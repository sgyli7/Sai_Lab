"""Optional long-transport teacher candidate. No teacher exists by implication.

A root-frozen plan is mandatory. Conversion requires actual audited autonomous
physical success, exact raw PNGs and next-row applied actions. Partial collections
produce a partial manifest, never an enabled training source or union norm.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import PROMPT,encode_row

SCHEMA='pi05_g1_mobile_supplementary_v1'
CONFIG_FIELDS=('scene','assets','source_harness_dir','geometry_definition_sha256','robot_geometry_sha256','prompt','policy_id','native_action_horizon','replan_interval_ticks')
RESULT_FIELDS=('wbc_config','source_joint_names','action_limits_sha256','frozen_interface_sha256')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def declared_splits(cases):
    """Seed42 stratified round-robin family allocation, independent of outcome.

Ten independent cases =>8/1/1. Shared replay ancestry is one family; approximate
ratios count families, never variants. All family members stay in one split.
"""
    families={}
    for case in cases:
        family=case['family_id'];stratum=case['stratum']
        if not isinstance(family,str) or not family or not isinstance(stratum,str) or not stratum:raise ValueError('Explicit family_id and stratum required')
        families.setdefault(family,[]).append(case)
    if len(families)<3:raise ValueError('At least3 independent families required for train/validation/test')
    buckets={}
    for family,items in sorted(families.items()):
        if len({x['stratum'] for x in items})!=1:raise ValueError('Family cannot change stratum')
        buckets.setdefault(items[0]['stratum'],[]).append(family)
    rng=random.Random(42)
    for bucket in buckets.values():rng.shuffle(bucket)
    strata=sorted(buckets);rng.shuffle(strata);ordered=[]
    while any(buckets.values()):
        for key in strata:
            if buckets[key]:ordered.append(buckets[key].pop())
    n=len(ordered);ntest=max(1,round(n*.1));nval=max(1,round(n*.1));ntrain=n-ntest-nval
    if ntrain<1:raise ValueError('Insufficient train families')
    allocations={f:('train' if i<ntrain else 'validation' if i<ntrain+nval else 'test') for i,f in enumerate(ordered)}
    return {x['case_id']:allocations[x['family_id']] for x in cases}


def validate_plan(plan):
    if plan.get('schema')!='pi05_g1_mobile_teacher_plan_v1' or plan.get('decision_frozen_before_collection') is not True or plan.get('split_seed')!=42:
        raise ValueError('Root-frozen teacher plan with split_seed42 required')
    if type(plan.get('declared_before_wall_time_ns')) is not int:raise ValueError('Precollection timestamp required')
    cases=plan['cases'];ids=[x['case_id'] for x in cases]
    if len(set(ids))!=len(ids) or not cases:raise ValueError('Nonempty unique predeclared IDs required')
    forbidden=set(plan['formal_case_ids'])
    for case in cases:
        if type(case['seed']) is not int or case['seed'] in range(1040,1044) or case['case_id'] in forbidden or 'formal' in case['case_id'].lower():raise ValueError('Formal evaluation data forbidden')
        # Replay/time-stretch variants must disclose a shared root family. A
        # replay-derived trajectory is never admitted as native policy-only data.
        if case.get('replay_origin_id') is not None and case['family_id']!=case['replay_origin_id']:raise ValueError('Replay variants must retain origin family')
    expected=declared_splits(cases)
    if any(case['split']!=expected[case['case_id']] for case in cases):raise ValueError('Declared splits differ from frozen seed42 family allocation')
    for key in CONFIG_FIELDS:
        if key not in plan['expected_config']:raise ValueError('Missing same-environment config identity:'+key)
    for key in RESULT_FIELDS:
        if key not in plan['expected_result']:raise ValueError('Missing same-controller/source identity:'+key)
    if not plan.get('expected_frozen_files'):raise ValueError('Pinned source/head-camera/controller files required')
    categories=plan.get('frozen_file_roles',{})
    for role in ('head_camera','controller','source'):
        if not categories.get(role) or any(path not in plan['expected_frozen_files'] for path in categories[role]):raise ValueError('Pinned file role missing:'+role)
    return expected


def read_png(path,expected):
    with Image.open(path) as image:
        if image.format!='PNG' or image.mode!='RGB':raise ValueError('Lossless raw RGB PNG required')
        rgb=np.asarray(image).copy()
    digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    if rgb.shape!=(480,640,3) or digest!=expected:raise ValueError('Raw head pixels differ from synchronized row')
    return {'path':str(path.resolve()),'byte_sha256':sha(path),'rgb_sha256':digest,'rgb_shape':[480,640,3]}


def convert(results_root,plan_path,source_manifest,output,*,auditor=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh conversion output required')
    output.mkdir(parents=True);plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());split=validate_plan(plan)
    source=json.loads(Path(source_manifest).read_text())
    if [len(source[k]) for k in ('train','validation','test')]!=[80,10,10]:raise ValueError('Original100-source split must remain80/10/10')
    if auditor is None:
        from .acceptance import evaluate_episode
        auditor=evaluate_episode
    # Reuse exact established next-row/source50 mapping checks; no truth enters
    # samples. Physical truth is read only by the independent acceptance auditor.
    from experiments.pi05_g1_round02.teacher_dataset import extract_samples
    records=[];rejected=[];contents={};pixel_sequences={}
    for eid,case in enumerate(plan['cases']):
        folder=Path(results_root)/case['case_id'];base={'episode_id':eid,'source_case':case['case_id'],'family_id':case['family_id'],'stratum':case['stratum'],'split':split[case['case_id']]}
        try:
            result_path=folder/'result.json';config_path=folder/'config.json';trace_path=folder/'trajectory.jsonl'
            result=json.loads(result_path.read_text());config=json.loads(config_path.read_text())
            if result.get('status')!='completed' or result.get('success') is not True:raise ValueError('No completed qualified success')
            if config.get('mode')!='policy' or config.get('policy_id')!='gr00t_n16' or result.get('policy')!='gr00t_n16':raise ValueError('Native policy-only collection required')
            if case.get('replay_origin_id') is not None:raise ValueError('Replay-derived teacher is not autonomous native policy-only evidence')
            if config.get('case')!={'case_id':case['case_id'],'seed':case['seed']} or result.get('case')!=config['case']:raise ValueError('Case/seed differs from predeclared identity')
            if config.get('prompt')!=PROMPT:raise ValueError('Mobile prompt differs')
            for key in CONFIG_FIELDS:
                if config.get(key)!=plan['expected_config'][key]:raise ValueError('Environment identity mismatch:'+key)
            for key in RESULT_FIELDS:
                if result.get(key)!=plan['expected_result'][key]:raise ValueError('Controller/source identity mismatch:'+key)
            if any(config.get('frozen_files',{}).get(path)!=digest or sha(path)!=digest for path,digest in plan['expected_frozen_files'].items()):raise ValueError('Pinned head/controller/source changed')
            if result.get('trace_sha256')!=sha(trace_path):raise ValueError('Trace/result binding differs')
            rows=[json.loads(line) for line in trace_path.read_text().splitlines()]
            if not rows or rows[0]['wall_time_ns']<=plan['declared_before_wall_time_ns']:raise ValueError('Plan not frozen before collection')
            if result.get('completed_control_ticks')!=len(rows)-1:raise ValueError('Trace not fully completed')
            audit=auditor(rows,{**config,'audit_body_names':result['body_names'],'audit_contact_metadata':result['contact_view']})
            if audit.get('success') is not True or audit.get('policy_external_interventions')!=0:raise ValueError('Independent physical acceptance failed')
            samples=extract_samples(rows,result['source_joint_names']);frames=[];visual=hashlib.sha256()
            for tick,row in enumerate(rows):
                if row.get('source_joint_names')!=result['source_joint_names']:raise ValueError('Named measured state differs')
                png=folder/'head_rgb'/f'{tick:05d}.png';entry=read_png(png,row['head_rgb_sha256']);visual.update(bytes.fromhex(entry['rgb_sha256']))
                if tick<len(samples['state']):frames.append({**entry,'control_tick':tick,'timestamp_seconds':tick/50})
            content=hashlib.sha256()
            for key in ('state','actions','timestamps'):content.update(samples[key].tobytes())
            digest=content.hexdigest();visual_digest=visual.hexdigest()
            if digest in contents or visual_digest in pixel_sequences:raise ValueError('Duplicate full state/action/time or visual trajectory, cannot count independent variant')
            contents[digest]=case['family_id'];pixel_sequences[visual_digest]=case['family_id']
            npz=output/f'episode_{eid:06d}.npz'
            with npz.open('xb') as f:np.savez(f,**samples)
            records.append({**base,'npz':npz.name,'npz_sha256':sha(npz),'length':len(samples['state']),'raw_rgb_frames':frames,'content_sha256':digest,'raw_visual_sequence_sha256':visual_digest,'result_path':str(result_path.resolve()),'result_sha256':sha(result_path),'config_path':str(config_path.resolve()),'config_sha256':sha(config_path),'trace_path':str(trace_path.resolve()),'trace_sha256':sha(trace_path),'physical_audit':audit,'last_unlabeled_state_dropped':True})
        except (ValueError,KeyError,OSError,TypeError) as error:rejected.append({**base,'reason':repr(error)})
    complete=len(records)==len(plan['cases']) and not rejected
    manifest={'schema':SCHEMA,'status':'completed' if complete else 'partial','fresh_source_enabled':complete,'episodes':records,'rejected':rejected,'nominal_cases':plan['cases'],'plan_path':str(plan_path.resolve()),'plan_sha256':sha(plan_path),'source_manifest_sha256':sha(source_manifest),'split_seed':42,'alignment':'head/state(t) to applied_action32 in row(t+1),50frame horizon,repeat tail with loss mask','family_grouping':'All declared replay ancestry variants retain one family; replay-derived traces rejected for policy-only admission','source_original_split_unchanged':True,**{k:[x['episode_id'] for x in records if x['split']==k] for k in ('train','validation','test')}}
    write_new(output/'manifest.json',manifest);return manifest


class SupplementaryEpisodes:
    """Raw PNG loader reuses validated teacher sampling, without MP4 fallback."""
    def __init__(self,manifest,split,normalizer):
        from experiments.pi05_g1_round02.datasets import TeacherEpisodes
        path=Path(manifest);data=json.loads(path.read_text())
        if data.get('schema')!=SCHEMA or data.get('status')!='completed' or not data.get('fresh_source_enabled'):raise ValueError('All predeclared teacher splits must be complete')
        if split not in ('train','validation','test'):raise ValueError('Unknown split')
        records={x['episode_id']:x for x in data['episodes']};self.episodes=[];self.indices=[];self.normalizer=normalizer
        sets=[set(data[k]) for k in ('train','validation','test')]
        if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('Supplementary split overlap')
        family_splits={}
        for key in ('train','validation','test'):
            for identifier in data[key]:
                family=records[identifier]['family_id']
                if family in family_splits and family_splits[family]!=key:raise ValueError('Supplementary family leakage')
                family_splits[family]=key
        for identifier in data[split]:
            item=records[identifier];npz=path.parent/item['npz']
            for kind in ('result','config','trace'):
                if sha(item[kind+'_path'])!=item[kind+'_sha256']:raise ValueError('Teacher provenance payload changed:'+kind)
            if sha(npz)!=item['npz_sha256']:raise ValueError('Teacher arrays changed')
            with np.load(npz,allow_pickle=False) as payload:
                states=payload['state'];actions=payload['actions'];times=payload['timestamps'];n=item['length']
            if states.shape!=(n,32) or actions.shape!=(n,32) or times.shape!=(n,) or not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.allclose(times,np.arange(n)/50,rtol=0,atol=1e-7):raise ValueError('Invalid teacher dimensions/finite/timing')
            frames=item['raw_rgb_frames']
            if len(frames)!=n:raise ValueError('Every supervised sample needs its original PNG')
            for tick,entry in enumerate(frames):
                if entry['control_tick']!=tick or abs(entry['timestamp_seconds']-times[tick])>1e-7:raise ValueError('Teacher frame alignment mismatch')
                TeacherEpisodes.read_raw_rgb(entry)
            self.episodes.append({'id':identifier,'states':states,'actions':normalizer.normalize(actions,'actions'),'timestamps':times,'raw_rgb_frames':frames,'video':None})
            self.indices.extend((len(self.episodes)-1,i) for i in range(n))
        if not self.episodes:raise ValueError('Empty declared supplementary split')
        self.read_raw_rgb=TeacherEpisodes.read_raw_rgb
    def sample(self,index):
        from experiments.pi05_g1_round02.datasets import TeacherEpisodes
        return TeacherEpisodes.sample(self,index)
    def validation_indices(self):return [i for i,(_,tick) in enumerate(self.indices) if tick%50==0]


def union_train_norm(source_manifest,supplementary_manifest,output):
    """Read only original TRAIN parquet and supplementary TRAIN arrays. No RGB."""
    source_path=Path(source_manifest);source=json.loads(source_path.read_text());fresh_path=Path(supplementary_manifest);fresh=json.loads(fresh_path.read_text())
    if fresh.get('status')!='completed' or fresh.get('fresh_source_enabled') is not True or fresh['source_manifest_sha256']!=sha(source_path):raise ValueError('Incomplete teacher or original identity mismatch')
    if [len(source[k]) for k in ('train','validation','test')]!=[80,10,10]:raise ValueError('Original split changed')
    import pyarrow.parquet as pq
    states=[];actions=[];files={x['episode_id']:x for x in source['files']};dataset=Path(source['dataset_root']);receipts=[]
    for identifier in source['train']:
        item=files[identifier];path=Path(item['parquet']);path=path if path.is_absolute() else dataset/path
        if sha(path)!=item['parquet_sha256']:raise ValueError('Original train parquet changed')
        values=[encode_row(row) for row in pq.read_table(path).to_pylist()]
        states.append(np.asarray([x['state'] for x in values],np.float32));actions.append(np.asarray([x['actions'] for x in values],np.float32));receipts.append({'source':'original','episode_id':identifier,'sha256':sha(path)})
    # Load only NPZ train data, not the image loader or validation/test payloads.
    records={x['episode_id']:x for x in fresh['episodes']}
    for identifier in fresh['train']:
        item=records[identifier];path=fresh_path.parent/item['npz']
        if sha(path)!=item['npz_sha256']:raise ValueError('Supplementary train changed')
        with np.load(path,allow_pickle=False) as payload:states.append(payload['state']);actions.append(payload['actions'])
        receipts.append({'source':'supplementary','episode_id':identifier,'sha256':sha(path)})
    stats={}
    for key,chunks in [('state',states),('actions',actions)]:
        values=np.concatenate(chunks)
        if values.shape[1]!=32 or not np.isfinite(values).all():raise ValueError('Invalid union train values')
        stats[key]={name:value.tolist() for name,value in [('mean',values.mean(0)),('std',values.std(0)),('q01',np.quantile(values,.01,axis=0)),('q99',np.quantile(values,.99,axis=0))]}
    write_new(output,stats);write_new(str(output)+'.receipt.json',{'schema':'pi05_g1_mobile_union_train_norm_v1','source_manifest_sha256':sha(source_path),'supplementary_manifest_sha256':sha(fresh_path),'norm_sha256':sha(output),'frames':sum(len(x) for x in states),'training_payloads':receipts,'validation_or_test_payloads_opened':False,'norm_policy':'All union TRAIN frames only, frame-weighted empirical quantiles; no val/test statistics'})
    return stats



class MixedTraining:
    """Original global-frame uniform vs fresh global-frame uniform,50/50."""
    def __init__(self,original,fresh):
        self.sources=[original,fresh];self.indices=list(range(len(original.indices)+len(fresh.indices)))
        self.draw_counts={'original':0,'supplementary':0}
    def sample(self,index):
        # The old runner's uniform index draw is intentionally consumed but not
        # used; an independent source draw then frame draw defines50/50 exactly.
        source=random.randrange(2);name=('original','supplementary')[source]
        self.draw_counts[name]+=1;data=self.sources[source]
        return data.sample(random.randrange(len(data.indices)))


def mixed_validate(runtime,sources,weights,deadline,validate_fn):
    values={}
    for name,data in sources.items():
        value=validate_fn(runtime,data,weights,deadline)
        if value is None:return None
        if value.get('complete') is not True:raise ValueError('Incomplete source validation cannot select checkpoint')
        values[name]=value
    if set(values)!={'original','supplementary'}:raise ValueError('Exactly two complete validation sources required')
    return {'flow_loss':sum(x['flow_loss'] for x in values.values())*.5,
            'complete':True,'frames':sum(x['frames'] for x in values.values()),
            'chunks':sum(x['chunks'] for x in values.values()),'episodes':sum(x['episodes'] for x in values.values()),
            'source_metrics':values,'source_weights':{'original':.5,'supplementary':.5},'noise_seed':42}


def supplementary_admission(manifest,source_manifest,norm):
    path=Path(manifest).resolve();fresh=json.loads(path.read_text());receipt_path=Path(str(norm)+'.receipt.json');receipt=json.loads(receipt_path.read_text())
    if fresh.get('schema')!=SCHEMA or fresh.get('status')!='completed' or fresh.get('fresh_source_enabled') is not True:raise ValueError('No fully qualified supplementary source')
    if receipt.get('schema')!='pi05_g1_mobile_union_train_norm_v1' or receipt.get('validation_or_test_payloads_opened') is not False:raise ValueError('Union train-only norm receipt required')
    if receipt['source_manifest_sha256']!=sha(source_manifest) or fresh['source_manifest_sha256']!=sha(source_manifest) or receipt['supplementary_manifest_sha256']!=sha(path) or receipt['norm_sha256']!=sha(norm):raise ValueError('Union norm/source identity mismatch')
    records={x['episode_id']:x for x in fresh['episodes']}
    if not all(fresh.get(split) for split in ('train','validation','test')):raise ValueError('All predeclared splits required')
    return {'manifest':str(path),'manifest_sha256':sha(path),'norm_receipt':str(receipt_path.resolve()),'norm_receipt_sha256':sha(receipt_path),'train_episodes':len(fresh['train']),'validation_episodes':len(fresh['validation']),'validation_chunks':sum((records[i]['length']+49)//50 for i in fresh['validation']),'test_opened':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    convert_parser=sub.add_parser('convert')
    for key in ('results-root','plan','source-manifest','output'):convert_parser.add_argument('--'+key,required=True)
    norm_parser=sub.add_parser('norm')
    for key in ('source-manifest','supplementary-manifest','output'):norm_parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if args.command=='convert':result=convert(args.results_root,args.plan,args.source_manifest,args.output);print(json.dumps({'status':result['status'],'fresh_source_enabled':result['fresh_source_enabled']}));return 0 if result['fresh_source_enabled'] else 1
    union_train_norm(args.source_manifest,args.supplementary_manifest,args.output);return 0

if __name__=='__main__':raise SystemExit(main())
