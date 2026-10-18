"""Convert only routed, successful teacher episodes into raw 32D supervised samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import av
from PIL import Image
from experiments.pi05_g1.contract import encode_state,decode_to_source
from .prepare_inputs import sha,write_once


def extract_samples(rows,names):
    if len(rows)<2:raise ValueError('Teacher episode has no next-row action labels')
    states=[];actions=[];timestamps=[]
    for i,row in enumerate(rows):
        if row['control_tick']!=i or row['physics_steps_since_reset']!=i*4:raise ValueError('Teacher trace has gaps/resets or wrong physical timing')
        if i==len(rows)-1:break  # No next action exists: never fabricate the last target.
        state=encode_state(row['joint_positions'],names)
        action=np.asarray(rows[i+1]['applied_action32'],np.float32)
        if action.shape!=(32,) or not np.isfinite(action).all():raise ValueError('Teacher next-row action32 invalid')
        source_action=np.asarray(rows[i+1].get('applied_source_action50'),np.float32)
        if source_action.shape!=(50,) or not np.array_equal(source_action,decode_to_source(action,row['joint_positions'],names)):
            raise ValueError('Teacher action32 differs from actual applied source50 command')
        states.append(state);actions.append(action);timestamps.append(i/50.)
    return {'state':np.asarray(states,np.float32),'actions':np.asarray(actions,np.float32),
            'timestamps':np.asarray(timestamps,np.float64)}


def duplicate_groups(records):
    parent={r['episode_id']:r['episode_id'] for r in records}
    def find(x):
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    seen={}
    for r in records:
        for key in (('content',r['content_sha256']),('video',r['video_sha256'])):
            if key in seen:parent[find(r['episode_id'])]=find(seen[key])
            else:seen[key]=r['episode_id']
    groups={}
    for r in records:groups.setdefault(find(r['episode_id']),[]).append(r)
    return list(groups.values())


def convert(results_root,cases_path,output,acceptance_path,routing_path):
    root=Path(results_root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cases_doc=json.loads(Path(cases_path).read_text());cases=cases_doc['cases'];routing=json.loads(Path(routing_path).read_text())
    if routing.get('selected_teacher_cases_sha256')!=sha(cases_path) or routing.get('decision_frozen_before_collection') is not True:
        raise ValueError('Teacher family must be routed and frozen before collection')
    expected_hash=json.loads(Path(acceptance_path).read_text())['acceptance_sha256']
    if any(c['seed'] in range(40,44) for c in cases):raise ValueError('Final evaluation seeds are forbidden in teacher data')
    train=set(cases_doc['train_case_ids']);val=set(cases_doc['validation_case_ids'])
    if train&val or (train|val)!={r['case_id'] for r in cases}:raise ValueError('Teacher nominal splits must partition exactly the routed cases')
    collected=[];rejected=[]
    for eid,case in enumerate(cases):
        cid=case['case_id'];folder=root/cid;rp=folder/'result.json'
        split='train' if cid in train else 'validation'
        base={'episode_id':eid,'source_case':cid,'split':split}
        try:
            if not rp.is_file():raise ValueError('not_collected')
            result=json.loads(rp.read_text())
            if result.get('case')!=case:raise ValueError('Result case differs from frozen teacher case')
            if result.get('status')!='completed' or result.get('success') is not True:raise ValueError('teacher_task_failed_or_infrastructure_incomplete')
            if result.get('acceptance_version')!='pi05_g1_placement_v2' or result.get('acceptance_sha256')!=expected_hash:
                raise ValueError('Teacher acceptance differs from frozen v2')
            rows=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
            names=result.get('source_joint_names',result.get('frozen_interface',{}).get('source_joint_names'))
            if names is None:raise ValueError('Actual source joint names missing')
            samples=extract_samples(rows,names)
            grasp=result.get('first_grasp_control_tick');candidate=result.get('source_success_candidate_tick');window=result.get('qualified_stability_window')
            if (type(grasp) is not int or type(candidate) is not int or not 0<=grasp<candidate<len(rows)
                    or not isinstance(window,dict) or window.get('passed') is not True
                    or type(window.get('end_tick')) is not int or not candidate<=window['end_tick']<len(rows)):
                raise ValueError('Successful teacher phase events are absent or inconsistent with trace coverage')
            visual=hashlib.sha256();raw_frames=[]
            for row in rows:
                relative=row.get('teacher_head_rgb_path')
                if not relative:raise ValueError('Full-rate teacher head PNG evidence missing')
                path=Path(relative);path=path if path.is_absolute() else folder/path
                rgb=np.asarray(Image.open(path))
                digest=hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgb.dtype!=np.uint8 or rgb.shape!=(480,640,3) or digest!=row['head_rgb_sha256']:
                    raise ValueError('Teacher raw PNG/state trace correspondence invalid')
                visual.update(bytes.fromhex(digest))
                if row['control_tick']<len(samples['state']):
                    raw_frames.append({'path':str(path.resolve()),'byte_sha256':sha(path),'rgb_sha256':digest,
                                       'rgb_shape':list(rgb.shape),'control_tick':row['control_tick'],
                                       'timestamp_seconds':row['control_tick']/50.})
            video=Path(result['video_path']);video=video if video.is_absolute() else folder/video
            with av.open(str(video)) as container:
                stream=container.streams.video[0];fps=float(stream.average_rate);pts=[float(f.pts*f.time_base) for f in container.decode(stream)]
            if fps!=50 or len(pts)!=len(rows) or not np.allclose(pts,np.arange(len(rows))/50.,atol=1e-7,rtol=0):raise ValueError('Teacher original video timing differs from physical trace')
            video_sha=sha(video);content=hashlib.sha256()
            for key in ('state','actions','timestamps'):content.update(samples[key].tobytes())
            content.update(visual.digest())
            collected.append({**base,'samples':samples,'video':str(video),'video_sha256':video_sha,'content_sha256':content.hexdigest(),
               'raw_rgb_frames':raw_frames,'source_success_candidate_tick':candidate,'first_grasp_control_tick':grasp,
               'qualified_stability_window':window,'raw_visual_sequence_sha256':visual.hexdigest(),'length':len(samples['state']),'original_video_frames':len(rows),
               'result_sha256':sha(rp),'trace_sha256':sha(folder/'trajectory.jsonl'),'acceptance_sha256':expected_hash})
        except (ValueError,KeyError,OSError) as exc:
            rejected.append({**base,'reason':str(exc),'result_path':str(rp)})
    kept=[]
    for group in duplicate_groups(collected):
        if len({r['split'] for r in group})>1:
            rejected.extend({k:r[k] for k in ('episode_id','source_case','split')}|{'reason':'duplicate_cross_split_group','group_episode_ids':[v['episode_id'] for v in group]} for r in group)
        else:
            group.sort(key=lambda r:r['episode_id']);kept.append(group[0])
            rejected.extend({k:r[k] for k in ('episode_id','source_case','split')}|{'reason':'duplicate_same_split','retained_episode_id':group[0]['episode_id']} for r in group[1:])
    records=[]
    for r in sorted(kept,key=lambda x:x['episode_id']):
        path=output/f"episode_{r['episode_id']:06d}.npz";samples=r.pop('samples')
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                if set(old.files)!=set(samples) or any(not np.array_equal(old[k],v) for k,v in samples.items()):raise FileExistsError(path)
        else:
            with path.open('xb') as stream:np.savez(stream,**samples)
        records.append({**r,'npz':path.name,'npz_sha256':sha(path),'fps':50,
                        'source50_mapping_validated':True,'last_unlabeled_state_dropped':True,'video_last_frame_not_supervised':True})
    split_ids={key:[r['episode_id'] for r in records if r['split']==key] for key in ('train','validation')}
    manifest={'schema':'pi05_g1_teacher_episodes_v1',**split_ids,'test':[],'episodes':records,'rejected':rejected,
       'nominal_train_case_ids':sorted(train),'nominal_validation_case_ids':sorted(val),'complete_9_train_3_validation':len(split_ids['train'])==9 and len(split_ids['validation'])==3,
       'fresh_source_enabled':bool(split_ids['train'] and split_ids['validation']),
       'selected_teacher_cases_sha256':sha(cases_path),'routing_receipt_sha256':sha(routing_path),
       'acceptance_sha256':expected_hash,'alignment':'state row k + actually applied action32 recorded in row k+1; row timestamps k/50; final state omitted without fake label',
       'duplicate_policy':'Full state/action/time/raw visual trajectory or complete video SHA grouped; cross-split groups entirely rejected. Different visual scene with identical joint targets is not a duplicate.'}
    write_once(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('results-root','cases','output','acceptance-frozen','routing-receipt'):parser.add_argument('--'+flag,type=Path,required=True)
    a=parser.parse_args();m=convert(a.results_root,a.cases,a.output,a.acceptance_frozen,a.routing_receipt)
    print(json.dumps({k:m[k] for k in ('train','validation','fresh_source_enabled','complete_9_train_3_validation')},indent=2))
