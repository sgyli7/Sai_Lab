"""Audit fixed mobile demonstrations before adapter/training approval. CPU only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pyarrow.parquet as pq
from experiments.pi05_g1 import contract
from experiments.pi05_g1.prepare_training_data import probe_video,check_video_alignment,grouped_split,fit_norm
from .download_dataset import DEST,REV,REPO,sha

ROOT=Path(__file__).resolve().parents[2]
OUT=DEST.parent/'prepared'

def save(path,value):
    with path.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    receipt=json.loads((DEST/'download_receipt.json').read_text())
    if receipt.get('status')!='complete' or receipt.get('revision')!=REV:raise ValueError('Complete fixed-revision verified download required')
    for file in receipt['files']:
        p=DEST/file['path']
        if p.stat().st_size!=file['bytes'] or sha(p)!=file['sha256']:raise ValueError('Download changed: '+str(p))
    if OUT.exists():raise FileExistsError('Preserve previous prepared output')
    OUT.mkdir()
    root=DEST/'lerobot';info=json.loads((root/'meta/info.json').read_text());modality=json.loads((root/'meta/modality.json').read_text())
    names=info['features']['observation.state']['names'];action_names=info['features']['action']['names']
    if set(names)!=set(contract.DATASET_NAMES) or set(action_names)!=set(contract.DATASET_NAMES):raise ValueError('Joint name set mismatch')
    if info['fps']!=50:raise ValueError('Frozen50Hz expected')
    meta={e['episode_index']:e for e in map(json.loads,(root/'meta/episodes.jsonl').read_text().splitlines())}
    records=[];rejected=[];arrays={};navstats={};rawdata={}
    for p in sorted(root.glob('data/**/*.parquet')):
        eid=int(p.stem.split('_')[-1]);rows=pq.read_table(p).to_pylist();n=len(rows);errors=[]
        times=np.array([r['timestamp'] for r in rows],dtype=np.float64)
        numerical=('observation.state','action','observation.eef_pose','action.eef_pose','teleop.base_height_command','teleop.navigate_command','teleop.torso_orientation_rpy_command','observation.img_state_delta')
        for key in numerical:
            a=np.array([r[key] for r in rows],dtype=np.float64)
            if not np.isfinite(a).all():errors.append('nonfinite '+key)
        data={'state':np.stack([contract.encode_state(r['observation.state'],names) for r in rows]),
              'actions':np.stack([contract.encode_action(r['action'],r['teleop.base_height_command'],r['teleop.navigate_command'],action_names) for r in rows])}
        # Whole source trajectory identity includes every numeric control/observation column,
        # normalized time origin, language and reward/done; episode/index bookkeeping excluded.
        h=hashlib.sha256()
        for key in sorted(set(rows[0])-{'episode_index','index','frame_index','timestamp'}):
            h.update(key.encode());h.update(json.dumps([r[key] for r in rows],separators=(',',':'),allow_nan=False).encode())
        h.update((times-times[0]).astype('<f8').tobytes())
        video=root/info['video_path'].format(episode_chunk=eid//info.get('chunks_size',1000),video_key='observation.images.ego_view',episode_index=eid)
        details=probe_video(video)
        try:check_video_alignment(times,details['frames'],details['fps'])
        except ValueError as e:errors.append(str(e))
        delta=np.array([r['observation.img_state_delta'] for r in rows]);duplicate=len(times)-len(np.unique(times))
        if duplicate or np.any(np.diff(times)<=0):errors.append('nonmonotonic/duplicate timestamps')
        if np.max(np.abs(delta))>.0100001:errors.append('stored image/state delta exceeds halfcontrolframe')
        if n!=meta[eid]['length']:errors.append('metadata length mismatch')
        rec={'episode_id':eid,'parquet':str(p.relative_to(root)),'parquet_sha256':sha(p),'video_sha256':{'observation.images.ego_view':sha(video)},'videos':{'observation.images.ego_view':{'path':str(video.relative_to(root)),**details}},'length':n,'metadata_length':meta[eid]['length'],'content_sha256':h.hexdigest(),'errors':errors,'duplicate_timestamps':duplicate,'image_state_delta':{'min':float(delta.min()),'max':float(delta.max()),'nonzero_count':int(np.count_nonzero(delta))},'success_labels':{k:v for k,v in meta[eid].items() if 'success' in k},'terminal_signals':{k:rows[-1][k] for k in ['next.reward','next.done'] if k in rows[-1]}}
        (rejected if errors else records).append(rec)
        if not errors:arrays[eid]=data;rawdata[eid]=np.array([r['teleop.navigate_command'] for r in rows])
        if (eid+1)%10==0:print(json.dumps({'audited_episodes':eid+1,'accepted':len(records),'rejected':len(rejected)}),flush=True)
    # Deduplicate whole trajectories: connected data-content OR full-video identity.
    parent={r['episode_id']:r['episode_id'] for r in records};seen={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in records:
        eid=r['episode_id']
        for key in [('data',r['content_sha256'])]+[('video',v) for v in r['video_sha256'].values()]:
            if key in seen:parent[find(eid)]=find(seen[key])
            else:seen[key]=eid
    groups={}
    for eid in sorted(parent):groups.setdefault(find(eid),[]).append(eid)
    unique_ids=sorted(min(g) for g in groups.values());unique_records=[r for r in records if r['episode_id'] in unique_ids]
    report={'schema':'pi05_g1_round03_mobile_data_audit_v1','repo_id':REPO,'revision':REV,'actual_parquet_files':len(records)+len(rejected),'actual_complete_episodes':len(records),'unique_complete_trajectories':len(unique_ids),'duplicate_groups':[v for v in groups.values() if len(v)>1],'duplicate_excluded_ids':sorted(set(parent)-set(unique_ids)),'rejected':rejected,'complete':records,'metadata_total_episodes':info['total_episodes'],'metadata_total_frames':info['total_frames'],'actual_complete_frames':sum(r['length'] for r in records),'readme_conflicts':['50trajectories vs100filepairs/metadata','26DoF vs43namedjoints','256square vs480x640'],'source_alignment_evidence':{'decoded_video_pts_vs_timestamps':'sameframecount;50Hz;zeroorigin half-frame max deviation','stored_image_state_delta':'values checked only; converter hardcodes zeros, so not independent physical synchronization evidence','action_semantics':'Published converter obs.robot_joint_pos[:-1], processed_actions[:-1], action teleop[:-1], camera frames[:-1] share row; timestamps are generated arange/50. No next-row shift applied. This is export convention, not verified physical actuator latency.','meta_modality':modality,'state_joint_names':names,'action_joint_names':action_names,'adapter_is_provisional_until_root_approval':True,'waist':'Preserved measured state; policy output32 excludes waist because published Homie lower controller owns it; decoder35 not proof35executed'},'converter_source_evidence':{'path':str(ROOT/'upstream/unitree_g1/isaaclab_arena/isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py'),'sha256':sha(ROOT/'upstream/unitree_g1/isaaclab_arena/isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py'),'joint_pairing_lines':[331,398],'teleop_pairing_lines':[208,215],'video_pairing_line':540,'synthetic_terminal_and_delta_lines':[438,448],'terminal_reward_done_not_independent_success_label':True},'physical_distance_evidence':'Inspected LeRobot schema lacks world root/object/contact. Nonzero nav is command evidence only; no autonomous distance or carrying duration inferred.'}
    save(OUT/'preparation_audit.json',report)
    if rejected:raise ValueError('Data alignment/validity rejections require explicit review; audit saved, no final splits/stats')
    split=grouped_split(unique_records,seed=42)
    manifest={'revision':REV,'repo_id':REPO,'seed':42,'dataset_root':str(root),**split,'metadata_sha256':{str(p.relative_to(root)):sha(p) for p in sorted((root/'meta').glob('*')) if p.is_file()},'readme_sha256':sha(DEST/'README.md'),'files':unique_records,'actual_ratios':{k:len(v)/len(unique_ids) for k,v in split.items()},'deduplication':{'whole_source_trajectory_or_full_video_hash':'connectedgroups;lowestepisodeid representative','duplicate_groups':[v for v in groups.values() if len(v)>1],'excluded_episode_ids':report['duplicate_excluded_ids']},'adapter_approval_status':'awaiting_root_training_adapter_confirmation'}
    save(OUT/'split_manifest.json',manifest)
    stats=fit_norm(arrays,split['train'])
    for key in ['state','actions']:
        a=np.concatenate([arrays[e][key].astype(np.float64) for e in split['train']]);stats[key]['min']=a.min(0).tolist();stats[key]['max']=a.max(0).tolist()
    save(OUT/'train_norm_stats.json',stats)
    for name,ids in split.items():
        nav=np.concatenate([rawdata[e] for e in ids]);navstats[name]={'episodes':len(ids),'frames':len(nav),'nonzero_per_axis':np.count_nonzero(nav,axis=0).tolist(),'min':nav.min(0).tolist(),'max':nav.max(0).tolist()}
    save(OUT/'adapter_diagnostics.json',{'schema':'pi05_g1_round03_mobile_adapter_diagnostics_v1','normalization_source':'train_only','split_nav':navstats,'state_names':list(contract.STATE_NAMES),'action_names':list(contract.ACTION_NAMES),'waist_action_excluded_lower_controller_owned':True,'raw_extrema_saved_as':'state/actions.min,max','contract_file_sha256':sha(Path(contract.__file__)),'training_adapter_approved':False})
    print(json.dumps({'status':'audited_prepared_pending_adapter_approval','unique_episodes':len(unique_ids),'split_counts':{k:len(v) for k,v in split.items()},'output':str(OUT)}),flush=True)

if __name__=='__main__':main()
