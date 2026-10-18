"""Versioned three-or-four-policy comparison on twenty frozen trials at five positions."""
from __future__ import annotations
import argparse
from itertools import combinations,product
import json
import math
from pathlib import Path
import re
import av
import numpy as np
from experiments.pi05_g1.summarize_evaluation import (timing_summary,distribution,wilson,
    paired_statistics,clustered_statistics)
from .prepare_inputs import sha
from experiments.pi05_g1.placement import acceptance_identity

VIDEO_KEYS=('head_raw','observer_raw','head_wall_annotated','observer_wall_annotated')
FLAGS=('success','grasped','stable_placement','dropped','fell','timeout')


def exact_cluster_sign_flip(first,second,positions):
    """Descriptive paired test with positions, rather than repeated seeds, as units."""
    if not (len(first)==len(second)==len(positions)) or not first:
        raise ValueError('Nonempty paired outcomes and position labels must align')
    grouped={}
    for a,b,position in zip(first,second,positions):
        if type(a) is not bool or type(b) is not bool:raise ValueError('Cluster outcomes must be booleans')
        grouped.setdefault(tuple(position),[]).append(int(b)-int(a))
    means=np.asarray([np.mean(grouped[key]) for key in sorted(grouped)],dtype=np.float64)
    observed=float(np.mean(means));k=len(means);total=2**k
    extreme=sum(abs(float(np.mean(means*np.asarray(signs))))>=abs(observed)-1e-12
                for signs in product((-1,1),repeat=k))
    return {'method':'exact_two_sided_position_cluster_sign_flip','cluster_count':k,
            'position_cluster_mean_differences':[{'position_xy_m':list(key),'paired_trials':len(grouped[key]),
                'second_minus_first_success_mean':float(np.mean(grouped[key]))} for key in sorted(grouped)],
            'mean_success_difference':observed,'enumerated_sign_patterns':total,
            'two_sided_p':extreme/total,'minimum_attainable_two_sided_p':min(1.,2/total),
            'interpretation':'Positions are the sign-flip units; repeated seeds are not independent scenes. Assumes exchangeable signs under the null. With five positions the smallest two-sided p is 2/32=0.0625; this reporting field does not alter the 16/20 task threshold or existing conclusion rules.'}


def load_benchmark_results(locations):
    """Read immutable raw outputs; resolve external exports and original launch receipts."""
    records=[];seen=set();project=Path(__file__).resolve().parents[2]
    for location in map(Path,locations):
        files=sorted(location.rglob('result.json')) if location.is_dir() else [location]
        for source in files:
            source=source.resolve()
            if source in seen:continue
            seen.add(source);row=json.loads(source.read_text());row['result_source_path']=str(source)
            config_file=source.parent/'config.json'
            config=json.loads(config_file.read_text()) if config_file.exists() else {}
            if config:
                if config.get('case')!=row.get('case') or config.get('policy_id',config.get('mode'))!=row.get('policy'):
                    raise ValueError('Saved case config identity differs from immutable result')
                if config.get('video_output_dir'):
                    destination=Path(config['video_output_dir']);destination=destination if destination.is_absolute() else project/destination
                    row['video_manifest_path']=str(destination/'video_manifest.json')
            search_root=next((ancestor for ancestor in source.parents if ancestor.name=='round02'),source.parent.parent)
            matches=[]
            for receipt_path in search_root.rglob('*.launch.json'):
                receipt=json.loads(receipt_path.read_text());command=receipt.get('command',[])
                if '--config' not in command:continue
                path=Path(command[command.index('--config')+1]);path=path if path.is_absolute() else project/path
                if not path.is_file():continue
                launched=json.loads(path.read_text());output=Path(launched['output_dir']);output=output if output.is_absolute() else project/output
                if output.resolve()!=source.parent:continue
                if launched.get('case')!=row.get('case') or launched.get('policy_id',launched.get('mode'))!=row.get('policy'):
                    raise ValueError('Launch receipt config differs from immutable case result')
                elapsed=receipt.get('elapsed_seconds')
                if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid launch elapsed_seconds')
                matches.append((receipt_path,elapsed))
            if len(matches)>1:raise ValueError('Ambiguous original launch wall-time receipts')
            if matches:row.update(launch_wall_receipt=str(matches[0][0]),launch_wall_seconds=matches[0][1])
            records.append(row)
    return records


def measured_wall_timing(row):
    """Compute task and RPC durations only from original captured monotonic timestamps."""
    source=row.get('result_source_path')
    if not source:return {'complete':False,'errors':['No raw result source path for physical wall-time audit']}
    trace=Path(source).parent/'trajectory.jsonl'
    if not trace.is_file():return {'complete':False,'errors':['Physical monotonic trace missing']}
    times=[]
    for i,line in enumerate(trace.read_text().splitlines()):
        sample=json.loads(line)
        if sample['control_tick']!=i or type(sample.get('monotonic_ns')) is not int:raise ValueError('Wall trace lacks exact control ticks/monotonic timestamps')
        times.append(sample['monotonic_ns'])
    if not times or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Physical wall trace must be nonempty and strictly monotonic')
    if len(times)!=row['completed_control_ticks']+1:raise ValueError('Wall trace/result control coverage differs')
    first,last=times[0],times[-1];intervals=[];rpc=[]
    for call in row.get('inference_calls',[]):
        start,end=call.get('request_started_monotonic_ns'),call.get('reply_received_monotonic_ns');tick=call.get('observation_control_tick')
        if (type(start) is not int or type(end) is not int or type(tick) is not int or not 0<=tick<len(times)-1
                or not times[tick]<=start<=end<=times[tick+1]):raise ValueError('RPC timing does not correspond to its same-tick observation and paused physical step')
        intervals.append((start,end));rpc.append({'tick':tick,'seconds':(end-start)/1e9,'model_load_seconds':call.get('model_load_seconds')})
    def paused_until(endpoint):
        spans=sorted((max(first,a),min(endpoint,b)) for a,b in intervals if a<endpoint and b>first)
        total=0;right=first
        for a,b in spans:
            if b>max(a,right):total+=b-max(a,right)
            right=max(right,b)
        return total/1e9
    episode=(last-first)/1e9;paused=paused_until(last)
    out={'complete':True,'errors':[],'trace_sha256':sha(trace),'episode_observed_wall_seconds':episode,
         'RPC_pause_episode_seconds':paused,'execution_wall_excluding_RPC_pause_seconds':max(0.,episode-paused),
         'first_rpc_wall_seconds':rpc[0]['seconds'] if rpc else None,
         'first_rpc_model_load_seconds':rpc[0]['model_load_seconds'] if rpc else None,
         'warm_rpc_wall_seconds':[r['seconds'] for r in rpc if r['tick']>0],
         'actual_task_completion_wall_seconds':None,'task_execution_wall_excluding_RPC_pause_seconds':None}
    started,ready=row.get('cold_process_started_monotonic_ns'),row.get('cold_ready_monotonic_ns')
    if type(started) is int and type(ready) is int and started<=ready<=first:
        out['cold_start_to_ready_seconds']=(ready-started)/1e9
        out['cold_start_to_first_frame_seconds']=(first-started)/1e9
    else:
        out.update(complete=False,errors=['Cold process start/ready timestamps absent or out of order'])
    if row['success']:
        tick=row['qualified_stability_window']['end_tick']
        if type(tick) is not int or not 0<=tick<len(times):raise ValueError('Successful stability endpoint missing from original physical trace')
        duration=(times[tick]-first)/1e9
        out['actual_task_completion_wall_seconds']=duration
        out['task_execution_wall_excluding_RPC_pause_seconds']=max(0.,duration-paused_until(times[tick]))
        out['RPC_pause_until_task_completion_seconds']=paused_until(times[tick])
        if type(started) is int:out['cold_process_to_task_completion_wall_seconds']=(times[tick]-started)/1e9
    return out


def wall_timing_summary(records):
    fields=('actual_task_completion_wall_seconds','cold_start_to_ready_seconds','cold_start_to_first_frame_seconds',
            'first_rpc_wall_seconds','first_rpc_model_load_seconds','episode_observed_wall_seconds',
            'execution_wall_excluding_RPC_pause_seconds','task_execution_wall_excluding_RPC_pause_seconds',
            'cold_process_to_task_completion_wall_seconds')
    out={field:distribution([row[field] for row in records.values() if row.get(field) is not None]) for field in fields}
    out['warm_rpc_wall_seconds']=distribution([value for row in records.values() for value in row.get('warm_rpc_wall_seconds',[])])
    out['measured_wall_time_by_case']=records
    out['wall_time_definitions']={
       'actual_task_completion_wall_seconds':'Only successful cases: qualified-window endpoint monotonic_ns minus first observed frame. Includes first model cold-load RPC and subsequent inference pauses; excludes simulator startup before frame0 and post-endpoint cleanup.',
       'cold_start_to_ready_seconds':'Simulator recorded cold_ready minus cold_process_started; this phase is distinct from first model RPC cold load.',
       'warm_rpc_wall_seconds':'Primary serving-speed metric: actual reply_received minus request_started for observation ticks >0, including IPC/preprocessing/inference.',
       'execution_wall_excluding_RPC_pause_seconds':'Full observed episode wall time minus measured RPC interval union. Includes physics, rendering, PNG/video logging and control overhead; not proof of hardware real-time1x.',
       'task_execution_wall_excluding_RPC_pause_seconds':'Successful endpoint wall interval minus RPC pauses up to that endpoint; still includes simulation/render/recording overhead.',
       'launch_wall_seconds':'Matched original launcher elapsed time additionally includes startup and final cleanup.'}
    return out


def failure_stage(row):
    """Mutually exclusive task-stage classification; infrastructure is separate."""
    if row.get('success') is None or row.get('status')!='completed':return 'infrastructure_incomplete'
    if row['fell']:return 'fell'
    if row['dropped']:return 'dropped'
    if row['success']:return 'success'
    if not row['grasped']:return 'missed_grasp'
    if row.get('source_success_candidate_tick') is None:return 'grasp_without_placement'
    return 'unstable_or_unreleased_placement'


def check_videos(row,probe=True):
    errors=[];entries={};path=row.get('video_manifest_path') or str(Path(row['result_source_path']).parent/'video_manifest.json')
    path=Path(path)
    if not path.is_absolute():path=Path(row['result_source_path']).parent/path
    if not path.is_file():return {'complete':False,'errors':['video manifest missing'],'videos':{}}
    manifest=json.loads(path.read_text())
    if manifest.get('raw_result_sha256')!=sha(row['result_source_path']):errors.append('Video export does not bind the immutable raw result SHA')
    if manifest.get('schema')!='pi05_g1_round02_video_manifest_v1' or manifest.get('case_id')!=row['case_id'] or manifest.get('policy')!=row['policy']:
        errors.append('video manifest identity/schema mismatch')
    trace=Path(row['result_source_path']).parent/'trajectory.jsonl'
    if not trace.is_file() or manifest.get('trace_sha256')!=sha(trace):errors.append('video wall annotation trace hash mismatch')
    expected=int(row['completed_control_ticks'])+1
    for key in VIDEO_KEYS:
        item=manifest.get('videos',{}).get(key)
        if not isinstance(item,dict):errors.append(key+': missing');continue
        video=Path(item['path']);video=video if video.is_absolute() else path.parent/video
        if not video.is_file():errors.append(key+': file missing');continue
        digest=sha(video)
        if digest!=item.get('sha256'):errors.append(key+': sha mismatch')
        if item.get('fps')!=50 or item.get('frames')!=expected or item.get('wall_annotation')!=key.endswith('wall_annotated'):
            errors.append(key+': FPS/frame count/annotation flag mismatch')
        entry={'path':str(video),'sha256':digest,'fps':item.get('fps'),'frames':item.get('frames'),'wall_annotation':item.get('wall_annotation')}
        if probe:
            with av.open(str(video)) as container:
                stream=container.streams.video[0];fps=float(stream.average_rate)
                pts=[float(frame.pts*frame.time_base) for frame in container.decode(stream)]
            entry.update(decoded_fps=fps,decoded_frames=len(pts))
            if fps!=50 or len(pts)!=expected or not np.allclose(pts,np.arange(expected)/50.,atol=1e-7,rtol=0):errors.append(key+': decoded timing differs from physical trace')
        entries[key]=entry
    wall_clock={}
    for key in ('head_wall_clock','observer_wall_clock'):
        item=manifest.get('videos',{}).get(key)
        if item is None:continue
        video=Path(item['path']);video=video if video.is_absolute() else path.parent/video
        fps=item.get('fps');preserved=manifest.get('timing',{}).get('total_preserved_wall_seconds')
        expected_wall=math.ceil(preserved*fps)+1 if isinstance(fps,(int,float)) and fps>0 and isinstance(preserved,(int,float)) else None
        if (not video.is_file() or item.get('sha256')!=sha(video) or item.get('frames')!=expected_wall
            or item.get('playback_kind')!='measured_wall_clock_resampled' or item.get('wall_annotation') is not True):
            errors.append(key+': actual wall-clock evidence invalid');continue
        if probe:
            with av.open(str(video)) as container:
                stream=container.streams.video[0];pts=[float(frame.pts*frame.time_base) for frame in container.decode(stream)]
                if float(stream.average_rate)!=fps or len(pts)!=expected_wall or not np.allclose(pts,np.arange(expected_wall)/fps,atol=1e-7,rtol=0):errors.append(key+': decoded wall-clock timing mismatch')
        wall_clock[key]={**item,'path':str(video)}
    return {'complete':not errors and len(entries)==4,'errors':errors,'videos':entries,
            'actual_wall_clock_videos':wall_clock,'both_actual_wall_clock_views_present':len(wall_clock)==2,
            'wall_clock_scope':'Actual wall clips preserve cold-start blank time and repeat last measured frame across pauses. Final cleanup after the last observed frame is outside these clips and remains in launch_wall_seconds.',
            'manifest_path':str(path),'manifest_sha256':sha(path),
            'interpretation':'All videos play at 50Hz physical time. Wall overlays disclose measured elapsed wall time; frames are not stretched to hide pauses and do not prove real-time 1x.'}


def aggregate(cases,policy_rows,gate,probe_videos=True):
    if len(policy_rows) not in (3,4):raise ValueError('Exactly three or four named policies are required')
    ids=[r['case_id'] for r in cases];positions={tuple(r['apple_offset_xy_m']) for r in cases}
    if len(ids)!=20 or len(set(ids))!=20 or len(positions)!=5 or any(sum(tuple(c['apple_offset_xy_m'])==p for c in cases)!=4 for p in positions):
        raise ValueError('Benchmark requires twenty unique frozen cases, five positions with four repeats each')
    errors=[];policies={};tables={};videos=[];infra=[];wall_records={}
    required_version='pi05_g1_placement_v2';expected_sha=gate.get('acceptance_sha256')
    if expected_sha!=acceptance_identity()['acceptance_sha256']:errors.append('Replay does not use the immutable acceptance-v2 module hash')
    if gate.get('acceptance_version')!=required_version or gate.get('replay_passed') is not True or gate.get('source_interface_passed') is not True:
        errors.append('A successful acceptance-v2 replay gate is required')
    if not isinstance(expected_sha,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha):errors.append('Replay acceptance hash missing')
    for policy,rows in policy_rows.items():
        table={};all_stages={stage:0 for stage in ('fell','dropped','missed_grasp','grasp_without_placement','unstable_or_unreleased_placement','success')}
        seen=set()
        for row in rows:
            cid=row.get('case_id')
            if cid not in ids or row.get('policy')!=policy:errors.append(f'{policy}/{cid}: unexpected policy or frozen case');continue
            if cid in seen:raise ValueError(f'Duplicate result: {policy}/{cid}')
            seen.add(cid)
            if row.get('success') is None or row.get('status')!='completed':
                infra.append({'policy':policy,'case_id':cid,'status':row.get('status'),'error':row.get('error')});continue
            if any(type(row.get(key)) is not bool for key in FLAGS):raise ValueError('All physical outcome flags must be explicit booleans')
            case=next(c for c in cases if c['case_id']==cid)
            if row.get('case')!=case:errors.append(f'{policy}/{cid}: frozen case mismatch')
            if row.get('acceptance_version')!=required_version or row.get('acceptance_sha256')!=expected_sha:errors.append(f'{policy}/{cid}: acceptance mismatch')
            if row.get('frozen_interface_sha256')!=gate.get('frozen_interface_sha256') or not row.get('frozen_interface_sha256'):
                errors.append(f'{policy}/{cid}: common physical/camera/bounds/replan interface mismatch')
            if row.get('source_interface_passed') is not True or row.get('replay_passed') is not True:errors.append(f'{policy}/{cid}: source/replay gate missing')
            if row['success'] and (not row['grasped'] or not row['stable_placement'] or row['fell'] or row['dropped'] or row['timeout']):raise ValueError('Success contradicts physical stages')
            table[cid]=row;all_stages[failure_stage(row)]+=1
            evidence=check_videos(row,probe_videos);videos.append({'policy':policy,'case_id':cid,**evidence})
            if not evidence['complete']:errors.append(f'{policy}/{cid}: incomplete video evidence')
        wall_records[policy]={cid:measured_wall_timing(row) for cid,row in table.items()}
        for cid,measurement in wall_records[policy].items():
            if not measurement['complete']:errors.append(f'{policy}/{cid}: incomplete actual wall-time evidence')
        tables[policy]=table;n=len(table);success=sum(r['success'] for r in table.values())
        artifact_identities={json.dumps(row.get('policy_artifacts',{}),sort_keys=True) for row in table.values()}
        if len(artifact_identities)>1:errors.append(policy+': policy artifacts changed between frozen cases')
        clip=[r['action_clip_fraction'] for r in table.values() if r.get('action_clip_fraction') is not None]
        clipped=sum(r.get('clipped_channels',0) for r in table.values());commanded=sum(r.get('commanded_channels',0) for r in table.values())
        policies[policy]={'policy_artifact_identities':[json.loads(value) for value in sorted(artifact_identities)],'completed':n,'expected':20,'missing_or_infrastructure_cases':[cid for cid in ids if cid not in table],
           'successes':success,'success_rate':success/20 if n==20 else None,'success_wilson_95ci':wilson(success,n),
           'task_failure_rate':(20-success)/20 if n==20 else None,
           'observed_task_failure_rate':(n-success)/n if n else None,'mutually_exclusive_stage_counts':all_stages,
           'counts':{key:sum(r[key] for r in table.values()) for key in FLAGS},'raw_action_clipping_fraction':distribution(clip),
           'total_clipped_channels':clipped,'total_commanded_channels':commanded,
           'pooled_source_command_clipping_fraction':clipped/commanded if commanded else None,
           'clipping_scope':'Simulator counts actual50Hz commands including held audit tail; diagnostic only.',
           'real_time_1x_verified':bool(table) and all(r.get('real_time_1x_verified') is True for r in table.values()),
           'meets_16_of_20':n==20 and success>=16,**timing_summary(table),**wall_timing_summary(wall_records[policy])}
    init_checks=[]
    for cid in ids:
        values={policy:table[cid].get('initial_physical_state_sha256') for policy,table in tables.items() if cid in table}
        valid=len(values)==len(policy_rows) and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in values.values()) and len(set(values.values()))==1
        init_checks.append({'case_id':cid,'actual_initial_hashes':values,'all_policies_match':bool(valid),'policies_expected':len(policy_rows)})
        if not valid:errors.append(cid+': missing or mismatched actual physical initial states')
    comparisons=[]
    for first,second in combinations(tables,2):
        a,b=tables[first],tables[second]
        paired=[cid for cid in ids if cid in a and cid in b and isinstance(a[cid].get('initial_physical_state_sha256'),str)
                and re.fullmatch('[0-9a-f]{64}',a[cid]['initial_physical_state_sha256'])
                and a[cid]['initial_physical_state_sha256']==b[cid].get('initial_physical_state_sha256')]
        x=[a[cid]['success'] for cid in paired];y=[b[cid]['success'] for cid in paired]
        xy=[next(c['apple_offset_xy_m'] for c in cases if c['case_id']==cid) for cid in paired]
        matched_success=[cid for cid in paired if a[cid]['success'] and b[cid]['success']]
        speed=[]
        for cid in matched_success:
            ta=a[cid]['qualified_stability_window']['end_tick']/a[cid]['control_hz'];tb=b[cid]['qualified_stability_window']['end_tick']/b[cid]['control_hz']
            wa=wall_records[first][cid].get('actual_task_completion_wall_seconds');wb=wall_records[second][cid].get('actual_task_completion_wall_seconds')
            speed.append({'case_id':cid,
                          first+'_actual_task_completion_wall_seconds':wa,second+'_actual_task_completion_wall_seconds':wb,
                          'second_minus_first_actual_task_wall_seconds':wb-wa if wa is not None and wb is not None else None,first+'_successful_completion_sim_seconds':ta,second+'_successful_completion_sim_seconds':tb,'second_minus_first_sim_seconds':tb-ta,
                          first+'_launch_wall_seconds':a[cid].get('launch_wall_seconds'),second+'_launch_wall_seconds':b[cid].get('launch_wall_seconds')})
        comparisons.append({'first_policy':first,'second_policy':second,'eligible_physical_pairs':len(paired),
          'paired_conditioned_trial_statistics':paired_statistics(x,y) if paired else None,
          'position_cluster_uncertainty':clustered_statistics(x,y,xy) if paired else None,
          'position_cluster_exact_sign_flip':exact_cluster_sign_flip(x,y,xy) if paired else None,
          'matched_success_case_ids':matched_success,'matched_success_speed':speed,
          'matched_success_sim_delta_seconds':distribution([r['second_minus_first_sim_seconds'] for r in speed]),
          'matched_success_actual_task_wall_delta_seconds':distribution([r['second_minus_first_actual_task_wall_seconds'] for r in speed if r['second_minus_first_actual_task_wall_seconds'] is not None]),
          'speed_interpretation':'Only cases where both policies succeeded are used for paired speed. Fast timeouts/failures are never speed wins; success-only marginal times have different cohorts.'})
    complete=all(len(t)==20 for t in tables.values()) and not infra and not errors
    physical_hashes={value for check in init_checks for value in check['actual_initial_hashes'].values() if isinstance(value,str)}
    return {'schema':'pi05_g1_round02_multi_policy_benchmark_v2','policy_count':len(policy_rows),'main_benchmark_default_policies':3,'complete':complete,'evidence_qualified':complete,
       'policies':policies,'comparisons':comparisons,'actual_initial_pair_checks':init_checks,
       'distinct_initial_state_hashes':len(physical_hashes),'physical_positions':5,'noise_repetitions_per_position':4,
       'uncertainty_scope':'Five position clusters; repeated seeds are not twenty independent physical scenes.',
       'failure_rate_definition':'Task failure=(20-success)/20 only for twenty completed physical trials; infrastructure is separately incomplete, never counted as task failure.',
       'failure_stage_priority':['fell','dropped','success','missed_grasp','grasp_without_placement','unstable_or_unreleased_placement'],
       'infrastructure_incomplete':infra,'video_evidence':videos,'errors':errors,'acceptance_sha256':expected_sha,
       'task_usability_threshold':16,'no_posthoc_tuning':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--policy',action='append',required=True,help='Repeat three times (or optional fourth): policy_id=result directory/file')
    parser.add_argument('--replay-gate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();sources={}
    for text in a.policy:
        policy,path=text.split('=',1)
        if policy in sources:raise ValueError('Duplicate policy ID')
        sources[policy]=load_benchmark_results([Path(path)])
    doc=json.loads(a.cases.read_text());cases=doc['cases'] if isinstance(doc,dict) else doc
    gate=json.loads(a.replay_gate.read_text());report=aggregate(cases,sources,gate)
    report['inputs_sha256']={'cases':sha(a.cases),'replay_gate':sha(a.replay_gate)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Benchmark outputs are versioned; use a new output path')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'complete':report['complete'],'errors':report['errors']}))
