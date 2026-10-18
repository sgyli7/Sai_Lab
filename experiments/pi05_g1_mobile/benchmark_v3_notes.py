"""CPU evidence-qualified mobile benchmark. Missing trials never become failures.

No model imports or policy selection; only frozen planned trials are aggregated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from experiments.pi05_g1.summarize_evaluation import wilson,paired_statistics

REQUIRED_POLICIES={'gr00t_n16','pi05_mobile','official_pi05_base'}
FLAGS=('success','grasped','stable_placement','fell','dropped','timeout')
METRICS=('source_target_distance_m','max_carried_box_displacement_m','contact_qualified_carried_path_m','carry_turn_degrees','hand_contact_gap_events_before_arrival','largest_hand_contact_gap_seconds')
INITIAL_KEYS=('source_joint_names','joint_positions','joint_velocities','root_link_pose_w','root_link_velocity_w','root_com_velocity_w','acceptance_truth_only')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def checked_json(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite JSON value')
    if isinstance(value,dict):
        for item in value.values():checked_json(item)
    elif isinstance(value,list):
        for item in value:checked_json(item)
    return value


def read(path):return checked_json(json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x))))

def finite(x):return type(x) in (int,float) and math.isfinite(x)

def distribution(values):
    if not values:return {'count':0,'median':None,'p95':None,'mean':None,'minimum':None,'maximum':None}
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nonfinite distribution')
    return {'count':len(a),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'mean':float(a.mean()),'minimum':float(a.min()),'maximum':float(a.max())}


def cluster_statistics(first,second,clusters,seed=42,samples=20000):
    if len(first)!=len(second) or len(first)!=len(clusters) or not first:raise ValueError('Nonempty paired cluster data required')
    groups={}
    for a,b,c in zip(first,second,clusters):
        if a not in [0,1] or b not in [0,1]:raise ValueError('Binary outcomes required')
        groups.setdefault(str(c),[]).append(float(b)-float(a))
    values=np.asarray([np.mean(groups[c]) for c in sorted(groups)])
    boot=values[np.random.default_rng(seed).integers(0,len(values),(samples,len(values)))].mean(1)
    observed=abs(float(values.mean()));signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(values))))
    signflip=float(np.mean(np.abs((signs*values).mean(1))>=observed-1e-12))
    return {'physical_position_clusters':len(values),'repeats_per_cluster':{c:len(v) for c,v in groups.items()},'cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'exact_cluster_sign_flip_two_sided_p':signflip,'minimum_possible_two_sided_cluster_p':min(1.,2/2**len(values)),'bootstrap_seed':seed,'bootstrap_samples':samples,'interpretation':'Five physical positions with four seed repeats each; not20 independent scenes. With5clusters minimum two-sided sign-flip p=2/32=.0625. This adds statistical context, not a success gate.'}


def path_from(plan_path,value):
    p=Path(value);return p.resolve() if p.is_absolute() else (plan_path.parent/p).resolve()


def decoded_evidence(config,result):
    command=config.get('policy_command',[])
    if command.count('--decoded-evidence')!=1:raise ValueError('Exactly one explicit decoded-evidence output required')
    i=command.index('--decoded-evidence')
    if i+1>=len(command):raise ValueError('Decoded-evidence output value missing')
    path=Path(command[i+1]).resolve();output=Path(config['output_dir']).resolve()
    if not path.is_relative_to(output) or path==output:raise ValueError('Decoded evidence must be a file inside its case output_dir')
    records=[checked_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not records or records[0].get('event')!='model_loaded' or sum(r.get('event')=='model_loaded' for r in records)!=1:
        raise ValueError('Decoded evidence lacks unique model_loaded record')
    expected_event='native_decoded' if config['policy_id']=='gr00t_n16' else 'pi05_decoded'
    decoded=records[1:]
    if not decoded or any(r.get('event')!=expected_event for r in decoded):raise ValueError('Decoded evidence event/rejection mismatch')
    if [r.get('control_tick') for r in decoded]!=[r['observation_control_tick'] for r in result['inference_calls']]:raise ValueError('Decoded evidence does not cover actual inference calls')
    for record in decoded:
        array=np.asarray(record.get('actions32'),dtype=np.float32)
        if record.get('episode_id')!=config['case']['case_id'] or array.shape!=(50,32) or not np.isfinite(array).all():raise ValueError('Decoded evidence case/actions mismatch')
    return path,records


def load_case(job,plan,plan_path,decoded_checker=None):
    config_path=path_from(plan_path,job['config_path']);result_path=path_from(plan_path,job['result_path'])
    expected=job.get('config_sha256',plan.get('frozen_files',{}).get(str(config_path)))
    if not expected or sha(config_path)!=expected:raise ValueError('Missing/changed frozen config SHA')
    config=read(config_path);result=read(result_path);raw_config_path=result_path.parent/'config.json';raw_config=read(raw_config_path)
    from .simulator import validate_config
    if raw_config!=validate_config(config):raise ValueError('Recorded executed config differs from frozen validated config')
    if config['mode']!='policy' or config.get('evaluate_acceptance') is not True:raise ValueError('Only policy-only acceptance-evaluated trials count')
    if (config.get('policy_id')!=job['policy'] or config['case']['case_id']!=job['case_id'] or result.get('policy')!=job['policy'] or result.get('case_id')!=job['case_id'] or result.get('case')!=config['case']):raise ValueError('Policy/case identity mismatch')
    if result.get('schema')!='pi05_g1_mobile_case_result_v1' or result.get('status')!='completed' or result.get('source_interface_passed') is not True or result.get('cleanup_errors'):raise ValueError('Case infrastructure/source interface incomplete')
    if result.get('control_hz')!=50 or result.get('physics_hz')!=200 or config.get('native_action_horizon',50)!=50:raise ValueError('Native50/control50/physics200 required')
    trace_path=result_path.parent/'trajectory.jsonl'
    if not result.get('trace_sha256') or sha(trace_path)!=result['trace_sha256']:raise ValueError('Trace SHA mismatch/missing')
    rows=[checked_json(json.loads(line)) for line in trace_path.read_text().splitlines() if line.strip()]
    if not rows or any(row['control_tick']!=i or row['physics_steps_since_reset']!=4*i for i,row in enumerate(rows)):raise ValueError('Physical trace gaps/reset/counters mismatch')
    times=[row['monotonic_ns'] for row in rows]
    if any(type(x) is not int for x in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Actual monotonic timing missing/nonincreasing')
    if result.get('completed_control_ticks')!=len(rows)-1:raise ValueError('Completed controls disagree with trace')
    initial={k:rows[0][k] for k in INITIAL_KEYS};initial_sha=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
    if result.get('initial_physical_state_schema')!='pi05_g1_mobile_initial_kinematic_v1' or result.get('initial_physical_state_sha256')!=initial_sha:raise ValueError('Initial measured kinematic state SHA missing/mismatch')
    acceptance=result.get('acceptance',{})
    if acceptance.get('schema')!='pi05_g1_mobile_acceptance_v1' or acceptance.get('placement_rule')!=config['acceptance']:raise ValueError('Missing/incompatible frozen task acceptance')
    for flag in FLAGS:
        if type(acceptance.get(flag)) is not bool:raise ValueError('Missing binary acceptance field '+flag)
    if acceptance['grasped'] and (type(acceptance.get('first_grasp_control_tick')) is not int or not 0<=acceptance['first_grasp_control_tick']<len(rows)):
        raise ValueError('Grasp flag lacks measured event tick')
    if acceptance['stable_placement']!=isinstance(acceptance.get('qualified_stability_window'),dict):raise ValueError('Stable placement/window mismatch')
    if not isinstance(acceptance.get('termination_reasons'),list) or any(not isinstance(reason,str) for reason in acceptance['termination_reasons']):raise ValueError('Termination reasons missing/malformed')
    if result.get('success') is not acceptance['success']:raise ValueError('Top-level success disagrees with acceptance')
    if acceptance['success'] and (not acceptance['grasped'] or not acceptance['stable_placement'] or acceptance['fell'] or acceptance['dropped'] or acceptance['timeout']):raise ValueError('Inconsistent full closed-loop success flags')
    if acceptance.get('policy_external_interventions')!=0:raise ValueError('Policy-only external intervention provenance missing')
    if acceptance.get('actual_control_ticks')!=len(rows)-1 or acceptance.get('physics_integrations')!=4*(len(rows)-1):raise ValueError('Acceptance counter mismatch')
    endpoint=acceptance.get('qualified_stability_window')
    if acceptance['success']:
        if not isinstance(endpoint,dict) or type(endpoint.get('end_tick')) is not int:raise ValueError('Success lacks qualified stable endpoint')
        end_tick=endpoint['end_tick']
        if type(acceptance.get('first_grasp_control_tick')) is not int or not acceptance['first_grasp_control_tick']<endpoint['start_tick']<=end_tick:raise ValueError('Success endpoint precedes grasp')
    else:end_tick=len(rows)-1
    if not 0<=end_tick<len(rows):raise ValueError('Endpoint outside trace')
    actual_wall=(times[end_tick]-times[0])/1e9
    for key,value in [('task_sim_seconds',end_tick/50),('task_wall_seconds',actual_wall)]:
        if not finite(acceptance.get(key)) or abs(acceptance[key]-value)>1e-7:raise ValueError('Measured task timing mismatch '+key)
    metrics={}
    for key in METRICS:
        value=acceptance.get(key)
        if not finite(value) or value<0:raise ValueError('Missing/nonfinite physical metric '+key)
        metrics[key]=value
    calls=result.get('inference_calls')
    if not isinstance(calls,list) or not calls:raise ValueError('Actual inference timing absent')
    warm=[];cold=[];paused=[];load=[];pause_until_endpoint=0.
    ticks=[]
    for call in calls:
        tick=call.get('observation_control_tick');ticks.append(tick)
        if type(tick) is not int or not 0<=tick<len(rows)-1:raise ValueError('Invalid inference control tick')
        for key in ['inference_seconds','simulation_paused_wall_seconds']:
            if not finite(call.get(key)) or call[key]<0:raise ValueError('Missing actual inference timing '+key)
        rpc=call['simulation_paused_wall_seconds'];paused.append(rpc)
        (cold if tick==0 else warm).append(call['inference_seconds'])
        if tick<end_tick:pause_until_endpoint+=rpc
        if call.get('model_load_seconds') is not None:
            if not finite(call['model_load_seconds']) or call['model_load_seconds']<0:raise ValueError('Invalid model load timing')
            if tick==0:load.append(call['model_load_seconds'])
    if len(load)!=1:raise ValueError('Actual tick0 model loading timing missing')
    if ticks[0]!=0 or len(set(ticks))!=len(ticks) or ticks!=sorted(ticks):raise ValueError('Inference sequence coverage invalid')
    if pause_until_endpoint>actual_wall+1e-3:raise ValueError('Inference pause exceeds measured task wall')
    cold_started=result.get('cold_process_started_monotonic_ns');finalized=result.get('evidence_finalized_monotonic_ns')
    if type(cold_started) is not int or type(finalized) is not int or not cold_started<=times[0]<=times[-1]<=finalized:raise ValueError('Actual process wall timestamps incomplete')
    evidence_path,_=(decoded_evidence if decoded_checker is None else decoded_checker)(config,result)
    return {'policy':job['policy'],'case_id':job['case_id'],'cluster_id':config['case'].get('cluster_id'),**{k:acceptance[k] for k in FLAGS},**metrics,
        'initial_physical_state_sha256':initial_sha,'frozen_interface_sha256':result.get('frozen_interface_sha256'),'task_sim_seconds':acceptance['task_sim_seconds'],'task_wall_seconds':actual_wall,
        'task_completion_sim_seconds_success_only':acceptance['task_sim_seconds'] if acceptance['success'] else None,'task_completion_wall_seconds_success_only':actual_wall if acceptance['success'] else None,
        'first_grasp_sim_seconds':acceptance['first_grasp_control_tick']/50 if acceptance.get('first_grasp_control_tick') is not None else None,
        'qualified_stability_endpoint_sim_seconds':endpoint['end_tick']/50 if endpoint else None,'episode_sim_seconds':(len(rows)-1)/50,'episode_wall_seconds':(times[-1]-times[0])/1e9,
        'cold_process_to_frame0_seconds':(times[0]-cold_started)/1e9,'process_to_evidence_finalized_wall_seconds':(finalized-cold_started)/1e9,
        'execution_wall_excluding_RPC_pause_seconds':max(0.,actual_wall-pause_until_endpoint),'cold_inference_seconds':cold,'warm_inference_seconds':warm,'model_load_seconds':load,'RPC_pause_wall_seconds':paused,'warm_RPC_pause_wall_seconds':[c['simulation_paused_wall_seconds'] for c in calls if c['observation_control_tick']>0],'cold_RPC_pause_wall_seconds':[c['simulation_paused_wall_seconds'] for c in calls if c['observation_control_tick']==0],
        'action_clipping_fraction':result.get('clipped_channels',0)/result['commanded_channels'] if result.get('commanded_channels') else None,
        'termination_reasons':acceptance.get('termination_reasons',[]),'result_path':str(result_path),'config_path':str(config_path),'inputs_sha256':{str(p):sha(p) for p in [config_path,raw_config_path,result_path,trace_path,evidence_path]},'config':config}


def aggregate(plan_path,output):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve()
    if output.exists() or output.with_suffix('.csv').exists():raise FileExistsError('Preserve prior benchmark outputs')
    plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_mobile_benchmark_plan_v1':raise ValueError('Unsupported frozen mobile benchmark plan')
    jobs=plan['jobs'];policies=plan.get('primary_policies',plan.get('policies',sorted({j['policy'] for j in jobs})))
    if not REQUIRED_POLICIES<=set(policies) or len(policies)!=len(set(policies)):raise ValueError('Unique primary policy roster must contain nativeN1.6/mobilePi/officialbase')
    cases=plan.get('cases',[]);case_ids=plan.get('case_ids',[c['case_id'] for c in cases])
    if len(case_ids)!=20 or len(set(case_ids))!=20:raise ValueError('Frozen20 unique case IDs required')
    expected={(policy,case) for policy in policies for case in case_ids};keys=[(j['policy'],j['case_id']) for j in jobs]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Plan must contain exactly every policy x frozen20 cases')
    errors=[];inputs={str(plan_path):sha(plan_path)};records={};missing=[]
    frozen=plan.get('frozen_files',{})
    acceptance_path=Path(__file__).with_name('acceptance.py').resolve()
    if not frozen or frozen.get(str(acceptance_path))!=sha(acceptance_path):errors.append('Frozen acceptance source SHA missing/mismatch')
    for name,digest in frozen.items():
        path=path_from(plan_path,name)
        try:
            if sha(path)!=digest:errors.append('Frozen input SHA mismatch: '+str(path))
        except OSError as error:errors.append('Frozen input unavailable: '+str(error))
    clusters={c['case_id']:c.get('cluster_id') for c in cases};cluster_counts={}
    for job in jobs:
        key=(job['policy'],job['case_id'])
        try:
            row=load_case(job,plan,plan_path)
            cluster=row['cluster_id'] if row['cluster_id'] is not None else clusters.get(job['case_id'])
            if cluster is None:raise ValueError('Frozen case cluster_id absent')
            if job['case_id'] in clusters and str(clusters[job['case_id']])!=str(cluster):raise ValueError('Plan/config cluster identity mismatch')
            clusters[job['case_id']]=str(cluster);row['cluster_id']=str(cluster);records[key]=row;inputs.update(row['inputs_sha256'])
        except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
            errors.append(job['policy']+'/'+job['case_id']+': '+str(error));missing.append({'policy':job['policy'],'case_id':job['case_id'],'reason':str(error)})
    for case in case_ids:
        if case in clusters:cluster_counts[str(clusters[case])]=cluster_counts.get(str(clusters[case]),0)+1
        pair=[records.get((policy,case)) for policy in policies]
        if all(pair):
            if len({r['initial_physical_state_sha256'] for r in pair})!=1:errors.append('Actual initial state differs across policies: '+case)
            if len({r['frozen_interface_sha256'] for r in pair})!=1 or not pair[0]['frozen_interface_sha256']:errors.append('Actual frozen interface differs/missing: '+case)
    if len(cluster_counts)!=5 or set(cluster_counts.values())!={4}:errors.append('Require5physical-position clusters x4seed repeats')
    pi=[row for row in records.values() if row['policy'] in ['pi05_mobile','official_pi05_base']]
    norm_paths={str(Path(row['config'].get('norm_stats','')).resolve()) for row in pi};commands=[]
    for row in pi:
        cmd=row['config'].get('policy_command',[])
        if '--checkpoint' not in cmd:errors.append('Pi model command lacks explicit checkpoint: '+row['case_id']);continue
        cmd=list(cmd);index=cmd.index('--checkpoint');cmd[index+1]='<DIFFERENT_PI_WEIGHTS_ONLY>'
        evidence_index=cmd.index('--decoded-evidence');cmd[evidence_index+1]='<CASE_OWNED_DECODED_EVIDENCE_OUTPUT>'
        commands.append(tuple(cmd))
    if len(norm_paths)!=1 or len(set(commands))>1:errors.append('Pi policies do not share norm/runtime command apart from weights')
    for path in norm_paths:
        if path not in frozen:errors.append('Pi common norm lacks frozen SHA binding: '+path)
    if len({json.dumps(row['config']['acceptance'],sort_keys=True) for row in records.values()})>1:errors.append('Task acceptance rule differs across policies/cases')
    complete=len(records)==len(expected) and not errors
    table={};pairs={}
    for policy in policies:
        own=[records[(policy,c)] for c in case_ids if (policy,c) in records];n=len(own)
        counts={flag:sum(row[flag] for row in own) for flag in FLAGS}
        failure_counts={flag:sum(row[flag] for row in own) for flag in ['fell','dropped','timeout']}
        failure_counts.update(no_verified_grasp=sum(not row['grasped'] for row in own),no_released_stable_placement=sum(not row['stable_placement'] for row in own))
        for reason in sorted({reason for row in own for reason in row['termination_reasons']}):
            failure_counts[reason]=sum(reason in row['termination_reasons'] for row in own)
        table[policy]={'completed_evidence_cases':n,'expected_cases':20,'counts_observed':counts,'formal_success_count':counts['success'] if complete else None,
            'success_rate':counts['success']/20 if complete else None,'success_wilson95ci':wilson(counts['success'],20) if complete else None,
            'task_failure_rate':(20-counts['success'])/20 if complete else None,'nonexclusive_failure_counts_observed':failure_counts,'nonexclusive_failure_rates':{flag:count/20 for flag,count in failure_counts.items()} if complete else None,
            'failure_rate_note':'Task failure = 1 - full closed-loop success. Fall/drop may overlap; timeout is residual and exclusive with both (not success, not fell, not dropped). Other diagnostic reasons may overlap and must not be summed. Zero ground-drop flags does not imply zero handling errors.',
            'success_only_task_sim_seconds':distribution([r['task_sim_seconds'] for r in own if r['success']]),'success_only_actual_task_wall_seconds':distribution([r['task_wall_seconds'] for r in own if r['success']]),
            'warm_inference_seconds':distribution([v for r in own for v in r['warm_inference_seconds']]),'cold_inference_seconds':distribution([v for r in own for v in r['cold_inference_seconds']]),
            'model_load_seconds':distribution([v for r in own for v in r['model_load_seconds']]),'warm_RPC_pause_wall_seconds':distribution([v for r in own for v in r['warm_RPC_pause_wall_seconds']]),'cold_RPC_pause_wall_seconds':distribution([v for r in own for v in r['cold_RPC_pause_wall_seconds']]),'RPC_pause_wall_seconds':distribution([v for r in own for v in r['RPC_pause_wall_seconds']]),
            'episode_actual_wall_seconds_all_outcomes':distribution([r['episode_wall_seconds'] for r in own]),'cold_process_to_frame0_seconds':distribution([r['cold_process_to_frame0_seconds'] for r in own]),
            'process_to_evidence_finalized_wall_seconds':distribution([r['process_to_evidence_finalized_wall_seconds'] for r in own]),'physical_metrics':{key:distribution([r[key] for r in own]) for key in METRICS}}
    if complete:
        for first,second in itertools.combinations(policies,2):
            a=[records[(first,c)] for c in case_ids];b=[records[(second,c)] for c in case_ids];matched=[c for c,x,y in zip(case_ids,a,b) if x['success'] and y['success']]
            pairs[first+'__'+second]={'first':first,'second':second,**paired_statistics([r['success'] for r in a],[r['success'] for r in b],seed=42),
                'physical_position_cluster':cluster_statistics([r['success'] for r in a],[r['success'] for r in b],[clusters[c] for c in case_ids]),
                'matched_success_case_ids':matched,'matched_success_count':len(matched),'speed_available':bool(matched),
                'matched_success_task_sim_difference_seconds_second_minus_first':distribution([records[(second,c)]['task_sim_seconds']-records[(first,c)]['task_sim_seconds'] for c in matched]),
                'matched_success_actual_task_wall_difference_seconds_second_minus_first':distribution([records[(second,c)]['task_wall_seconds']-records[(first,c)]['task_wall_seconds'] for c in matched]),
                'speed_interpretation':'Only both-success matched cases count. Empty intersection=N/A; fast failure is never a speed win. Negative difference means second completed faster conditional on both succeeding.'}
    public=[]
    for row in records.values():public.append({k:v for k,v in row.items() if k not in ['config','inputs_sha256']})
    fields=['policy','case_id','cluster_id',*FLAGS,'task_sim_seconds','task_wall_seconds','first_grasp_sim_seconds','qualified_stability_endpoint_sim_seconds','task_completion_sim_seconds_success_only','task_completion_wall_seconds_success_only',*METRICS,'episode_sim_seconds','episode_wall_seconds','cold_process_to_frame0_seconds','process_to_evidence_finalized_wall_seconds','execution_wall_excluding_RPC_pause_seconds','action_clipping_fraction','initial_physical_state_sha256','result_path']
    output.parent.mkdir(parents=True,exist_ok=True);csv_path=output.with_suffix('.csv')
    with csv_path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(public)
    report={'schema':'pi05_g1_mobile_benchmark_summary_v1','status':'complete' if complete else 'incomplete','complete':complete,'evidence_qualified':complete,'formal_verdict_available':complete,'formal_verdict':'Complete measured simulation comparison; see scores, not hardware proof' if complete else None,
        'expected_cases_per_policy':20,'policies':table,'policy_order':policies,'case_ids':case_ids,'position_cluster_counts':cluster_counts,'paired_comparisons':pairs,'errors':errors,'missing_or_invalid_cases':missing,'episodes':public,
        'scope':'Simulation only,5physicalpositions x4seed repeats; not20 independent scenes. Historical original T2 source30s proximity score is not a long-distance5m or released-stable success score.',
        'timing_definitions':{'task_wall':'Trace row0 to qualified stability endpoint for successes; failures retain episode endpoint diagnostically. Includes render/PNG/physics/inference pauses after frame0.','cold_inference':'Model-reported tick0 compute; loading measured separately and may overlap first RPC wait.','warm_inference':'Model-reported inference_seconds at controltick>0; primary inference compute speed.','RPC_pause':'Wall wait for entire policy RPC, includes IPC/preprocessing and tick0 model readiness.','execution_wall_excluding_RPC_pause':'Task wall minus RPC pauses before endpoint, still includes physics/render/recording, not a hardware1x claim.','process_wall':'cold process start to evidence finalized includes startup/cleanup inside simulator, not host Docker launch wall.'},
        'adapter_comparison':'Both Pi policies require identical norm path and runtime command except checkpoint weights; GR00T uses native50tick decoder. Same actual initial state/interface required per case.','noncausal_interpretation':'Validation-selected training and closed-loop task scores are distinct; this summary does not attribute changes to individual recipe components.',
        'inputs_sha256':inputs,'episode_csv':{'path':str(csv_path),'sha256':sha(csv_path)},'tool_sha256':sha(Path(__file__))}
    with output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=aggregate(a.plan,a.output);print(json.dumps({'status':report['status'],'formal_verdict_available':report['formal_verdict_available'],'errors':report['errors']}))

if __name__=='__main__':main()
