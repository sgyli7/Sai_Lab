"""Read-only transport metrics using frozen physical acceptance/contact semantics.

--episodes CASE_DIR [CASE_DIR ...] or --benchmark COMPLETE_SUMMARY.json
--output NEW.json. No new arrival/success threshold; no policy or GPU execution.
Infrastructure failures remain unknown full-task outcomes; prefix observations
are explicitly diagnostic. A50Hz last-solve force sample is not continuous grasp.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import acceptance


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def intervals(ticks):
    result=[]
    for tick in ticks:
        if result and tick==result[-1]['end_tick']+1:result[-1]['end_tick']=tick
        else:result.append({'start_tick':tick,'end_tick':tick})
    for item in result:
        item['sample_count']=item['end_tick']-item['start_tick']+1
        item['sample_span_seconds']=(item['end_tick']-item['start_tick'])/50
        item['missing_sample_count_over_50Hz_seconds']=item['sample_count']/50
    return result


def stats(values):
    array=np.asarray(values,float)
    if not len(array) or not np.isfinite(array).all():raise ValueError('Empty/nonfinite measured statistic')
    return {'minimum':float(array.min()),'maximum':float(array.max()),'mean':float(array.mean()),'final':float(array[-1])}


def summarize_rows(rows,judgment,context):
    """Acceptance samples come from an independent re-run over the same rows."""
    samples=judgment['samples']
    if len(samples)!=len(rows) or any(s['tick']!=i for i,s in enumerate(samples)):raise ValueError('Acceptance sample coverage differs')
    errors=[];per_tick=[];held_positions=[];held_yaws=[];gaps=[]
    first=judgment['first_grasp_control_tick'];carry_origin=None;max_displacement=0.;first_4m60=None
    rule=context['acceptance'];source_distance=judgment['source_target_distance_m']
    for row,sample in zip(rows,samples):
        tick=row['control_tick'];truth=row['acceptance_truth_only'];bp=np.asarray(truth['brown_box']['pose'][:2],float);tp=np.asarray(truth['blue_sorting_bin']['pose'][:2],float)
        error=float(np.linalg.norm(bp-tp));errors.append(error)
        held=sample['grasp_contact_lift_proxy']
        if held:
            if carry_origin is None:carry_origin=bp.copy()
            held_positions.append(bp);max_displacement=max(max_displacement,float(np.linalg.norm(bp-carry_origin)))
            rot=acceptance.rotation_xyzw(row['root_link_pose_w']);held_yaws.append(math.atan2(rot[1,0],rot[0,0]))
        elif first is not None and tick>=first and not sample['box_inside_bin_prism']:gaps.append(tick)
        turn=float(math.degrees(np.ptp(np.unwrap(held_yaws)))) if held_yaws else 0.
        if (first_4m60 is None and held and source_distance>=rule['minimum_source_target_distance_m'] and
            max_displacement>=4. and turn>=60.):first_4m60=tick
        contacts=row['acceptance_contacts_only'];meta=context['audit_contact_metadata']
        robot_points=any(p['filter_index'] in meta['robot_filter_indices'] and (p['separation_m']<=0 or abs(p['normal_force_n'])>0) for p in contacts['points'])
        bin_points=any(p['filter_index'] in meta['bin_filter_indices'] and abs(p['normal_force_n'])>0 for p in contacts['points'])
        robot_force=np.asarray(contacts['box_robot_force_matrix_world_n'],float)
        hand_force=np.asarray(contacts['box_hand_force_matrix_world_n'],float)
        per_tick.append({'control_tick':tick,'sim_seconds':tick/50,'actor_origin_xy_error_m':error,
            'contact_qualified_max_displacement_m':max_displacement,'contact_qualified_heading_span_degrees':turn,
            'left_hand_contact':sample['left_hand_contact'],'right_hand_contact':sample['right_hand_contact'],
            'grasp_contact_lift_proxy':held,'hand_force_body_norm_sum_n':float(np.linalg.norm(hand_force,axis=1).sum()),
            'box_inside_bin_prism':sample['box_inside_bin_prism'],'bin_upward_force_n':sample['bin_upward_force_n'],
            'bin_support_contact':bool(sample['bin_upward_force_n']>0 and bin_points),
            'robot_contact_points':bool(robot_points),'robot_force_nonzero':bool(np.any(robot_force)),
            'released_no_sampled_robot_contact':bool(not robot_points and not np.any(robot_force)),
            'released_separation_proved':sample['released_separation_proved'],'placement_ready':sample['placement_ready'],
            'standing':sample['standing'],'fell_so_far':sample['fell'],'ground_drop_this_tick':sample['ground_drop']})
    start=max(0,rows[-1]['control_tick']-500);tail=per_tick[start:]
    predicates=('left_hand_contact','right_hand_contact','grasp_contact_lift_proxy','box_inside_bin_prism',
        'bin_support_contact','robot_contact_points','robot_force_nonzero','released_no_sampled_robot_contact',
        'released_separation_proved','placement_ready','standing','fell_so_far','ground_drop_this_tick')
    counts={key:{'true_samples':sum(bool(r[key]) for r in tail),'samples':len(tail),
                 'all':all(r[key] for r in tail),'any':any(r[key] for r in tail)} for key in predicates}
    closest=int(np.argmin(errors));qualified=judgment['carry_qualified_control_tick']
    if abs(max_displacement-judgment['max_carried_box_displacement_m'])>1e-9 or abs(per_tick[-1]['contact_qualified_heading_span_degrees']-judgment['carry_turn_degrees'])>1e-9:
        raise ValueError('Independent cumulative transport differs from acceptance')
    if rule['minimum_carried_displacement_m']==4 and rule['minimum_carry_turn_degrees']==60 and first_4m60!=qualified:
        raise ValueError('First4m60deg tick differs from frozen acceptance')
    return {'first_grasp_control_tick':first,'carry_qualified_control_tick':qualified,
        'first_4m_60deg_qualified_tick':first_4m60,'transport_condition_is_not_arrival_or_task_success':True,
        'existing_carry_rule':{k:v for k,v in rule.items() if k.startswith('minimum_')},
        'source_target_distance_m':source_distance,'max_carried_box_displacement_m':max_displacement,
        'contact_qualified_carried_path_m':judgment['contact_qualified_carried_path_m'],
        'carry_turn_degrees':judgment['carry_turn_degrees'],
        'first_transport_event':None if qualified is None else {'control_tick':qualified,'sim_seconds':qualified/50,
            'actual_observation_wall_seconds':(rows[qualified]['monotonic_ns']-rows[0]['monotonic_ns'])/1e9,
            'box_target_actor_origin_xy_error_m':errors[qualified]},
        'holding_gaps':{'existing_acceptance_events':judgment['hand_contact_gap_events_before_arrival'],
            'existing_acceptance_largest_gap_seconds':judgment['largest_hand_contact_gap_seconds'],
            'sampled_intervals_after_first_grasp_outside_bin':intervals(gaps),
            'scope':judgment['contact_gap_note']},
        'observed_physical_flags':{k:judgment[k] for k in ('grasped','stable_placement','fell','dropped')},
        'box_target_actor_origin_xy_error_m':{'final':errors[-1],'minimum':errors[closest],'minimum_tick':closest,
            'at_carry_qualified':None if qualified is None else errors[qualified],
            'definition':'WorldXY Euclidean recorded box actor origin minus contemporaneous measured bin actor origin; no binary arrival threshold.'},
        'last10seconds':{'inclusive_ticks':[start,rows[-1]['control_tick']],'sample_count':len(tail),'predicates':counts,
            'actor_origin_xy_error_m':stats([r['actor_origin_xy_error_m'] for r in tail]),
            'hand_force_body_norm_sum_n':stats([r['hand_force_body_norm_sum_n'] for r in tail]),
            'bin_upward_force_n':stats([r['bin_upward_force_n'] for r in tail]),
            'release_note':'No sampled robot contacts and conservative separated geometry are distinct. Existing separation bit is tested only after all placement candidate prerequisites; false alone is not proof of physical contact.'},
        'per_tick_diagnostics':per_tick}


def analyze_case(path):
    path=Path(path).resolve();folder=path if path.is_dir() else path.parent
    config_path=folder/'config.json';result_path=folder/'result.json';runtime_path=folder/'runtime_identity.json';trace_path=folder/'trajectory.jsonl'
    bound={}
    def bind(p):
        p=Path(p).resolve();bound[str(p)]=sha(p);return p
    config=read(bind(config_path));result=read(bind(result_path)) if result_path.exists() else None
    runtime=read(bind(runtime_path)) if runtime_path.exists() else None
    rows=[]
    with bind(trace_path).open() as stream:
        for line in stream:
            if line.strip():rows.append(json.loads(line))
    if not rows:raise ValueError('Empty trace')
    if result and result.get('trace_sha256')!=bound[str(trace_path)]:raise ValueError('Recorded trace SHA mismatch')
    metadata=result or runtime
    if metadata is None:raise ValueError('No persisted runtime contact/body identity')
    context={**config,'audit_contact_metadata':metadata['contact_view'],'audit_body_names':metadata['body_names']}
    for key in ['geometry_definition','robot_geometry']:bind(config[key])
    for p in [acceptance.__file__,__file__]:bind(p)
    judgment=acceptance.evaluate_episode(rows,context)
    complete=bool(result and result.get('status')=='completed' and result.get('source_interface_passed') is True and not result.get('cleanup_errors'))
    if complete:
        for key in ('success','grasped','stable_placement','fell','dropped','timeout'):
            if result.get('acceptance',{}).get(key)!=judgment[key]:raise ValueError('Producer acceptance disagrees: '+key)
    summary=summarize_rows(rows,judgment,context)
    summary.update(schema='pi05_g1_mobile_transport_episode_v1',case_id=config['case']['case_id'],policy=config.get('policy_id',config['mode']),
        result_status=None if result is None else result.get('status'),original_result_success=None if result is None else result.get('success'),
        original_termination_reason=None if result is None else result.get('termination_reason'),
        infrastructure_completed=complete,full_task_verdict=judgment['success'] if complete else None,
        full_task_timeout=judgment['timeout'] if complete else None,
        diagnostic_prefix_only=not complete,control_ticks=rows[-1]['control_tick'],physics_integrations=rows[-1]['physics_steps_since_reset'],
        task_verdict_scope='Existing frozen acceptance only for infrastructure-complete episode. Failed/missing result retains unknown full-task verdict; observed prefix flags are diagnostic.',
        inputs_sha256=bound,no_raw_files_modified=True)
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('Input changed during transport analysis: '+p)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes',nargs='+',type=Path);group.add_argument('--benchmark',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=args.episodes;benchmark=None
    if args.benchmark:
        benchmark=read(args.benchmark)
        if benchmark.get('complete') is not True or benchmark.get('status')!='complete':raise ValueError('Benchmark must be complete')
        paths=[Path(e['result_path']) for e in benchmark['episodes']]
        if len(paths)!=60 or len(set(paths))!=60:raise ValueError('Complete60 unique episode paths required')
    reports=[analyze_case(path) for path in paths]
    report={'schema':'pi05_g1_mobile_transport_analysis_v1','episodes':reports,'source_benchmark':None if args.benchmark is None else {'path':str(args.benchmark.resolve()),'sha256':sha(args.benchmark)},
        'scope':'Descriptive physical transport/contact metrics; no new arrival criterion, threshold edits, causal claim or GPU execution.',
        'tool_sha256':sha(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'episodes':len(reports),'output':str(args.output),'full_task_verdicts':[r['full_task_verdict'] for r in reports]}))

if __name__=='__main__':main()
