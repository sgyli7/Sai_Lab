"""CPU-only final delivery evidence gate. Never modifies scores or raw evidence."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

PRIMARY={'gr00t_n17','pi05_round01','pi05_round02'}
OPTIONAL={'official_pi05_base','pi05_fresh_validation'}
FLAGS=('success','grasped','stable_placement','dropped','fell','timeout')
STAGES=('fell','dropped','missed_grasp','grasp_without_placement','unstable_or_unreleased_placement','success')
CASE_IDS={f'p{p}_seed{s}' for p in range(5) for s in range(40,44)}
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(doc):return hashlib.sha256(json.dumps(doc,sort_keys=True,allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def require(ok,message):
    if not ok:raise ValueError(message)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def equal(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7)


def self_hash(doc):
    value=dict(doc);expected=value.pop('audit_sha256',None)
    require(expected==canonical(value),'Independent audit canonical SHA mismatch')


def stage(row):
    if row['fell']:return 'fell'
    if row['dropped']:return 'dropped'
    if row['success']:return 'success'
    if not row['grasped']:return 'missed_grasp'
    return 'grasp_without_placement' if row.get('source_success_candidate_tick') is None else 'unstable_or_unreleased_placement'


def audit(evaluation_root,benchmark,episode_audits,video_export_receipt,video_index,plot_manifest,gpu_ledger,quiescence_receipt,output):
    root=Path(evaluation_root).resolve();output=Path(output).resolve()
    require(not output.exists(),'Use a new delivery audit path')
    input_sha={};json_inputs={};checks={};errors=[];raw={};traces={};video_manifests={};additional_incomplete=[];export_overall_complete=None
    def load(path):
        path=Path(path).resolve();digest=sha(path);input_sha[str(path)]=digest;json_inputs[str(path)]=digest;return read(path)
    def section(name,fn):
        try:checks[name]={'passed':True,'details':fn()}
        except Exception as error:
            message=type(error).__name__+': '+str(error)
            errors.append(name+': '+message);checks[name]={'passed':False,'error':message}
    report=load(benchmark);benchmark_sha=sha(benchmark);policies=set(report.get('policies',{}));pairs={(p,c) for p in policies for c in CASE_IDS}
    def outcomes():
        require(report.get('schema')=='pi05_g1_round02_multi_policy_benchmark_v2' and report.get('complete') is True and report.get('evidence_qualified') is True,'Formal complete benchmark required')
        require(PRIMARY<=policies and policies<=PRIMARY|OPTIONAL and len(policies-PRIMARY)<=1 and report['policy_count']==len(policies),'Primary three policies required; optional fourth only if complete')
        require(report.get('errors')==[] and report.get('infrastructure_incomplete')==[],'Benchmark contains errors/infrastructure gaps')
        for policy in sorted(policies):
            summary=report['policies'][policy];counts={key:0 for key in FLAGS};stages={key:0 for key in STAGES}
            require(summary.get('completed')==20 and summary.get('expected')==20 and summary.get('missing_or_infrastructure_cases')==[],'Every policy must have twenty completed cases')
            for cid in sorted(CASE_IDS):
                path=root/policy/cid/'result.json';r=load(path);raw[policy,cid]=r
                require(r.get('case_id')==cid and r.get('policy')==policy and r.get('status')=='completed','Raw completed result identity mismatch')
                require(all(type(r.get(key)) is bool for key in FLAGS),'Outcome flags must be explicit booleans')
                require(not r['success'] or (r['grasped'] and r['stable_placement'] and not r['fell'] and not r['dropped'] and not r['timeout']),'Raw success contradicts stages')
                for key in FLAGS:counts[key]+=int(r[key])
                stages[stage(r)]+=1
            require(counts==summary['counts'],'Summary raw outcome counts differ from actual cases')
            require(stages==summary['mutually_exclusive_stage_counts'] and sum(stages.values())==20,'Mutually exclusive failure/success counts differ')
            require(summary['successes']==counts['success'] and equal(summary['success_rate'],counts['success']/20) and equal(summary['task_failure_rate'],(20-counts['success'])/20),'Success/failure rate denominator is inconsistent')
            require(summary.get('meets_16_of_20') is (counts['success']>=16),'Reported task threshold flag differs from fixed 16/20 rule')
        return {'policies':sorted(policies),'cases':len(raw),'same_twenty_case_ids':True}
    section('outcomes_and_failure_counts',outcomes)
    def timing():
        launch_by_config={}
        for path in sorted((root/'configs').glob('*.launch.json')):
            receipt=load(path);argv=receipt.get('command',[])
            if '--config' not in argv:continue
            cp=Path(argv[argv.index('--config')+1]);cp=cp if cp.is_absolute() else ROOT/cp
            launch_by_config.setdefault(cp.resolve(),[]).append((path.resolve(),receipt))
        for policy in sorted(policies):
            summary=report['policies'][policy];measured=summary['measured_wall_time_by_case'];per=summary['per_case_timing']
            require(set(measured)==CASE_IDS and set(per)==CASE_IDS,'All twenty measured wall/per-case timing entries required')
            require(isinstance(summary.get('launch_wall_seconds'),dict) and summary['launch_wall_seconds'].get('count')==20,'Launch wall distribution must contain twenty genuine receipts')
            for cid in sorted(CASE_IDS):
                r=raw[policy,cid];wall=measured[cid];timing=per[cid]
                require(wall.get('complete') is True and wall.get('errors')==[],'Measured wall entry is incomplete')
                cp=root/'configs'/f'{policy}_{cid}.json';config=load(cp)
                require(config['policy_id']==policy and config['case']['case_id']==cid and Path(config['output_dir']).resolve()==root/policy/cid,'Launch config differs from formal case')
                matches=launch_by_config.get(cp.resolve(),[]);require(len(matches)==1,'Missing/ambiguous genuine case launch receipt')
                path,receipt=matches[0]
                require(receipt.get('exit_code')==0 and finite(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds']>=0,'Launch receipt is not a successful measured launch')
                require(Path(timing['launch_wall_receipt']).resolve()==path and equal(timing['launch_wall_seconds'],receipt['elapsed_seconds']),'Summary launch wall does not match genuine receipt')
                trace_path=root/policy/cid/'trajectory.jsonl';input_sha[str(trace_path)]=sha(trace_path)
                rows=[json.loads(line) for line in trace_path.read_text().splitlines()];traces[policy,cid]=rows
                require(len(rows)==r['completed_control_ticks']+1 and all(row['control_tick']==i for i,row in enumerate(rows)),'Trace control coverage mismatch')
                require(wall['trace_sha256']==sha(trace_path),'Measured wall trace SHA differs')
                first,last=rows[0]['monotonic_ns'],rows[-1]['monotonic_ns']
                require(equal(wall['episode_observed_wall_seconds'],(last-first)/1e9),'Episode wall differs from measured trace')
                if r['success']:
                    window=r['qualified_stability_window'];require(window.get('passed') is True,'Successful qualified window missing')
                    tick=window['end_tick'];require(type(tick) is int and 0<=tick<len(rows),'Qualified endpoint outside trace')
                    require(equal(wall['actual_task_completion_wall_seconds'],(rows[tick]['monotonic_ns']-first)/1e9),'Success task wall differs from qualified endpoint')
                    require(equal(timing['task_completion_sim_seconds_success_only'],tick/r['control_hz']),'Success simtime differs from qualified endpoint')
                else:require(wall.get('actual_task_completion_wall_seconds') is None and 'task_completion_sim_seconds_success_only' not in timing,'Failed trial was assigned success speed')
                grasp=r.get('first_grasp_control_tick')
                if grasp is None:require('time_to_first_grasp_sim_seconds' not in timing,'Missing grasp was assigned a grasp time')
                else:require(equal(timing['time_to_first_grasp_sim_seconds'],grasp/r['control_hz']),'Grasp time differs from recorded proxy')
        return {'measured_wall_entries':len(pairs),'genuine_unique_launch_receipts':len(pairs),'success_endpoint_only':True,'release_latency_claimed':False}
    section('timing_and_genuine_launch_receipts',timing)
    def independent():
        seen={}
        for path in episode_audits:
            doc=load(path);self_hash(doc)
            require(doc.get('schema')=='pi05_g1_round02_independent_episode_audit_v1' and doc.get('passed') is True and doc.get('cpu_only') is True,'Complete independent CPU physics audit required')
            for case in doc['cases']:
                self_hash(case);pair=(case['policy'],case['case_id'])
                require(pair in pairs and pair not in seen,'Audit mixes other phases or duplicate cases')
                require(case.get('passed') is True and case.get('anomalies')==[],'Independent case audit contains anomalies')
                require(case.get('audit_tool_sha256')==sha(ROOT/'experiments/pi05_g1_round02/audit_episodes.py'),'Independent audit tool identity differs')
                sections=('telemetry_and_initial_hash','raw_physics_geometry_and_acceptance','action_execution_and_observation_feedback')
                require(all(case['section_passed'].get(key) is True for key in sections),'Physics/action/input audit sections not all passed')
                p=root/pair[0]/pair[1];require(Path(case['case_directory']).resolve()==p,'Audit case directory is outside formal results')
                for relative,digest in case['input_sha256'].items():
                    target=(p/relative).resolve();require(target.is_relative_to(p),'Unexpected external audit input')
                    require(sha(target)==digest,'Audited physics/action/input bytes changed: '+str(target));input_sha[str(target)]=digest
                require(case['input_sha256'].get('result.json')==sha(p/'result.json') and case['input_sha256'].get('trajectory.jsonl')==sha(p/'trajectory.jsonl'),'Audit lacks raw result/trace SHA')
                require(case['recorded_success']==raw[pair]['success'] and case['checks']['raw_physics_geometry_and_acceptance']['success']==raw[pair]['success'],'Independent score differs from raw result')
                telemetry=case['checks']['telemetry_and_initial_hash']
                require(telemetry['initial_schema']=='pi05_g1_initial_kinematic_state_v2' and telemetry['recomputed_initial_hash']==raw[pair]['initial_physical_state_sha256'],'Full kinematic initial-state audit missing')
                seen[pair]=case
        require(set(seen)==pairs,'Independent audit does not cover every formal policy/case')
        for cid in CASE_IDS:
            values={seen[p,cid]['checks']['telemetry_and_initial_hash']['recomputed_initial_hash'] for p in policies}
            require(len(values)==1,'Actual recomputed initial states differ across policies')
        summary_checks=report['actual_initial_pair_checks'];require(len(summary_checks)==20,'Summary initial-state checks incomplete')
        for check in summary_checks:
            require(check['case_id'] in CASE_IDS and check['all_policies_match'] is True and check['policies_expected']==len(policies),'Summary initial-state pair is ineligible')
            require(check['actual_initial_hashes']=={p:raw[p,check['case_id']]['initial_physical_state_sha256'] for p in policies},'Summary hashes differ from actual independently audited initial states')
        return {'audited_cases':len(seen),'initial_pairs':20,'physics_action_input_anomalies':0}
    section('independent_physics_action_input_and_actual_initial_pairs',independent)
    def videos():
        nonlocal export_overall_complete
        exported=load(video_export_receipt)
        require(exported.get('schema')=='pi05_g1_round02_cpu_video_export_batch_v1','CPU video export receipt schema differs')
        additional_incomplete.extend(exported.get('additional_incomplete',[]));export_overall_complete=exported.get('complete') is True
        if policies==PRIMARY:
            require(exported.get('primary_complete',export_overall_complete) is True,'Primary sixty video exports are incomplete')
            require(export_overall_complete or bool(additional_incomplete),'Incomplete export needs explicit additional-phase failure disclosure')
        else:
            require(export_overall_complete and exported.get('status')=='complete' and exported.get('errors')==[] and not additional_incomplete,
                    'A four-policy delivery requires its complete supplementary twenty exports')
        require(exported.get('tool_sha256')==sha(ROOT/'experiments/pi05_g1_round02/batch_video_export.py') and exported.get('exporter_sha256')==sha(ROOT/'experiments/pi05_g1_round02/video_export.py'),'CPU video exporter source receipt differs')
        require(exported.get('gpu_ledger_sha256')==sha(gpu_ledger),'GPU ledger changed since post-GPU video export')
        export_jobs={(j['job']['policy'],j['job']['case_id']):j for j in exported['jobs']}
        require(len(export_jobs)==len(exported['jobs']) and pairs<=set(export_jobs),'Video export receipt lacks formal cases or duplicates jobs')
        complete_policies={p for p,c in export_jobs if all(export_jobs.get((p,cid),{}).get('status')=='complete' for cid in CASE_IDS)}
        declared_deliverable=set(exported.get('deliverable_policies',complete_policies))
        require(declared_deliverable==complete_policies and policies<=declared_deliverable,'Declared video deliverable scope differs from completed case receipts')
        require(exported.get('deliverable_cases',20*len(complete_policies))==20*len(complete_policies),'Deliverable video case count differs')
        attempted_optional=(complete_policies-PRIMARY)|{item['policy'] for item in additional_incomplete}
        require(attempted_optional<=OPTIONAL and len(attempted_optional)<=1,'Official base and fresh supplementary policies are mutually exclusive')
        index_path=Path(video_index).resolve()
        if index_path.suffix.lower()=='.csv':index_path=index_path.with_suffix('.provenance.json')
        index=load(index_path)
        require(index.get('schema')=='pi05_g1_round02_full_video_csv_index_v1' and index.get('summary_sha256')==benchmark_sha
                and Path(index['summary_path']).resolve()==Path(benchmark).resolve(),'Native CSV provenance does not bind formal benchmark')
        csv_path=Path(index['csv_path']).resolve();require(index.get('csv_sha256')==sha(csv_path),'Native video index CSV SHA mismatch')
        input_sha[str(csv_path)]=sha(csv_path)
        if Path(video_index).suffix.lower()=='.csv':require(csv_path==Path(video_index).resolve(),'CSV argument differs from provenance csv_path')
        from .video_index import FIELDS
        with csv_path.open(newline='') as stream:
            reader=csv.DictReader(stream);require(tuple(reader.fieldnames or ())==FIELDS,'Video index CSV columns differ from native producer')
            csv_rows=list(reader)
        require(index.get('indexed_video_rows')==len(csv_rows),'Native CSV row count differs from provenance')
        entries={(e['policy'],e['case_id']):e for e in index['manifests']}
        require(len(entries)==len(index['manifests']) and set(entries)==pairs and index.get('indexed_cases')==len(pairs),
                'Native CSV provenance must cover every formal policy/case exactly once')
        require(index.get('policy_case_counts')=={p:20 for p in policies},'Native provenance policy case counts differ')
        by_case={pair:{} for pair in pairs}
        for row in csv_rows:
            pair=(row['policy'],row['case_id']);kind=row['video_kind']
            require(pair in by_case and kind not in by_case[pair],'Foreign or duplicate CSV video view')
            by_case[pair][kind]=row
        representative=[]
        for pair,entry in entries.items():
            policy,cid=pair;job=export_jobs[pair];require(job.get('status')=='complete','Case export incomplete')
            config=load(root/'configs'/f'{policy}_{cid}.json');mp=Path(config['video_output_dir']).resolve()/'video_manifest.json'
            require(mp.parent==root/'video_exports'/policy/cid and job['job']['config_sha256']==sha(root/'configs'/f'{policy}_{cid}.json'),'Video export config identity differs from final external destination')
            require(Path(entry['manifest_path']).resolve()==mp and entry['manifest_sha256']==sha(mp),'Native provenance manifest path/hash differs from config export')
            manifest=load(mp);video_manifests[pair]=manifest;require(manifest['schema']=='pi05_g1_round02_video_manifest_v1' and manifest['policy']==policy and manifest['case_id']==cid,'Video manifest identity differs')
            digest=sha(root/policy/cid/'result.json');require(entry['raw_result_sha256']==digest==manifest['raw_result_sha256']==job['job']['result_sha256'],'Raw result SHA binding differs across index/export/manifest')
            require(manifest['trace_sha256']==sha(root/policy/cid/'trajectory.jsonl'),'Video trace SHA differs')
            required={'head_raw','observer_raw','head_wall_annotated','observer_wall_annotated'}
            if cid=='p0_seed40':required|={'head_wall_clock','observer_wall_clock'};representative.append(policy)
            require(set(manifest['videos'])==required and set(by_case[pair])==required,'Standard/representative CSV view count or manifest coverage differs')
            require(job['evidence']['manifest_sha256']==sha(mp),'Verified batch manifest SHA no longer matches')
            for key in required:
                value=manifest['videos'][key];path=Path(value['path']);path=path if path.is_absolute() else mp.parent/path
                require(sha(path)==value['sha256']==job['evidence']['videos'][key]['sha256'],'Delivered video bytes differ from verified export')
                row=by_case[pair][key]
                require(row['view']==('head' if key.startswith('head') else 'observer') and row['playback_kind']==('measured_wall_clock' if key.endswith('wall_clock') else 'simulation_time'),'CSV view/playback kind differs from native meaning')
                require(Path(row['video_path']).resolve()==path.resolve() and row['video_sha256']==value['sha256'],'CSV path/SHA differs from its manifest video')
                require(equal(float(row['fps']),value['fps']) and int(row['frames'])==value['frames'],'CSV FPS/frame count differs from manifest')
                require(Path(row['raw_result_path']).resolve()==root/policy/cid/'result.json' and row['raw_result_sha256']==digest,'CSV raw result identity differs')
                require(Path(row['video_manifest_path']).resolve()==mp and row['video_manifest_sha256']==sha(mp),'CSV manifest binding differs')
                require(Path(row['benchmark_summary_path']).resolve()==Path(benchmark).resolve() and row['benchmark_summary_sha256']==benchmark_sha,'CSV row benchmark binding differs')
                if key.endswith('wall_clock'):require(value['playback_kind']=='measured_wall_clock_resampled' and value['fps']==25,'Representative wall video is only a subtitle replay')
        require(set(representative)=={p for p,c in entries},'Each indexed policy needs frozen p0_seed40 real wall video pair')
        return {'index_schema':index['schema'],'csv_path':str(csv_path),'csv_sha256':index['csv_sha256'],'indexed_video_rows':len(csv_rows),
                'indexed_cases':len(entries),'formal_benchmark_cases':len(pairs),
                'primary_complete':exported.get('primary_complete',export_overall_complete),'overall_export_complete':export_overall_complete,
                'additional_incomplete':list(additional_incomplete),'export_errors':exported.get('errors',[]),'deliverable_policies':sorted(complete_policies),
                'representative_wall_case':'p0_seed40','representative_policies':sorted(representative)}
    section('video_exports_index_and_representative_actual_wall',videos)
    def plots():
        path=Path(plot_manifest).resolve();doc=load(path)
        require(doc.get('schema')=='pi05_g1_round02_static_plot_export_v1' and doc['summary_sha256']==benchmark_sha and Path(doc['summary_path']).resolve()==Path(benchmark).resolve(),'Plots do not bind formal summary')
        require(doc.get('trials_per_policy')==20 and doc.get('position_clusters')==5 and doc.get('simulation_only') is True and set(doc['policies'])==policies,'Plot cohort/scope differs')
        require('Matplotlib' in doc.get('render_backend',''),'Vector Matplotlib plot artifact required')
        required={'benchmark.png','benchmark.pdf','matched_success_speed.csv'};require(required<=set(doc['files']),'PNG/PDF/success-only speed table incomplete')
        for name in required:
            target=path.parent/name;require(sha(target)==doc['files'][name]['sha256'] and target.stat().st_size==doc['files'][name]['bytes'],'Plot/table artifact hash or size differs')
        require((path.parent/'benchmark.pdf').read_bytes().startswith(b'%PDF'),'PDF output invalid')
        dependency=Path(doc['dependency_receipt']);require(doc.get('dependency_receipt_sha256')==sha(dependency),'Pinned plotting dependency receipt changed/missing');load(dependency)
        with (path.parent/'matched_success_speed.csv').open() as stream:rows=list(csv.DictReader(stream))
        comparisons={(c['first_policy'],c['second_policy']):c for c in report['comparisons']}
        require(len(rows)==len(comparisons),'Speed table pair count differs')
        for row in rows:
            c=comparisons[row['first_policy'],row['second_policy']];count=len(c['matched_success_case_ids'])
            require(int(row['matched_success_count'])==count,'Speed table changes matched-success cohort')
            if not count:require(row['mean_second_minus_first_sim_seconds']=='N/A' and row['mean_second_minus_first_actual_task_wall_seconds']=='N/A','No-common-success speed must be N/A')
            else:
                require(equal(float(row['mean_second_minus_first_sim_seconds']),c['matched_success_sim_delta_seconds']['mean']) and equal(float(row['mean_second_minus_first_actual_task_wall_seconds']),c['matched_success_actual_task_wall_delta_seconds']['mean']),'Speed table differs from successful-pair summary')
        return {'summary_sha256':benchmark_sha,'files':sorted(required),'speed_uses_matched_success_only':True}
    section('plots_and_matched_success_speed_table',plots)
    def budget():
        ledger=load(gpu_ledger);require(ledger.get('schema')=='pi05_g1_gpu_budget_v1','GPU ledger schema differs')
        require(ledger.get('total_limit_seconds')==28800,'Additional-round budget must be eight hours')
        total=0.;latest=None
        for r in ledger['runs']:
            require(r.get('status') not in ('running','cleanup_pending','preparing') and not r.get('accounting_open') and not r.get('launcher_cleanup_pending'),'GPU job/accounting/cleanup remains open')
            require(finite(r.get('elapsed_seconds')) and r['elapsed_seconds']>=0,'Actual elapsed budget accounting missing')
            total+=r['elapsed_seconds'];stamp=r.get('supervisor_ended_utc',r.get('ended_utc'))
            require(isinstance(stamp,str),'Closed run end timestamp absent');dt=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00'))
            latest=max(latest,dt) if latest else dt
        require(total<=28800+1e-6,'Actual accumulated additional GPU occupancy exceeds eight hours')
        quiet=load(quiescence_receipt)
        require(quiet.get('schema')=='pi05_g1_round02_gpu_quiescence_v1' and quiet.get('passed') is True and quiet['ledger_sha256']==sha(gpu_ledger),'Quiescence receipt not bound to current ledger')
        require(quiet.get('active_owned_units')==[] and quiet.get('active_owned_container_ids')==[] and quiet.get('accounting_open_tokens')==[]
                and quiet.get('errors')==[] and quiet.get('unexpected_compute_processes')==[],'Quiescence receipt reports remaining owned resources')
        checked=datetime.datetime.fromisoformat(quiet['checked_utc'].replace('Z','+00:00'));require(latest is None or checked>=latest,'Quiescence evidence predates last GPU closure')
        require(equal(quiet['total_gpu_seconds'],total),'Quiescence total differs from actual ledger')
        require(isinstance(quiet.get('evidence_files'),dict) and bool(quiet['evidence_files']),'Actual service/container probe evidence required')
        for path,digest in quiet['evidence_files'].items():
            require(Path(path).is_absolute() and sha(path)==digest,'Quiescence raw probe evidence SHA mismatch');input_sha[path]=digest
        return {'actual_total_gpu_seconds':total,'limit_seconds':28800,'ledger_closed':True,'quiescence_checked_utc':quiet['checked_utc'],'system_state_scope':'Current resource absence is attested by the supplied actual system-probe receipt; this tool does not invent probe outputs.'}
    section('eight_hour_actual_budget_and_quiescence_receipt',budget)
    for path,digest in json_inputs.items():
        try:
            if sha(path)!=digest:errors.append('JSON input changed during audit: '+path)
        except OSError as error:errors.append('JSON input became unavailable during audit: '+path+': '+str(error))
    final={'schema':'pi05_g1_round02_delivery_audit_v1','passed':not errors and all(c['passed'] for c in checks.values()),
           'checks':checks,'errors':errors,'input_sha256':input_sha,'cpu_only':True,'scores_modified':False,
           'delivery_scope_policies':sorted(policies),'additional_incomplete':additional_incomplete,'all_attempted_video_exports_complete':export_overall_complete,
           'scope':'Evidence completeness and consistency only; no physical benefit/causal interpretation or score changes. Grasp proxy and qualified endpoint are used; source candidate is not a release timestamp.',
           'tool_sha256':sha(__file__),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    final['audit_sha256']=canonical(final);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(final,indent=2,allow_nan=False)+'\n')
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('evaluation-root','benchmark','video-export-receipt','plot-manifest','gpu-ledger','quiescence-receipt','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--video-index',type=Path,required=True,help='Native video_index CSV or its .provenance.json sidecar')
    p.add_argument('--episode-audit',type=Path,action='append',required=True);a=p.parse_args()
    report=audit(a.evaluation_root,a.benchmark,a.episode_audit,a.video_export_receipt,a.video_index,a.plot_manifest,a.gpu_ledger,a.quiescence_receipt,a.output)
    print(json.dumps({'passed':report['passed'],'errors':report['errors']}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
