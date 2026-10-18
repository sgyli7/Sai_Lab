"""Synthetic metadata/opaque media fixtures; never execute physics or GPU code."""
import csv
import datetime
import subprocess
from unittest.mock import patch
from types import SimpleNamespace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pi05_g1_round02.delivery_audit import audit,sha,canonical,ROOT,PRIMARY
from experiments.pi05_g1_round02.video_index import make_index


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def seal(value):value['audit_sha256']=canonical(value);return value


def fixture(root,supplementary=False):
    policies=sorted(PRIMARY|({'pi05_fresh_validation'} if supplementary else set()));cases=[f'p{p}_seed{s}' for p in range(5) for s in range(40,44)]
    report={'schema':'pi05_g1_round02_multi_policy_benchmark_v2','complete':True,'evidence_qualified':True,'errors':[],
        'infrastructure_incomplete':[],'policy_count':len(policies),'policies':{},'actual_initial_pair_checks':[],
        'video_evidence':[],
        'comparisons':[{'first_policy':a,'second_policy':b,'matched_success_case_ids':[],'matched_success_speed':[],
                        'matched_success_sim_delta_seconds':None,'matched_success_actual_task_wall_delta_seconds':None} for a,b in itertools.combinations(policies,2)]}
    independent=[];video_jobs=[];batch_jobs=[];batch_records=[];config_hashes={}
    for cid in cases:report['actual_initial_pair_checks'].append({'case_id':cid,'all_policies_match':True,'policies_expected':len(policies),'actual_initial_hashes':{p:f'{int(cid[1]):064x}' for p in policies}})
    for policy in policies:
        measured={};per={}
        for cid in cases:
            p=root/policy/cid;p.mkdir(parents=True);positions=((0,0),(.02,0),(-.02,0),(0,.02),(0,-.02));config={'policy_id':policy,'case':{'case_id':cid,'seed':int(cid.split('_seed')[1]),'apple_offset_xy_m':list(positions[int(cid[1])])},'output_dir':str(p),'video_output_dir':str(root/'video_exports'/policy/cid)}
            cp=root/'configs'/f'{policy}_{cid}.json';write(cp,config);write(p/'config.json',config)
            result={'case_id':cid,'case':config['case'],'policy':policy,'status':'completed','success':False,'grasped':False,'stable_placement':False,'fell':False,'dropped':False,
                    'timeout':True,'source_success_candidate_tick':None,'first_grasp_control_tick':None,'completed_control_ticks':1,'control_hz':50,
                    'initial_physical_state_sha256':f'{int(cid[1]):064x}'}
            write(p/'result.json',result);name=policy+'_'+cid;config_hashes[str(cp)]=sha(cp)
            batch_jobs.append({'name':name,'config_path':str(cp),'config_sha256':sha(cp)})
            batch_records.append({'name':name,'config_file_sha256':sha(cp),'status':'completed','output':str(p),'result_sha256':sha(p/'result.json')})
            trace=p/'trajectory.jsonl';trace.write_text(json.dumps({'control_tick':0,'monotonic_ns':1000})+'\n'+json.dumps({'control_tick':1,'monotonic_ns':1000001000})+'\n')
            lp=cp.with_suffix('.launch.json');write(lp,{'command':['python','--config',str(cp)],'exit_code':0,'elapsed_seconds':2.})
            measured[cid]={'complete':True,'errors':[],'trace_sha256':sha(trace),'episode_observed_wall_seconds':1.,'actual_task_completion_wall_seconds':None}
            per[cid]={'launch_wall_receipt':str(lp),'launch_wall_seconds':2.}
            case=seal({'case_directory':str(p),'case_id':cid,'policy':policy,'passed':True,'anomalies':[],'recorded_success':False,
                'audit_tool_sha256':sha(ROOT/'experiments/pi05_g1_round02/audit_episodes.py'),
                'section_passed':{key:True for key in ['telemetry_and_initial_hash','raw_physics_geometry_and_acceptance','action_execution_and_observation_feedback']},
                'checks':{'telemetry_and_initial_hash':{'initial_schema':'pi05_g1_initial_kinematic_state_v2','recomputed_initial_hash':result['initial_physical_state_sha256']},'raw_physics_geometry_and_acceptance':{'success':False}},
                'input_sha256':{'result.json':sha(p/'result.json'),'trajectory.jsonl':sha(trace)}});independent.append(case)
            dest=Path(config['video_output_dir']);dest.mkdir(parents=True);videos={}
            keys=['head_raw','observer_raw','head_wall_annotated','observer_wall_annotated']+(['head_wall_clock','observer_wall_clock'] if cid=='p0_seed40' else [])
            for key in keys:
                path=(p/('head_50fps.mp4' if key=='head_raw' else 'observer_50fps.mp4')) if key.endswith('_raw') else dest/(key+'.mp4');path.write_bytes(b'opaque synthetic media - no actual exporter executed')
                videos[key]={'path':str(path),'sha256':sha(path),'fps':25 if key.endswith('wall_clock') else 50,'frames':26 if key.endswith('wall_clock') else 2,'wall_annotation':not key.endswith('_raw'),'playback_kind':'measured_wall_clock_resampled' if key.endswith('wall_clock') else ('simulation_time_raw' if key.endswith('_raw') else 'simulation_time_with_wall_subtitles')}
            mp=dest/'video_manifest.json';write(mp,{'schema':'pi05_g1_round02_video_manifest_v1','policy':policy,'case_id':cid,'raw_result_sha256':sha(p/'result.json'),'trace_sha256':sha(trace),'videos':videos,'source_case_dir':str(p),'timing':{'total_preserved_wall_seconds':1.}})
            report['video_evidence'].append({'policy':policy,'case_id':cid,'complete':True,'errors':[],'manifest_path':str(mp),'manifest_sha256':sha(mp),'videos':{k:v for k,v in videos.items() if not k.endswith('wall_clock')},'actual_wall_clock_videos':{k:v for k,v in videos.items() if k.endswith('wall_clock')}})
            video_jobs.append({'status':'complete','job':{'policy':policy,'case_id':cid,'result_sha256':sha(p/'result.json'),'config_sha256':sha(cp)},'evidence':{'manifest_sha256':sha(mp),'videos':{key:{'sha256':v['sha256']} for key,v in videos.items()}}})
        report['policies'][policy]={'completed':20,'expected':20,'missing_or_infrastructure_cases':[],'counts':{'success':0,'grasped':0,'stable_placement':0,'dropped':0,'fell':0,'timeout':20},
            'mutually_exclusive_stage_counts':{'fell':0,'dropped':0,'missed_grasp':20,'grasp_without_placement':0,'unstable_or_unreleased_placement':0,'success':0},
            'successes':0,'success_rate':0.,'task_failure_rate':1.,'meets_16_of_20':False,'success_wilson_95ci':[0.,.17],'warm_rpc_wall_seconds':{'count':140,'p50':.2,'p95':.3},'measured_wall_time_by_case':measured,'per_case_timing':per,'launch_wall_seconds':{'count':20}}
    report.update(physical_positions=5,noise_repetitions_per_position=4)
    bench=root/'benchmark.json';write(bench,report)
    exporter=ROOT/'experiments/pi05_g1_round02/video_export.py'
    primary=root/'primary_plan.json';write(primary,{'schema':'pi05_g1_round02_frozen_benchmark_plan_v1','phase':'primary','policies':policies,'jobs':batch_jobs,'frozen_files':{str(exporter):sha(exporter)}})
    write(root/'primary_batch_receipt.json',{'schema':'pi05_g1_round02_batch_receipt_v1','status':'completed','plan':str(primary),'plan_sha256':sha(primary),'frozen_files':config_hashes,'jobs':batch_records})
    physics=root/'physics.json';write(physics,seal({'schema':'pi05_g1_round02_independent_episode_audit_v1','cpu_only':True,'passed':True,'cases':independent}))
    export=root/'video_export.json';write(export,{'schema':'pi05_g1_round02_cpu_video_export_batch_v1','complete':True,'status':'complete','errors':[],'jobs':video_jobs,'tool_sha256':sha(ROOT/'experiments/pi05_g1_round02/batch_video_export.py'),'exporter_sha256':sha(ROOT/'experiments/pi05_g1_round02/video_export.py')})
    index=root/'video_index.provenance.json';make_index(bench,root/'video_index.csv')
    plots=root/'plots';plots.mkdir();(plots/'benchmark.png').write_bytes(b'synthetic png');(plots/'benchmark.pdf').write_bytes(b'%PDF-1.7 synthetic vector fixture')
    with (plots/'matched_success_speed.csv').open('w',newline='') as stream:
        fields=['first_policy','second_policy','matched_success_count','mean_second_minus_first_sim_seconds','mean_second_minus_first_actual_task_wall_seconds'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for pair in report['comparisons']:w.writerow({**{k:pair[k] for k in ['first_policy','second_policy']},'matched_success_count':0,'mean_second_minus_first_sim_seconds':'N/A','mean_second_minus_first_actual_task_wall_seconds':'N/A'})
    dependency=root/'plotenv.json';write(dependency,{'synthetic':True})
    manifest=plots/'manifest.json';write(manifest,{'schema':'pi05_g1_round02_static_plot_export_v1','summary_sha256':sha(bench),'summary_path':str(bench),
        'trials_per_policy':20,'position_clusters':5,'simulation_only':True,'policies':policies,'render_backend':'Matplotlib synthetic metadata fixture',
        'dependency_receipt':str(dependency),'dependency_receipt_sha256':sha(dependency),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in plots.iterdir()}})
    ledger=root/'ledger.json';write(ledger,{'schema':'pi05_g1_gpu_budget_v1','total_limit_seconds':28800,'runs':[{'status':'completed','accounting_open':False,'elapsed_seconds':100.,'supervisor_ended_utc':(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=1)).isoformat(),'ended_utc':(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=1)).isoformat(),'budget_token':'a'*32,'owned_unit':'pi05-g1-owned-'+('a'*32)+'.service','docker_cleanup':{'complete':True,'containers':[]}}]})
    exported=json.loads(export.read_text());exported['gpu_ledger_sha256']=sha(ledger);write(export,exported)
    evidence=root/'actual_probe.txt';evidence.write_text('synthetic quiescence probe fixture; no running units or containers')
    quiet=root/'quiescence.json';write(quiet,{'schema':'pi05_g1_round02_gpu_quiescence_v1','passed':True,'ledger_sha256':sha(ledger),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'errors':[],'unexpected_compute_processes':[],
        'active_owned_units':[],'active_owned_container_ids':[],'accounting_open_tokens':[],'total_gpu_seconds':100.,'evidence_files':{str(evidence):sha(evidence)}})
    return {'evaluation_root':root,'benchmark':bench,'episode_audits':[physics],'video_export_receipt':export,'video_index':index,'plot_manifest':manifest,'gpu_ledger':ledger,'quiescence_receipt':quiet,'output':root/'delivery.json'}


class DeliveryAuditTests(unittest.TestCase):
    def test_complete_evidence_can_pass_even_all_tasks_failed_without_changing_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');before=args['benchmark'].read_bytes();report=audit(**args)
            self.assertTrue(report['passed'],report['errors']);self.assertFalse(report['scores_modified']);self.assertEqual(len(report['checks']),6)
            self.assertEqual(args['benchmark'].read_bytes(),before)
            with self.assertRaises(ValueError):audit(**args)

    def test_native_csv_provenance_rejects_missing_view_even_if_csv_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');sidecar=args['video_index'];provenance=json.loads(sidecar.read_text())
            self.assertEqual(provenance['schema'],'pi05_g1_round02_full_video_csv_index_v1')
            self.assertEqual(provenance['indexed_video_rows'],246)
            csv_path=Path(provenance['csv_path'])
            with csv_path.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            with csv_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows[1:])
            provenance.update(csv_sha256=sha(csv_path),indexed_video_rows=len(rows)-1);write(sidecar,provenance)
            args['video_index']=csv_path  # Both native CSV and native sidecar arguments are supported.
            report=audit(**args);self.assertFalse(report['passed'])
            self.assertIn('CSV view count',report['checks']['video_exports_index_and_representative_actual_wall']['error'])

    def test_real_index_batch_plot_and_quiescence_producer_interfaces(self):
        from experiments.pi05_g1_round02.batch_video_export import run as batch_export
        from experiments.pi05_g1_round02.quiescence_probe import probe
        plot_python=ROOT/'.scratch/pi05_g1/round02/plot_venv/bin/python'
        if not plot_python.exists():self.skipTest('Isolated Matplotlib runtime is not present')
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');root=args['evaluation_root']
            # make_index was truly called in fixture; its native CSV+sidecar are consumed unchanged.
            def mock_decode_validation(job):
                mp=Path(job['output'])/'video_manifest.json';m=json.loads(mp.read_text())
                return {'manifest':str(mp),'manifest_sha256':sha(mp),'videos':{k:{'path':v['path'],'sha256':v['sha256'],'fps':v['fps'],'frames':v['frames']} for k,v in m['videos'].items()}}
            with patch('experiments.pi05_g1_round02.batch_video_export.verify_export',side_effect=mock_decode_validation),patch('builtins.print'):
                export_receipt,exported=batch_export(root,args['gpu_ledger'],workers=2)
            self.assertTrue(exported['complete']);self.assertEqual(len(exported['jobs']),60)
            args['video_export_receipt']=export_receipt
            def mock_system_queries(argv,**kwargs):
                if argv[0]=='systemctl':out='ActiveState=inactive\nSubState=dead\nResult=success\nLoadState=loaded\n'
                elif argv[0] in ('docker','nvidia-smi'):out=''
                else:raise AssertionError('Unexpected system query '+str(argv))
                return SimpleNamespace(returncode=0,stdout=out,stderr='')
            quiet=root/'real_quiescence.json'
            with patch('experiments.pi05_g1_round02.quiescence_probe.subprocess.run',side_effect=mock_system_queries):
                closure=probe(args['gpu_ledger'],quiet)
            self.assertTrue(closure['passed'],closure['errors']);args['quiescence_receipt']=quiet
            # Actual Matplotlib exporter, separate CPU-only plotting venv, using synthetic summary only.
            real_plots=root/'real_matplotlib_plots'
            completed=subprocess.run([str(plot_python),'-m','experiments.pi05_g1_round02.plot_benchmark','--summary',str(args['benchmark']),'--output',str(real_plots)],
                cwd=ROOT,capture_output=True,text=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            args['plot_manifest']=real_plots/'manifest.json';args['output']=root/'cross_tool_delivery.json'
            report=audit(**args);self.assertTrue(report['passed'],report['errors'])
            self.assertEqual(report['checks']['video_exports_index_and_representative_actual_wall']['details']['indexed_video_rows'],246)

    def test_partial_supplement_does_not_invalidate_main60_but_is_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');path=args['video_export_receipt'];exported=json.loads(path.read_text())
            failure={'phase':'supplementary_fresh','status':'incomplete','policy':'pi05_fresh_validation','expected_cases':20,'complete_exports':0,'error':'Budget stage interrupted'}
            exported.update(status='incomplete',complete=False,primary_complete=True,additional_incomplete=[failure],
                deliverable_policies=sorted(PRIMARY),deliverable_cases=60,errors=['Supplementary phase incomplete: budget interruption'])
            write(path,exported);report=audit(**args)
            self.assertTrue(report['passed'],report['errors']);self.assertEqual(report['delivery_scope_policies'],sorted(PRIMARY))
            self.assertEqual(report['additional_incomplete'],[failure]);self.assertFalse(report['all_attempted_video_exports_complete'])
            details=report['checks']['video_exports_index_and_representative_actual_wall']['details']
            self.assertTrue(details['primary_complete']);self.assertTrue(details['export_errors'])

    def test_complete_fresh_fourth_policy_is_accepted_and_optional_base_is_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final',supplementary=True)
            report=audit(**args);self.assertTrue(report['passed'],report['errors'])
            self.assertEqual(report['checks']['video_exports_index_and_representative_actual_wall']['details']['indexed_video_rows'],328)
            exported=json.loads(args['video_export_receipt'].read_text())
            exported['additional_incomplete']=[{'policy':'official_pi05_base','phase':'optional_base','status':'incomplete'}]
            write(args['video_export_receipt'],exported);args['output']=args['evaluation_root']/'mutually_exclusive.json'
            self.assertFalse(audit(**args)['passed'])

    def test_missing_launch_receipt_or_count_blocks_delivery_despite_complete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');next((args['evaluation_root']/'configs').glob('*.launch.json')).unlink()
            report=audit(**args);self.assertFalse(report['passed']);self.assertFalse(report['checks']['timing_and_genuine_launch_receipts']['passed'])

    def test_missing_launch_distribution_and_actual_budget_overrun_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');benchmark=json.loads(args['benchmark'].read_text())
            benchmark['policies']['gr00t_n17']['launch_wall_seconds']=None;write(args['benchmark'],benchmark)
            ledger=json.loads(args['gpu_ledger'].read_text());ledger['runs'][0]['elapsed_seconds']=28801.;write(args['gpu_ledger'],ledger)
            report=audit(**args);self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['timing_and_genuine_launch_receipts']['passed'])
            self.assertIn('eight hours',report['checks']['eight_hour_actual_budget_and_quiescence_receipt']['error'])

    def test_audit_anomaly_video_tamper_and_open_budget_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp)/'round02/final');physics=json.loads(args['episode_audits'][0].read_text());physics['cases'][0]['anomalies']=['actual action mismatch'];write(args['episode_audits'][0],physics)
            video=next((args['evaluation_root']/'video_exports').rglob('head_wall_clock.mp4'));video.write_bytes(b'changed video')
            ledger=json.loads(args['gpu_ledger'].read_text());ledger['runs'][0]['accounting_open']=True;write(args['gpu_ledger'],ledger)
            report=audit(**args);self.assertFalse(report['passed'])
            for key in ['independent_physics_action_input_and_actual_initial_pairs','video_exports_index_and_representative_actual_wall','eight_hour_actual_budget_and_quiescence_receipt']:
                self.assertFalse(report['checks'][key]['passed'])


if __name__=='__main__':unittest.main()
