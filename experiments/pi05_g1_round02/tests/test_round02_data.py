import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import av
from PIL import Image
import numpy as np
import pyarrow.parquet as pq
from experiments.pi05_g1.contract import DATASET_NAMES,normalize,denormalize,decode_to_source
from experiments.pi05_g1.placement import acceptance_identity
from experiments.pi05_g1_round02.prepare_inputs import make_cases,make_source_fallback,sha
from experiments.pi05_g1_round02.teacher_dataset import extract_samples,duplicate_groups,convert
from experiments.pi05_g1_round02.benchmark import aggregate,failure_stage,check_videos,load_benchmark_results,measured_wall_timing,exact_cluster_sign_flip

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'.scratch/pi05_g1/round02'

class Round02DataTests(unittest.TestCase):
    def test_exact_five_position_cluster_sign_flip_extreme_and_zero(self):
        positions=[(i,0) for i in range(5) for _ in range(4)]
        positive=exact_cluster_sign_flip([False]*20,[True]*20,positions)
        self.assertEqual(positive['cluster_count'],5);self.assertEqual(positive['enumerated_sign_patterns'],32)
        self.assertEqual(positive['two_sided_p'],.0625);self.assertEqual(positive['minimum_attainable_two_sided_p'],.0625)
        negative=exact_cluster_sign_flip([True]*20,[False]*20,positions)
        self.assertEqual(negative['two_sided_p'],positive['two_sided_p'])
        zero=exact_cluster_sign_flip([False]*20,[False]*20,positions)
        self.assertEqual(zero['mean_success_difference'],0.);self.assertEqual(zero['two_sided_p'],1.)
        self.assertEqual(exact_cluster_sign_flip([False,True]*10,[True,False]*10,positions)['two_sided_p'],1.)

    def test_frozen_cases_are_disjoint_and_reserved(self):
        docs=make_cases();final=docs['final_cases.json']['cases'];dev=docs['development_cases.json']['cases'];teacher=docs['teacher_cases.json']
        self.assertEqual([len(final),len(dev),len(teacher['cases'])],[20,4,12])
        self.assertEqual({r['seed'] for r in final},{40,41,42,43})
        self.assertTrue({r['seed'] for r in final}.isdisjoint({r['seed'] for r in dev+teacher['cases']}))
        self.assertEqual(len(teacher['train_case_ids']),9);self.assertEqual(len(teacher['validation_case_ids']),3)
        self.assertTrue(all(r['seed']==203 for r in teacher['cases'] if r['case_id'] in teacher['validation_case_ids']))
        fallback=make_source_fallback();self.assertEqual(len({tuple(r['apple_offset_xy_m']) for r in fallback['cases']}),12)
        self.assertEqual({r['seed'] for r in fallback['cases'] if r['case_id'] in fallback['validation_case_ids']},{213,214,215})

    def test_actual_57_65_sparse_targets_roundtrip_and_asymmetric_bounds(self):
        if not (ART/'norm_stats_v2.json').exists():self.skipTest('Frozen local data artifacts unavailable')
        parent=ART.parent;manifest=json.loads((parent/'prepared/split_manifest.json').read_text());old=json.loads((parent/'prepared/train_norm_stats.json').read_text());new=json.loads((ART/'norm_stats_v2.json').read_text())
        ids=set(manifest['train']);raw=[]
        for r in manifest['files']:
            if r['episode_id'] in ids:raw.extend(pq.read_table(Path(manifest['dataset_root'])/r['parquet'],columns=['teleop.navigate_command']).column(0).to_pylist())
        nav=np.asarray(raw,np.float32);values=np.zeros((len(nav),32),np.float32);values[:,29:32]=nav
        old_restored=denormalize(normalize(values,old['actions']),old['actions']);new_restored=denormalize(normalize(values,new['actions']),new['actions'])
        for j,count in ((29,57),(30,65)):
            rare=values[:,j]!=0;self.assertEqual(int(rare.sum()),count)
            self.assertGreater(float(np.max(np.abs(old_restored[rare,j]-values[rare,j]))),1e-3)
            self.assertLessEqual(float(np.max(np.abs(new_restored[rare,j]-values[rare,j]))),2e-8)
            self.assertEqual(float(normalize(np.zeros(32,np.float32),new['actions'])[j]),0.)
        self.assertEqual(old['state'],new['state'])
        self.assertEqual(old['actions']['mean'],new['actions']['mean']);self.assertEqual(old['actions']['std'],new['actions']['std'])
        for field in ('q01','q99'):
            self.assertTrue(all(old['actions'][field][j]==new['actions'][field][j] for j in range(32) if j not in (29,30)))
        a=json.loads((ART/'physical_bounds_A_quantile_plus_sparse_translation.json').read_text());b=json.loads((ART/'physical_bounds_B_train_raw_extrema_candidate.json').read_text())
        self.assertGreater(new['actions']['q99'][30],0);self.assertEqual(a['upper'][30],0)
        self.assertEqual(a['lower'][29],-.0234375);self.assertFalse(b['selected'])

    def test_teacher_next_row_alignment_omits_unlabelled_last_state(self):
        rows=[{'control_tick':i,'physics_steps_since_reset':4*i,'joint_positions':[i/100]*43,
               'applied_action32':None if i==0 else [i/10]*32} for i in range(4)]
        for i in range(1,len(rows)):
            rows[i]['applied_source_action50']=decode_to_source(rows[i]['applied_action32'],rows[i-1]['joint_positions'],list(DATASET_NAMES)).tolist()
        x=extract_samples(rows,list(DATASET_NAMES))
        self.assertEqual(x['state'].shape,(3,32));self.assertTrue(np.allclose(x['actions'][:,0],[.1,.2,.3]))
        self.assertTrue(np.array_equal(x['timestamps'],[0,.02,.04]));self.assertTrue(np.all(x['state'][:,-1]==0))
        rows[-1]['control_tick']=99
        with self.assertRaises(ValueError):extract_samples(rows,list(DATASET_NAMES))

    def test_full_trajectory_duplicate_groups_cross_splits(self):
        rows=[{'episode_id':0,'split':'train','content_sha256':'x','video_sha256':'a'},
              {'episode_id':1,'split':'validation','content_sha256':'x','video_sha256':'b'},
              {'episode_id':2,'split':'train','content_sha256':'y','video_sha256':'b'}]
        groups=duplicate_groups(rows);self.assertEqual(len(groups),1)
        self.assertEqual({r['split'] for r in groups[0]},{'train','validation'})

    def test_unfrozen_teacher_routing_is_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cases=p/'cases.json';cases.write_text(json.dumps(make_source_fallback()))
            (p/'route.json').write_text(json.dumps({'selected_teacher_cases_sha256':sha(cases),'decision_frozen_before_collection':False}))
            with self.assertRaises(ValueError):convert(p,cases,p/'out',p/'not_needed.json',p/'route.json')

    def test_failure_stages_do_not_call_missed_grasp_zero_drops(self):
        r={'status':'completed','success':False,'fell':False,'dropped':False,'grasped':False,'source_success_candidate_tick':42}
        self.assertEqual(failure_stage(r),'missed_grasp')
        self.assertEqual(failure_stage({**r,'grasped':True,'source_success_candidate_tick':None}),'grasp_without_placement')
        self.assertEqual(failure_stage({**r,'grasped':True}),'unstable_or_unreleased_placement')
        self.assertEqual(failure_stage({**r,'fell':True,'dropped':True}),'fell')
        self.assertEqual(failure_stage({**r,'status':'blocked','success':None}),'infrastructure_incomplete')

    def test_three_policy_success_speed_pairs_and_infrastructure_denominator(self):
        cases=make_cases()['final_cases.json']['cases'];h=acceptance_identity()['acceptance_sha256'];gate={'acceptance_version':'pi05_g1_placement_v2','acceptance_sha256':h,'replay_passed':True,'source_interface_passed':True,'frozen_interface_sha256':'f'*64}
        data={}
        for policy,successes in (('groot',0),('old_pi',2),('new_pi',16)):
            rows=[]
            for i,c in enumerate(cases):
                ok=i<successes;end=300 if policy=='old_pi' else 250
                rows.append({'case_id':c['case_id'],'case':c,'policy':policy,'status':'completed','success':ok,'grasped':ok,'stable_placement':ok,
                  'dropped':False,'fell':False,'timeout':not ok,'source_success_candidate_tick':100 if ok else None,
                  'first_grasp_control_tick':20 if ok else None,'qualified_stability_window':{'end_tick':end,'passed':True} if ok else None,
                  'completed_control_ticks':400 if ok else 5,'control_hz':50,'inference_calls':[],
                  'acceptance_version':'pi05_g1_placement_v2','acceptance_sha256':h,'frozen_interface_sha256':'f'*64,
                  'initial_physical_state_sha256':f'{i//4:064x}','source_interface_passed':True,'replay_passed':True})
            data[policy]=rows
        with patch('experiments.pi05_g1_round02.benchmark.check_videos',return_value={'complete':True,'videos':{},'errors':[]}), patch('experiments.pi05_g1_round02.benchmark.measured_wall_timing',side_effect=lambda r: {'complete':True,'errors':[],'actual_task_completion_wall_seconds':r['qualified_stability_window']['end_tick']/25 if r['success'] else None}):
            report=aggregate(cases,data,gate)
            self.assertTrue(report['complete']);self.assertEqual(report['policies']['new_pi']['task_failure_rate'],.2)
            first=next(c for c in report['comparisons'] if c['first_policy']=='groot')
            self.assertEqual(first['matched_success_case_ids'],[]);self.assertIsNone(first['matched_success_sim_delta_seconds'])
            pair=next(c for c in report['comparisons'] if c['first_policy']=='old_pi' and c['second_policy']=='new_pi')
            self.assertEqual(len(pair['matched_success_case_ids']),2);self.assertEqual(pair['matched_success_sim_delta_seconds']['mean'],-1.)
            four=copy.deepcopy(data);four['official_base']=[dict(r,policy='official_base') for r in data['groot']]
            four_report=aggregate(cases,four,gate);self.assertTrue(four_report['complete']);self.assertEqual(four_report['policy_count'],4)
            self.assertTrue(all(r['all_policies_match'] for r in four_report['actual_initial_pair_checks']))
            four['official_base'].pop();self.assertFalse(aggregate(cases,four,gate)['complete'])
            self.assertEqual(pair['matched_success_actual_task_wall_delta_seconds']['mean'],-2.)
            bad=copy.deepcopy(data);bad['new_pi'][0]['success']=None;bad['new_pi'][0]['status']='blocked'
            incomplete=aggregate(cases,bad,gate)
            self.assertFalse(incomplete['complete']);self.assertIsNone(incomplete['policies']['new_pi']['task_failure_rate'])
            self.assertEqual(len(incomplete['infrastructure_incomplete']),1)

    def test_actual_wall_time_uses_success_endpoint_and_measured_rpc(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'result.json').write_text('{}')
            trace=[{'control_tick':i,'monotonic_ns':int(t*1e9)} for i,t in enumerate((10,16,18,25))]
            (p/'trajectory.jsonl').write_text('\n'.join(json.dumps(r) for r in trace))
            row={'result_source_path':str(p/'result.json'),'completed_control_ticks':3,'success':True,
                 'qualified_stability_window':{'end_tick':2},'cold_process_started_monotonic_ns':int(2e9),
                 'cold_ready_monotonic_ns':int(9e9),'inference_calls':[
                    {'observation_control_tick':0,'request_started_monotonic_ns':int(11e9),'reply_received_monotonic_ns':int(15e9),'model_load_seconds':3.},
                    {'observation_control_tick':2,'request_started_monotonic_ns':int(19e9),'reply_received_monotonic_ns':int(24e9)}]}
            report=measured_wall_timing(row)
            self.assertTrue(report['complete']);self.assertEqual(report['actual_task_completion_wall_seconds'],8.)
            self.assertEqual(report['cold_start_to_ready_seconds'],7.);self.assertEqual(report['warm_rpc_wall_seconds'],[5.])
            self.assertEqual(report['task_execution_wall_excluding_RPC_pause_seconds'],4.)
            self.assertEqual(report['execution_wall_excluding_RPC_pause_seconds'],6.)
            failed=measured_wall_timing({**row,'success':False,'source_success_candidate_tick':1})
            self.assertIsNone(failed['actual_task_completion_wall_seconds'])
            row['inference_calls'][0]['reply_received_monotonic_ns']=int(17e9)
            with self.assertRaises(ValueError):measured_wall_timing(row)

    def test_teacher_manifest_lossless_supervised_frames_and_phase_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);case={'case_id':'teacher','seed':200,'apple_offset_xy_m':[0,0]}
            cases={'cases':[case],'train_case_ids':['teacher'],'validation_case_ids':[]}
            cp=p/'cases.json';cp.write_text(json.dumps(cases));rp=p/'route.json'
            rp.write_text(json.dumps({'selected_teacher_cases_sha256':sha(cp),'decision_frozen_before_collection':True}))
            h=acceptance_identity()['acceptance_sha256'];ap=p/'acceptance.json';ap.write_text(json.dumps({'acceptance_sha256':h}))
            folder=p/'runs/teacher';folder.mkdir(parents=True);rows=[]
            video=folder/'head.mp4';out=av.open(str(video),'w');stream=out.add_stream('libx264',rate=50);stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
            for i in range(3):
                rgb=np.full((480,640,3),i,np.uint8);image=folder/f'{i}.png';Image.fromarray(rgb).save(image)
                row={'control_tick':i,'physics_steps_since_reset':i*4,'joint_positions':[0.]*43,
                     'teacher_head_rgb_path':image.name,'head_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
                if i:row.update(applied_action32=[0.]*32,applied_source_action50=decode_to_source([0.]*32,[0.]*43,list(DATASET_NAMES)).tolist())
                rows.append(row)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
            out.close();(folder/'trajectory.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result={'case':case,'status':'completed','success':True,'source_joint_names':list(DATASET_NAMES),
                    'acceptance_version':'pi05_g1_placement_v2','acceptance_sha256':h,'video_path':video.name,
                    'first_grasp_control_tick':0,'source_success_candidate_tick':1,'qualified_stability_window':{'passed':True,'end_tick':2}}
            (folder/'result.json').write_text(json.dumps(result))
            manifest=convert(p/'runs',cp,p/'dataset',ap,rp);episode=manifest['episodes'][0]
            self.assertEqual(episode['source_success_candidate_tick'],1);self.assertEqual(episode['first_grasp_control_tick'],0)
            self.assertEqual(episode['qualified_stability_window'],result['qualified_stability_window'])
            self.assertEqual(len(episode['raw_rgb_frames']),2)
            for i,frame in enumerate(episode['raw_rgb_frames']):
                self.assertEqual(frame['control_tick'],i);self.assertEqual(frame['timestamp_seconds'],i/50.)
                self.assertEqual(frame['byte_sha256'],sha(frame['path']));self.assertEqual(frame['rgb_sha256'],rows[i]['head_rgb_sha256'])
                self.assertEqual(frame['rgb_shape'],[480,640,3])

    def test_external_video_exports_and_actual_launch_receipt_match_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'round02';raw=root/'raw/old_pi/p0_seed40';raw.mkdir(parents=True)
            export=root/'video_exports/old_pi/p0_seed40';configs=root/'configs';configs.mkdir()
            case=make_cases()['final_cases.json']['cases'][0]
            config={'case':case,'policy_id':'old_pi','mode':'policy','output_dir':str(raw),'video_output_dir':str(export)}
            configfile=configs/'p0_seed40.json';configfile.write_text(json.dumps(config))
            (raw/'config.json').write_text(json.dumps(config))
            result=raw/'result.json';result.write_text(json.dumps({'case_id':case['case_id'],'case':case,'policy':'old_pi'}));before=result.read_bytes()
            configfile.with_suffix('.launch.json').write_text(json.dumps({'command':['python','--config',str(configfile)],'elapsed_seconds':12.3}))
            row=load_benchmark_results([raw])[0]
            self.assertEqual(row['video_manifest_path'],str(export/'video_manifest.json'))
            self.assertEqual(row['launch_wall_seconds'],12.3);self.assertEqual(result.read_bytes(),before)

    def test_all_four_video_views_are_independently_probed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);result=p/'result.json';result.write_text('{}');trace=p/'trajectory.jsonl';trace.write_text('trace')
            videos={}
            for key in ('head_raw','observer_raw','head_wall_annotated','observer_wall_annotated'):
                path=p/(key+'.mp4');out=av.open(str(path),'w');stream=out.add_stream('libx264',rate=50);stream.width=16;stream.height=16;stream.pix_fmt='yuv420p'
                for _ in range(2):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((16,16,3),np.uint8),format='rgb24')):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
                out.close();videos[key]={'path':path.name,'fps':50,'frames':2,'wall_annotation':key.endswith('wall_annotated'),'sha256':sha(path)}
            manifest={'schema':'pi05_g1_round02_video_manifest_v1','case_id':'x','policy':'groot','raw_result_sha256':sha(result),'trace_sha256':sha(trace),'videos':videos}
            (p/'video_manifest.json').write_text(json.dumps(manifest));row={'case_id':'x','policy':'groot','completed_control_ticks':1,'result_source_path':str(result)}
            self.assertTrue(check_videos(row)['complete'])
            (p/'observer_raw.mp4').write_bytes(b'corrupt')
            bad=check_videos(row,probe=False);self.assertFalse(bad['complete'])

if __name__=='__main__':unittest.main()
