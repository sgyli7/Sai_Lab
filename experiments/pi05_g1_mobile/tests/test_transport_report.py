"""Full60 temporary producer seam fixtures, no physical score or policy execution."""
import copy,json,tempfile,unittest
from pathlib import Path
from experiments.pi05_g1_mobile import transport_report as t
from experiments.pi05_g1_mobile.tests import test_delivery_v2 as delivery


def write(p,o):p.write_text(json.dumps(o))


class TransportReportTests(unittest.TestCase):
    def fixture(self,p,qualified=()):
        sp,ap,summary,audits=delivery.DeliveryV2Tests().report_fixture(p)
        trace=p/'trajectory.jsonl';trace.write_text(''.join(json.dumps({'control_tick':i,'monotonic_ns':10**9+i*10**8,'acceptance_truth_only':{'brown_box':{'pose':[0,0,0,0,0,0,1]},'blue_sorting_bin':{'pose':[5,0,0,0,0,0,1]}}})+'\n' for i in range(6)))
        trace_sha=t.sha(trace);records=[];rule={'minimum_source_target_distance_m':5.,'minimum_carried_displacement_m':4.,'minimum_carry_turn_degrees':60.}
        for episode,audit in zip(summary['episodes'],audits):
            policy=episode['policy'];case=episode['case_id'];yes=policy in qualified;T=2 if yes else None
            episode.update(cluster_id=int(case[1:])//4,warm_inference_seconds=[.03,.04],grasped=yes)
            metric={'source_target_distance_m':5.,'max_carried_box_displacement_m':4. if yes else 0.,'contact_qualified_carried_path_m':4. if yes else 0.,'carry_turn_degrees':60. if yes else 0.}
            episode.update({k:metric[k] for k in t.report_v2.METRICS});audit['recomputed_acceptance'].update(metric,placement_rule=rule,carry_qualified_control_tick=T,grasped=yes,first_grasp_control_tick=0 if yes else None)
            audit['inputs_sha256'][str(trace)]=trace_sha
            for name in ['config.json','trajectory.jsonl']:
                summary['inputs_sha256'][str(p/name)]=t.sha(p/name)
            audit['audit_sha256']=t.report_v2.audit_identity(audit)
            records.append({'schema':'pi05_g1_mobile_transport_episode_v1','policy':policy,'case_id':case,'infrastructure_completed':True,'diagnostic_prefix_only':False,'result_status':'completed','full_task_verdict':False,'full_task_timeout':True,'original_result_success':False,'inputs_sha256':dict(audit['inputs_sha256']),'observed_physical_flags':{k:episode[k] for k in ['grasped','stable_placement','fell','dropped']},'existing_carry_rule':rule,'first_grasp_control_tick':0 if yes else None,'carry_qualified_control_tick':T,'first_4m_60deg_qualified_tick':T,'transport_condition_is_not_arrival_or_task_success':True,**metric,'control_ticks':5,'physics_integrations':20,'box_target_actor_origin_xy_error_m':{'final':5.,'minimum':5.},'first_transport_event':None if T is None else {'control_tick':T,'sim_seconds':T/50,'actual_observation_wall_seconds':.2}})
        write(sp,summary);write(ap,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':audits})
        transport=p/'transport.json';doc={'schema':'pi05_g1_mobile_transport_analysis_v1','episodes':records,'source_benchmark':{'path':str(sp),'sha256':t.sha(sp)}};write(transport,doc)
        return sp,ap,transport,doc

    def test_all_zero_complete_report_preserves_na_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);sp,ap,tp,doc=self.fixture(p);before={str(path):t.sha(path) for path in [sp,ap,tp]};out=p/'delivery';r=t.generate(sp,[ap],tp,out)
            self.assertEqual(r['status'],'complete');self.assertEqual(len(r['raw_videos']),120)
            for policy in r['policies'].values():self.assertEqual(policy['carry_qualified'],0);self.assertEqual(policy['full_task_success'],0);self.assertFalse(policy['full_task_16_of20_threshold_met'])
            for pair in r['paired_comparisons'].values():self.assertFalse(pair['transport_speed_available']);self.assertFalse(pair['full_task_speed_available']);self.assertEqual(pair['physical_position_cluster']['exact_cluster_sign_flip_two_sided_p'],1.)
            self.assertIn('运输速度 N/A',(out/'index.html').read_text());self.assertEqual(len((out/'episodes.csv').read_text().splitlines()),61)
            for path,digest in before.items():self.assertEqual(t.sha(path),digest)
            with self.assertRaises(FileExistsError):t.generate(sp,[ap],tp,out)

    def test_one_sided_transport_never_becomes_speed_or_full_success_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);sp,ap,tp,_=self.fixture(p,qualified=['a']);s,rows,v,c=t.load(sp,[ap],tp);r=t.aggregate(s,rows,c)
            self.assertEqual(r['policies']['a']['carry_qualified'],20);self.assertEqual(r['policies']['a']['full_task_success'],0)
            pair=r['paired_comparisons']['a__b'];self.assertEqual(pair['matched_carry_count'],0);self.assertIsNone(pair['matched_carry_wall_difference_seconds_second_minus_first']['median']);self.assertEqual(pair['physical_position_cluster']['exact_cluster_sign_flip_two_sided_p'],.0625)

    def test_matched_transport_is_distinct_from_complete_speed_and_fixed_case_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);sp,ap,tp,_=self.fixture(p,qualified=['a','b','c']);s,rows,v,c=t.load(sp,[ap],tp);r=t.aggregate(s,rows,c)
            pair=r['paired_comparisons']['a__b'];self.assertEqual(pair['matched_carry_count'],20);self.assertFalse(pair['full_task_speed_available']);self.assertEqual(pair['matched_carry_wall_difference_seconds_second_minus_first']['median'],0.)
            self.assertEqual(r['representatives'],[{'kind':'first_frozen_case','case_id':'p0'}])

    def test_missing_case_audit_mismatch_and_metric_tamper_refuse_before_output(self):
        for kind in ['missing','case','sha','audit','metric']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);sp,ap,tp,doc=self.fixture(p)
                if kind=='missing':doc['episodes'].pop()
                elif kind=='case':doc['episodes'][0]['case_id']='UNPLANNED'
                elif kind=='sha':doc['episodes'][0]['inputs_sha256'][str(p/'trajectory.jsonl')]='0'*64
                elif kind=='metric':doc['episodes'][0]['box_target_actor_origin_xy_error_m']['final']=0.
                else:a=t.read(ap);a['episodes'][0]['checks']['actions']['passed']=False;a['episodes'][0]['audit_sha256']=t.report_v2.audit_identity(a['episodes'][0]);write(ap,a)
                write(tp,doc)
                with self.assertRaises(ValueError):t.generate(sp,[ap],tp,p/'rejected')
                self.assertFalse((p/'rejected').exists())

if __name__=='__main__':unittest.main()
