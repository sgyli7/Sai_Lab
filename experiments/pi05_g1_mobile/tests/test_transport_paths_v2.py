"""All60 are tiny synthetic records with mocked audit/context,not physical proof."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import transport_paths_v2 as t

CONTEXT={'schema':'synthetic_pose_context'}
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def figures(paths,cases,output):
    for name in ('all20_paths.png','all20_paths.pdf','first_case_path_timeline.png','first_case_path_timeline.pdf'):(output/name).write_bytes(b'SYNTHETIC_MOCK_FIGURE')

class PathsTests(unittest.TestCase):
    def fixture(self,root):
        cases=[f'c{i}' for i in range(20)];jobs=[];episodes=[];analysis=[];audits=[];rule={'minimum_carried_displacement_m':4,'minimum_carry_turn_degrees':60};toolsha=t.sha(t.ROOT/'experiments/pi05_g1_mobile/audit_episodes_v3.py')
        for ci,cid in enumerate(cases):
            for policy in t.POLICIES:
                folder=root/'raw'/policy/cid;folder.mkdir(parents=True);config=folder/'config.json';result=folder/'result.json';trace=folder/'trajectory.jsonl';case={'case_id':cid,'seed':1040+ci%4}
                write(config,{'mode':'policy','policy_id':policy,'case':case});rows=[{'control_tick':i,'physics_steps_since_reset':4*i,'root_link_pose_w':[i*.01,ci*.01,0,0,0,0,1],'acceptance_truth_only':{'brown_box':{'pose':[i*.01,ci*.01,.8,0,0,0,1]},'blue_sorting_bin':{'pose':[4.,2.,.4,0,0,0,1]}}} for i in range(3)];trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                flags={'success':ci==0,'grasped':True,'stable_placement':ci==0,'fell':False,'dropped':False,'timeout':ci!=0};acceptance={**flags,'measurement_source':CONTEXT,'placement_rule':rule,'first_grasp_control_tick':0,'carry_qualified_control_tick':1,'qualified_stability_window':{'end_tick':2} if ci==0 else None}
                write(result,{'status':'completed','source_interface_passed':True,'case':case,'acceptance':acceptance,'trace_sha256':t.sha(trace),'completed_control_ticks':2})
                inputs={str(p):t.sha(p) for p in (config,result,trace)};jobs.append({'policy':policy,'case_id':cid,'config_path':str(config),'config_sha256':t.sha(config),'result_path':str(result)})
                episodes.append({'policy':policy,'case_id':cid,'result_path':str(result),'termination_reasons':[] if ci==0 else ['timeout'],'task_completion_sim_seconds_success_only':.04 if ci==0 else None,**flags})
                analysis.append({'policy':policy,'case_id':cid,'inputs_sha256':inputs,'first_grasp_control_tick':0,'carry_qualified_control_tick':1,'diagnostic_prefix_only':False,'infrastructure_completed':True,'full_task_verdict':ci==0,'per_tick_diagnostics':[{'control_tick':i,'ground_drop_this_tick':False,'fell_so_far':False} for i in range(3)]})
                audits.append({'result_path':str(result),'passed':True,'tool_sha256':toolsha,'checks':{'synthetic':{'passed':True}},'inputs_sha256':inputs,'recomputed_acceptance':acceptance})
        plan=root/'plan.json';write(plan,{'schema':'pi05_g1_mobile_benchmark_plan_v1','primary_policies':list(t.POLICIES),'case_ids':cases,'jobs':jobs});summary=root/'summary.json';write(summary,{'schema':'pi05_g1_mobile_benchmark_summary_v1','status':'complete','complete':True,'evidence_qualified':True,'formal_verdict_available':True,'policy_order':list(t.POLICIES),'case_ids':cases,'episodes':episodes,'inputs_sha256':{str(plan):t.sha(plan),**{key:value for episode in analysis for key,value in episode['inputs_sha256'].items()}}})
        audit=root/'audit.json';write(audit,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'tool_sha256':toolsha,'episodes':audits});transport=root/'transport.json';write(transport,{'schema':'pi05_g1_mobile_transport_analysis_v1','source_benchmark':{'path':str(summary),'sha256':t.sha(summary)},'episodes':analysis})
        return plan,summary,[audit],transport
    def test_all60_stream_csv_uses_existing_events_and_only_common_success_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            with patch.object(t,'measurement_identity',return_value=CONTEXT):result=t.run(*args,root/'output',plotter=figures)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(list((root/'output').rglob('*.csv'))),60);self.assertEqual(result['first_frozen_case'],'c0')
            self.assertEqual([x['case_id'] for x in result['common_success_cases_only_fulltask_speed']],['c0']);self.assertFalse(result['new_task_success_judgment'])
            csv=(root/'output/paths/gr00t_n16/c0.csv').read_text();self.assertIn('root_x_m',csv);self.assertEqual(len(csv.splitlines()),4);self.assertIn('destination_reference_x_m',csv);self.assertEqual(result['existing_case_results'][0]['destination_reference_xy_m'],[4.,2.]);self.assertEqual(result['existing_case_results'][0]['box_start_xy_m'],[0.,0.]);self.assertIn('not policy input',result['destination_reference_scope'])
    def test_planned_config_path_can_differ_from_raw_recorded_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);plan=t.read(args[0]);raw=Path(plan['jobs'][0]['config_path']);planned=root/'planned/config.json';write(planned,t.read(raw));plan['jobs'][0]['config_path']=str(planned);plan['jobs'][0]['config_sha256']=t.sha(planned);write(args[0],plan)
            summary=t.read(args[1]);summary['inputs_sha256'][str(args[0])]=t.sha(args[0]);summary['inputs_sha256'][str(planned)]=t.sha(planned);write(args[1],summary)
            transport=t.read(args[3]);transport['source_benchmark']['sha256']=t.sha(args[1]);write(args[3],transport)
            with patch.object(t,'measurement_identity',return_value=CONTEXT):result=t.run(*args,root/'output',plotter=figures)
            self.assertEqual(result['status'],'completed')

    def test_partial_benchmark_missingaudit_and_dev_seed_rejected(self):
        for kind in ('partial','audit','dev'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.fixture(root)
                if kind=='partial':doc=t.read(args[1]);doc['complete']=False;write(args[1],doc)
                elif kind=='audit':doc=t.read(args[2][0]);doc['episodes'].pop();write(args[2][0],doc)
                else:
                    config=Path(t.read(args[0])['jobs'][0]['config_path']);doc=t.read(config);doc['case']['seed']=201;write(config,doc)
                with patch.object(t,'measurement_identity',return_value=CONTEXT):result=t.run(*args,root/'output',plotter=lambda *a:self.fail('must not plot'))
                self.assertEqual(result['status'],'failed')
    def test_raw_trace_tamper_rejected_by_stream_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);trace=Path(t.read(args[0])['jobs'][0]['result_path']).parent/'trajectory.jsonl';trace.write_text(trace.read_text().replace('0.01','0.02'))
            with patch.object(t,'measurement_identity',return_value=CONTEXT):result=t.run(*args,root/'output',plotter=figures)
            self.assertEqual(result['status'],'failed');self.assertIn('SHA',result['error'])
    def test_missing_destination_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            with patch.object(t,'measurement_identity',return_value=CONTEXT):_,records,_,_=t.prepare_inputs(*args)
            trace=records[0]['trace'];rows=[json.loads(line) for line in trace.read_text().splitlines()];del rows[0]['acceptance_truth_only']['blue_sorting_bin'];trace.write_text('\n'.join(json.dumps(row) for row in rows)+'\n');records[0]['expected_trace_sha256']=t.sha(trace)
            with self.assertRaises(KeyError):t.stream_path(records[0],root/'output')

    def test_real_matplotlib_tiny_figures(self):
        try:import matplotlib
        except ImportError:self.skipTest('Plot suite requires separate plot_venv;no install in trainingvenv')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            with patch.object(t,'measurement_identity',return_value=CONTEXT):result=t.run(*args,root/'output')
            self.assertEqual(result['status'],'completed');self.assertGreater((root/'output/all20_paths.pdf').stat().st_size,1000)

if __name__=='__main__':unittest.main()
