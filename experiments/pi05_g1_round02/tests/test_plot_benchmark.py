import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from experiments.pi05_g1.summarize_evaluation import wilson
from experiments.pi05_g1_round02.plot_benchmark import export,validate,sha


def summary_fixture():
    policies={}
    for policy,n in [('gr00t_n17',0),('pi05_round01',2),('pi05_round02',16)]:
        policies[policy]={'successes':n,'completed':20,'expected':20,'missing_or_infrastructure_cases':[],
                        'success_rate':n/20,'task_failure_rate':(20-n)/20,'success_wilson_95ci':wilson(n,20),
                        'mutually_exclusive_stage_counts':{'success':n,'missed_grasp':20-n},
                        'warm_rpc_wall_seconds':{'count':140,'p50':.25,'p95':.45}}
    comparisons=[]
    for a,b in itertools.combinations(policies,2):
        n=min(policies[a]['successes'],policies[b]['successes']);ids=[f'case{i}' for i in range(n)]
        comparisons.append({'first_policy':a,'second_policy':b,'matched_success_case_ids':ids,
            'matched_success_speed':[{'case_id':c} for c in ids],
            'matched_success_sim_delta_seconds':{'mean':-1.,'count':n} if n else None,
            'matched_success_actual_task_wall_delta_seconds':{'mean':-2.,'count':n} if n else None})
    return {'schema':'pi05_g1_round02_multi_policy_benchmark_v2','complete':True,'evidence_qualified':True,
            'errors':[],'infrastructure_incomplete':[],'policy_count':3,'physical_positions':5,'noise_repetitions_per_position':4,
            'policies':policies,'comparisons':comparisons,
            'actual_initial_pair_checks':[{'case_id':f'case{i}','all_policies_match':True,'policies_expected':3} for i in range(20)],
            'video_evidence':[{'complete':True} for _ in range(60)]}


class PlotBenchmarkTests(unittest.TestCase):
    def test_cpu_export_binds_summary_png_pdf_and_na_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);summary=p/'summary.json';summary.write_text(json.dumps(summary_fixture()));before=summary.read_bytes()
            manifest=export(summary,p/'plots')
            self.assertEqual(manifest['summary_sha256'],sha(summary));self.assertFalse(manifest['raw_case_reads'])
            self.assertEqual(manifest['position_clusters'],5);self.assertEqual(manifest['trials_per_policy'],20)
            with Image.open(p/'plots/benchmark.png') as image:self.assertEqual(image.size,(3000,1440))
            pdf=(p/'plots/benchmark.pdf').read_bytes()
            self.assertTrue(pdf.startswith(b'%PDF'));self.assertNotIn(b'/Subtype /Image',pdf)
            self.assertIn('Matplotlib',manifest['render_backend'])
            self.assertIn('N/A: no common successful cases',(p/'plots/matched_success_speed.csv').read_text())
            for name,entry in manifest['files'].items():self.assertEqual(sha(p/'plots'/name),entry['sha256'])
            self.assertEqual(summary.read_bytes(),before)
            with self.assertRaises(FileExistsError):export(summary,p/'plots')

    def test_incomplete_or_unqualified_reports_never_render(self):
        report=summary_fixture()
        for field,value in [('complete',False),('evidence_qualified',False),('infrastructure_incomplete',[{'status':'blocked'}])]:
            with self.subTest(field=field),self.assertRaises(ValueError):validate({**report,field:value})
        bad=copy.deepcopy(report);bad['policies']['pi05_round02']['completed']=19
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(report);bad['actual_initial_pair_checks'][0]['all_policies_match']=False
        with self.assertRaises(ValueError):validate(bad)

    def test_invalid_stages_latency_or_fake_no_success_speed_rejected(self):
        for kind in ('stages','latency','speed'):
            bad=summary_fixture()
            if kind=='stages':bad['policies']['gr00t_n17']['mutually_exclusive_stage_counts']['missed_grasp']=19
            elif kind=='latency':bad['policies']['gr00t_n17']['warm_rpc_wall_seconds']['p50']=float('nan')
            else:bad['comparisons'][0]['matched_success_sim_delta_seconds']={'mean':-10.,'count':1}
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
