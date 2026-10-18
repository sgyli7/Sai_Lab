"""Tiny synthetic checkpoints; no final results/test observations/GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pi05_g1_round02.tests import test_benchmark_plan as base_fixture
from experiments.pi05_g1_round02.make_benchmark_plan import make_plan as main_factory,sha,write
from experiments.pi05_g1_round02.make_supplementary_plan import make_plan,POLICY
from experiments.pi05_g1_round02.verify_plan_phase import verify
from experiments.pi05_g1_round02.checkpoint_lineage import resolve_lineage


def fixture(root,duplicate=False,other_parent=False,fresh_loss=.4):
    args,norm,failed,parent=base_fixture.BenchmarkPlanTests().recovery_fixture(root)
    new=args['training_dir']/'step_000200';path=new/'metadata.json';meta=json.loads(path.read_text())
    meta['validation_sources']['fresh_teacher']['flow_loss']=fresh_loss
    meta['validation_sources']['original']['flow_loss']=1.8-fresh_loss  # Mixed .9 loses to primary .6, fresh .4 wins.
    if other_parent:
        branch=failed/'step_000100';branch.mkdir();(branch/'model.safetensors').write_bytes(b'other branch')
        branchmeta=json.loads((parent/'metadata.json').read_text());branchmeta['step']=100;branchmeta['validation_flow_loss']=2.
        for value in branchmeta['validation_sources'].values():value['flow_loss']=2.
        write(branch/'metadata.json',branchmeta)
        failure=root/'other_parent_failure.json';write(failure,{'source_training_dir':str(failed),'source_step':100,'status':'host_memory_guard','exit_code':137,
            'source_metadata_sha256':sha(branch/'metadata.json'),'source_weights_sha256':sha(branch/'model.safetensors')})
        lineage=resolve_lineage(branch,False)
        meta.update(local_step=200,inherited_round_updates=100,cumulative_round_updates=300)
        meta['warmstart'].update(checkpoint=str(branch),weights_sha256=sha(branch/'model.safetensors'))
        meta['checkpoint_lineage'].update(immediate_parent={'path':str(branch),'metadata_sha256':sha(branch/'metadata.json'),'weights_sha256':sha(branch/'model.safetensors')},
            genesis=lineage['genesis'],inherited_round_updates=100,cumulative_round_updates=300,source_failure_receipt={'path':str(failure),'sha256':sha(failure)})
    path.write_text(json.dumps(meta))
    if duplicate:(new/'model.safetensors').write_bytes((args['training_dir']/'step_000000/model.safetensors').read_bytes())
    main_factory(**args)
    declaration=root/'predeclaration.json';write(declaration,{'schema':'pi05_g1_round02_supplementary_fresh_validation_predeclaration_v1',
        'final_outcomes_observed':False,'primary_models_and_mixed_selection_unchanged':True,'supplementary_policy_id':POLICY,
        'selection_rule':'Minimum complete fresh validation; tie by explicit run order then checkpoint path'})
    return args,declaration,new,parent


class SupplementaryPlanTests(unittest.TestCase):
    def test_fresh_improvement_selects_mixed_worse_candidate_without_changing_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,decl,fresh,parent=fixture(Path(tmp));root=args['output'];original=(root/'selection_receipt.json').read_bytes()
            receipt,plan=make_plan(root,args['training_dir'],decl)
            self.assertEqual(receipt['selected']['path'],str(fresh));self.assertEqual(receipt['selected_scores']['fresh_validation_flow_loss'],.4)
            self.assertEqual(receipt['selected_scores']['combined_validation_flow_loss'],.9)
            self.assertEqual(plan['policies'],[POLICY]);self.assertEqual(len(plan['jobs']),20)
            self.assertEqual((root/'selection_receipt.json').read_bytes(),original)
            for job in plan['jobs']:
                config=json.loads(Path(job['config_path']).read_text());cid=config['case']['case_id']
                base=json.loads((root/'configs'/f'pi05_round02_{cid}.json').read_text())
                for key in set(base)-{'policy_id','output_dir','video_output_dir','policy_command','policy_artifacts'}:self.assertEqual(config[key],base[key])
                self.assertEqual(config['policy_id'],POLICY)
            verify(root/'supplementary_fresh_plan.json','start',root/'supplementary_start.json')
            (fresh/'model.safetensors').write_bytes(b'tampered selected supplement')
            with self.assertRaises(ValueError):verify(root/'supplementary_fresh_plan.json','end',root/'supplementary_end.json')

    def test_same_payload_records_duplicate_without_plan_or_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,decl,fresh,parent=fixture(Path(tmp),duplicate=True);root=args['output']
            receipt,plan=make_plan(root,args['training_dir'],decl)
            self.assertEqual(receipt['status'],'duplicate_primary_payload');self.assertIsNone(plan)
            self.assertFalse((root/'supplementary_fresh_plan.json').exists());self.assertEqual(list((root/'configs').glob(POLICY+'_*.json')),[])
            self.assertFalse(receipt['plan_created'])
            with self.assertRaises(FileExistsError):make_plan(root,args['training_dir'],decl)

    def test_equal_fresh_validation_uses_fixed_checkpoint_path_tie_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,decl,fresh,parent=fixture(Path(tmp),fresh_loss=.6)
            receipt,plan=make_plan(args['output'],args['training_dir'],decl)
            self.assertEqual(receipt['selected']['path'],str(args['training_dir']/'step_000000'))
            self.assertEqual(receipt['tie_rule'],'explicit run order, then checkpoint path')
            self.assertIsNone(plan)  # Deterministic tie selects the existing primary payload.

    def test_wrong_identity_parent_branch_and_ancestor_tamper_rejected(self):
        for failure in ('identity','ancestry','payload'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                args,decl,fresh,parent=fixture(Path(tmp),other_parent=failure=='ancestry')
                if failure=='identity':
                    path=fresh/'metadata.json';m=json.loads(path.read_text());m['round02_data_identity']['norm_sha256']='wrong';path.write_text(json.dumps(m))
                elif failure=='payload':(parent/'model.safetensors').write_bytes(b'tampered ancestor')
                with self.assertRaises(ValueError):make_plan(args['output'],args['training_dir'],decl)
                self.assertFalse((args['output']/'supplementary_fresh_plan.json').exists())

    def test_final_outcome_existence_blocks_late_selection_without_reading_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,decl,fresh,parent=fixture(Path(tmp));root=args['output'];out=root/'pi05_round02/p0_seed40';out.mkdir(parents=True)
            (out/'result.json').write_text('DO NOT READ: deliberately not JSON')
            with self.assertRaisesRegex(ValueError,'before final outcomes'):make_plan(root,args['training_dir'],decl)


if __name__=='__main__':unittest.main()
