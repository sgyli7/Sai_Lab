"""Synthetic CPU fixtures only; no real model/scene/final data reads."""
import ast
import difflib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import make_benchmark_plan_pose_speed as f
from experiments.pi05_g1_mobile.tests import test_make_benchmark_plan_v2 as oldtests


class PoseFactoryTests(unittest.TestCase):
    def setUp(self):
        pin=patch.object(oldtests,'f',f);pin.start();self.addCleanup(pin.stop)

    fixture=oldtests.FactoryTests.fixture

    def adoption(self,p):
        sources=p/'measurement_sources';sources.mkdir()
        pins={};expected={}
        for name in f.MEASUREMENT_SOURCE_PINS:
            q=sources/name;q.write_text('# synthetic '+name+'\n');expected[name]=f.sha(q);pins[str(q)]=expected[name]
        for name in ['root_pose_speed_validation_decision.json','result.json','placement_oracle_attempt03_pose_speed_independent_cpu_audit.json','placement_oracle_attempt03_pose_speed_qualification.json','review_receipt.json']:
            q=p/name;oldtests.write(q,{'synthetic':True});pins[str(q)]=f.sha(q)
        pins[str(f.SCENARIOS)]=f.SCENARIOS_SHA256
        doc={'schema':'pi05_g1_mobile_root_formal_measurement_adoption_v1','root_reviewed':True,'formal_adoption':True,'formal_trials_before_adoption':0,'no_model_observation_action_physics_controller_camera_scenario_changes':True,'measurement_module':'experiments.pi05_g1_mobile.acceptance_pose_speed','measurement_sha256':expected['acceptance_pose_speed.py'],'new_simulator_entry':'experiments.pi05_g1_mobile.sim_launcher_pose_speed','new_batch_entry':'experiments.pi05_g1_mobile.batch_pose_speed','pins':pins}
        path=p/'adoption.json';oldtests.write(path,doc)
        changes=patch.multiple(f,ADOPTION=path,ADOPTION_SHA256=f.sha(path),MEASUREMENT_SOURCE_ROOT=sources,MEASUREMENT_SOURCE_PINS=expected);changes.start();self.addCleanup(changes.stop)
        return path

    def test_all60_configs_selection_plan_same_proof_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=self.fixture(p);adopt=self.adoption(p);plan=f.make_plan(*args)
            selection=f.read(plan['selection_receipt']);identity=plan['acceptance_measurement']
            self.assertEqual(len(plan['jobs']),60)
            self.assertEqual(identity,selection['acceptance_measurement']);self.assertEqual(plan['batch_entry'],'experiments.pi05_g1_mobile.batch_pose_speed')
            for job in plan['jobs']:
                config=f.read(job['config_path']);self.assertEqual(config['acceptance_measurement'],identity)
                for path,digest in identity['proof_frozen_files'].items():self.assertEqual(config['frozen_files'][path],digest);self.assertEqual(selection['frozen_files'][path],digest);self.assertEqual(plan['frozen_files'][path],digest)
                self.assertEqual(config['ticks'],3000);self.assertEqual(config['replan_interval_ticks'],50);self.assertFalse(config['save_full_rgb'])
            phase=f.verify_artifacts(args[2]/'benchmark_plan.json',p/'phase.json');self.assertEqual(phase['status'],'verified')
            target=Path(next(iter(f.read(adopt)['pins'])));target.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen execution input changed'):f.verify_artifacts(args[2]/'benchmark_plan.json',p/'badphase.json')

    def test_invalid_approval_source_proof_and_adoption_sha_reject_before_models(self):
        for kind in ['approval','prior_formal','source','proof','adoption']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);self.fixture(p);adopt=self.adoption(p);doc=f.read(adopt)
                if kind=='approval':doc['formal_adoption']=False
                if kind=='prior_formal':doc['formal_trials_before_adoption']=1
                if kind=='source':doc['pins'][str(f.MEASUREMENT_SOURCE_ROOT/'acceptance.py')]='0'*64
                if kind=='proof':Path(next(iter(doc['pins']))).write_text('tampered')
                if kind in ['approval','prior_formal','source']:oldtests.write(adopt,doc)
                with patch.object(f,'ADOPTION_SHA256',f.sha(adopt) if kind!='adoption' else '0'*64):
                    with self.assertRaises(ValueError):f.verify_measurement_adoption({})

    def test_old_key_functions_ast_unchanged(self):
        p=Path(f.__file__);old=ast.parse(p.with_name('make_benchmark_plan_v2.py').read_text());new=ast.parse(p.read_text())
        oldfunc={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};newfunc={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for name,node in oldfunc.items():
            if name!='make_plan':self.assertEqual(ast.dump(node),ast.dump(newfunc[name]),name)
        self.assertNotIn('training_carry_run01',p.read_text())

if __name__=='__main__':unittest.main()
