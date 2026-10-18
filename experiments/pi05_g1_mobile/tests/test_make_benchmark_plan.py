"""Synthetic CPU-only factory fixtures; never creates a real formal plan."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import make_benchmark_plan as f
from experiments.pi05_g1_mobile import contract
from experiments.pi05_g1_mobile.batch import load_plan


def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,allow_nan=False))


class FactoryTests(unittest.TestCase):
    def fixture(self,p):
        base=p/'base';base.mkdir();(base/'model.safetensors').write_bytes(b'OFFICIAL_SYNTHETIC');write(base/'config.json',{'fake':True})
        norm=p/'norm.json';write(norm,{'synthetic':True});manifest=p/'manifest.json';write(manifest,{'train':list(range(80)),'validation':list(range(80,90)),'test':list(range(90,100)),'files':[{'episode_id':i,'length':50} for i in range(100)]})
        run=p/'run';run.mkdir();admission=run/'input_manifest.json';write(admission,{'official_base':{'path':str(base/'model.safetensors'),'sha256':f.sha(base/'model.safetensors')},'validation_expected_chunks':10})
        identity={'manifest_sha256':f.sha(manifest),'norm_sha256':f.sha(norm),'prompt':contract.PROMPT,'contract_schema':contract.SCHEMA,'input_manifest_sha256':f.sha(admission)}
        for step,loss in [(0,1.),(400,.5)]:
            d=run/f'step_{step:06d}';d.mkdir();(d/'model.safetensors').write_bytes(b'SYNTHETIC'+str(step).encode());metadata={'schema':'pi05_g1_mobile_training_checkpoint_v1','step':step,'identity':identity,'weights_sha256':f.sha(d/'model.safetensors'),'validation_complete':True,'validation_flow_loss':loss,'validation':{'complete':True,'episodes':10,'chunks':10,'frames':500,'flow_loss':loss}}
            write(d/'metadata.json',metadata);write(d/'metadata_sha256.json',{'metadata_sha256':f.sha(d/'metadata.json')})
        selected=run/'step_000400';write(run/'best_checkpoint.json',{'path':str(selected),'validation_flow_loss':.5,'metadata_sha256':f.sha(selected/'metadata.json')})
        result=run/'training_result.json';write(result,{'schema':'pi05_g1_mobile_training_result_v1','status':'completed','selected_checkpoint':str(selected),'selected_validation_flow_loss':.5,'step':400,'training_stopped':{'step':400,'complete_optimizer_update_boundary':True},'test_loaded':False,'identity':identity})
        source=p/'native_source';source.mkdir();(source/'fake.py').write_text('# CPU fixture\n');harness=p/'harness';harness.mkdir();(harness/'fake.py').write_text('# CPU fixture\n')
        token=p/'token.model';token.write_bytes(b'SYNTHETIC_TOKENIZER');native=p/'native_model';native.mkdir();(native/'weights.safetensors').write_bytes(b'FAKE_NATIVE');write(native/'config.json',{'synthetic':True});nr=p/'native_receipt.json';write(nr,{'profile':'mobile_box','revision':'NATIVE_REV','model_root':str(native),'files':[{'path':q.name,'bytes':q.stat().st_size,'sha256':f.sha(q)} for q in native.iterdir()],'mobile_gr00t_source':{'revision':'SOURCE_REV','files':{'fake.py':f.sha(source/'fake.py')}}})
        geometry=p/'geometry.json';write(geometry,{'fake':True});limits=p/'limits.json';write(limits,{'lower':[-1.]*32,'upper':[1.]*32});overlay_base=p/'base.usda';overlay_base.write_text('#usda1.0\n');overlay=p/'overlay.usda';overlay.write_text('#usda1.0\n@'+str(overlay_base)+'@\n');precision=p/'precision.json';write(precision,{'fake':True});inventory=p/'inventory.json';write(inventory,{'runtime':{'owned_docker_image':'FAKE','image_id':'FAKE','actual_sdk_VERSION':'FAKE','arena_head':'FAKE','lab_head':'FAKE'}})
        cases=[{'case_id':f'c{pos}_s{seed}','cluster_id':pos,'seed':seed,'scene':{'bin_pose':{'position_xyz':[0,-5,0],'rotation_xyzw':[0,0,0,1]},'box_pose':{'position_xyz':[pos*.001,0,0],'rotation_xyzw':[0,0,0,1]}}} for pos in range(5) for seed in [1040,1041,1042,1043]]
        scenarios=p/'scenarios.json';write(scenarios,{'schema':'pi05_g1_mobile_frozen_scenarios_v1','primary_policies':list(f.POLICIES),'episode_seconds':60.,'ticks':3000,'control_hz':50,'physics_hz':200,'native_action_horizon':50,'replan_interval_ticks':50,'cases':cases,'observer_camera':{'synthetic_frozen':True},'acceptance':{'task_family':'SYNTHETIC_TEST_ONLY'},'action_limits_file':str(limits),'frozen_files':{str(overlay):f.sha(overlay)},'task_scope':'Synthetic fixture only, not actual physical evidence','source_reset_randomization':'Synthetic'})
        template=p/'template.json';write(template,{'mode':'policy','assets':{'background':str(overlay)},'mesh_url_paths':{},'source_harness_dir':str(harness),'geometry_definition':str(geometry),'robot_geometry':str(geometry),'case':{'case_id':'DEV_MUST_NOT_BE_USED','seed':201},'output_dir':'DEV_MUST_NOT_BE_USED','policy_command':['native_python','-m','experiments.pi05_g1_mobile.gr00t_service','--seed','201','--decoded-evidence','DEV_PATH'],'HFmodel':{'revision':'NATIVE_REV'},'frozen_files':{str(overlay_base):f.sha(overlay_base)},'save_full_rgb':True,'offline_unused_assets':True})
        processor=source/'gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2';processor.mkdir(parents=True)
        for name in ['vocab.json','merges.txt','tokenizer_config.json','preprocessor_config.json','processor_config.json','chat_template.json']:(processor/name).write_text('{}')
        helper=p/'offline_assets.py';helper.write_text('# SYNTHETIC strict helper identity\n')
        library=p/'object_library.py';library.write_text('class BrownBox: pass\n'+''.join('class '+name+': pass\n' for name in f.offline_assets.EXCLUDED_CLASSES))
        doc=f.read(template);doc['frozen_files'].update({str(helper):f.sha(helper),str(library):f.sha(library)});write(template,doc)
        pin=patch.object(f.offline_assets,'SOURCE_SHA256',f.sha(library));pin.start();self.addCleanup(pin.stop)
        changes={'OFFLINE_HELPER':helper,'OBJECT_LIBRARY':library,'SCENARIOS':scenarios,'SCENARIOS_SHA256':f.sha(scenarios),'TEMPLATE':template,'NORM':norm,'MANIFEST':manifest,'OFFICIAL_BASE':base,'OFFICIAL_BASE_SHA256':f.sha(base/'model.safetensors'),'TOKENIZER':token,'TOKENIZER_SHA256':f.sha(token),'NATIVE_RECEIPT':nr,'NATIVE_SOURCE':source,'RUNTIME_INVENTORY':inventory,'PRECISION_RULE':precision,'SOURCE_ROOTS':(source,),'PI_PYTHON':p/'fake_pi_python'}
        overrides=patch.multiple(f,**changes);overrides.start();self.addCleanup(overrides.stop)
        for function in ['git_identity','docker_identity']:
            mock=patch.object(f,function,return_value={'synthetic_cpu_identity':True});mock.start();self.addCleanup(mock.stop)
        return selected,result,p/'plan',p/'episodes'

    def test_exact60_rotation_common_pi_runtime_norm_and_dependency_phase_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=self.fixture(p);plan=f.make_plan(*args)
            self.assertEqual(len(plan['jobs']),60);self.assertFalse(args[3].exists())
            self.assertEqual([j['policy'] for j in plan['jobs'][:9]],['gr00t_n16','pi05_mobile','official_pi05_base','pi05_mobile','official_pi05_base','gr00t_n16','official_pi05_base','gr00t_n16','pi05_mobile'])
            self.assertIn(str(p/'base.usda'),plan['frozen_files']);self.assertIn(str(p/'token.model'),plan['frozen_files']);self.assertIn(str(p/'native_source/fake.py'),plan['frozen_files'])
            selection=f.read(plan['selection_receipt']);offline=selection['dependencies']['offline_unused_assets']
            self.assertEqual(offline['filter_receipt']['excluded_classes'],list(f.offline_assets.EXCLUDED_CLASSES))
            self.assertIn('BrownBox',offline['filter_receipt']['retained_class_ast_sha256'])
            self.assertIn(str(p/'offline_assets.py'),plan['frozen_files']);self.assertIn(str(p/'object_library.py'),plan['frozen_files'])
            commands=[]
            for job in plan['jobs']:
                config=f.read(job['config_path']);self.assertIs(config['offline_unused_assets'],True);self.assertIs(config['save_full_rgb'],False);self.assertEqual(config['max_run_wall_seconds'],840);self.assertEqual(job['estimated_wall_seconds'],900);self.assertNotIn('DEV_MUST_NOT_BE_USED',json.dumps(config));self.assertEqual(config['ticks'],3000);self.assertEqual(config['observer_camera'],{'synthetic_frozen':True})
                if job['policy']!='gr00t_n16':
                    cmd=list(config['policy_command']);self.assertNotIn('--head-only',cmd);cmd[cmd.index('--checkpoint')+1]='WEIGHTS';cmd[cmd.index('--decoded-evidence')+1]='EVIDENCE';commands.append(cmd);self.assertEqual(config['norm_stats'],str(p/'norm.json'))
            self.assertTrue(all(c==commands[0] for c in commands))
            self.assertIn(str(p/'native_source/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/vocab.json'),plan['frozen_files'])
            self.assertIn(str(p/'native_source/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/merges.txt'),plan['frozen_files'])
            load_plan(args[2]/'benchmark_plan.json')
            receipt=f.verify_artifacts(args[2]/'benchmark_plan.json',p/'phase.json');self.assertEqual(receipt['status'],'verified')
            (args[0]/'model.safetensors').write_bytes(b'TAMPERED')
            with self.assertRaisesRegex(ValueError,'artifact SHA'):f.verify_artifacts(args[2]/'benchmark_plan.json',p/'tampered_phase.json')

    def test_supplied_nonbest_incomplete_training_and_metadata_tamper_reject(self):
        for kind in ['nonbest','incomplete','metadata','weights']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);args=list(self.fixture(p))
                if kind=='nonbest':args[0]=args[0].parent/'step_000000'
                elif kind=='incomplete':o=f.read(args[1]);o['status']='failed';write(args[1],o)
                elif kind=='metadata':mp=args[0]/'metadata.json';o=f.read(mp);o['validation_flow_loss']=.001;write(mp,o)
                else:(args[0]/'model.safetensors').write_bytes(b'CHANGED')
                with self.assertRaises(ValueError):f.make_plan(*args)
                self.assertFalse(args[2].exists())

    def test_offline_flag_and_both_pinned_sources_are_required(self):
        for kind in ['disabled','missing_pin','helper_tamper','source_tamper','ast_tamper']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);args=self.fixture(p);doc=f.read(f.TEMPLATE)
                if kind=='disabled':doc['offline_unused_assets']=False
                elif kind=='missing_pin':del doc['frozen_files'][str(f.OFFLINE_HELPER)]
                elif kind=='helper_tamper':f.OFFLINE_HELPER.write_text('# changed')
                elif kind=='source_tamper':f.OBJECT_LIBRARY.write_text('class Changed: pass')
                else:
                    # A newly recorded SHA still cannot bypass the helper's pinned
                    # official-source identity and six-class retained-AST check.
                    f.OBJECT_LIBRARY.write_text('class BrownBox: pass')
                    doc['frozen_files'][str(f.OBJECT_LIBRARY)]=f.sha(f.OBJECT_LIBRARY)
                write(f.TEMPLATE,doc)
                with self.assertRaises(ValueError):f.make_plan(*args)
                self.assertFalse(args[2].exists())

    def test_scenario_pin_and_existing_roots_refuse_without_reading_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=self.fixture(p);sc=f.SCENARIOS;sc.write_text(sc.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'scenarios SHA'):f.make_plan(*args)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=self.fixture(p);args[3].mkdir();(args[3]/'result.json').write_text('NEVER_PARSE_OUTCOME')
            with self.assertRaises(FileExistsError):f.make_plan(*args)

if __name__=='__main__':unittest.main()
