"""Synthetic files only: plan construction never imports or runs GPU libraries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pi05_g1_round02.make_benchmark_plan import make_plan,select_checkpoint,ROOT,write
from experiments.pi05_g1_round02.prepare_inputs import sha,make_cases
from experiments.pi05_g1_round02.clone_plan import clone
from experiments.pi05_g1_round02.verify_plan_phase import verify


class BenchmarkPlanTests(unittest.TestCase):
    def fixture(self,p):
        old=p/'old';base=p/'base';run=p/'train';old.mkdir();base.mkdir();run.mkdir()
        (old/'model.safetensors').write_bytes(b'old');(base/'model.safetensors').write_bytes(b'official')
        write(old/'metadata.json',{'step':1200});norm=p/'norm.json';write(norm,{'actions':{}})
        loss=p/'loss.json';write(loss,{'actions':[1.]*32})
        declarations=[];sources=[]
        for name,kind,n in [('original','lerobot',21),('fresh_teacher','npz_video',3)]:
            manifest=p/(name+'.json');write(manifest,{'synthetic_source':name})
            declaration={'name':name,'kind':kind,'manifest':str(manifest),'train_sampling_weight':.5,'validation_weight':.5}
            declarations.append(declaration)
            sources.append({**declaration,'manifest_sha256':sha(manifest),'train_episodes':9,'validation_episodes':n,
                            'validation_chunks':n*2,'time_sampling':{'kind':'uniform'}})
        data=p/'data.json';write(data,{'sources':declarations})
        identity={'data_config_sha256':sha(data),'norm_sha256':sha(norm),'loss_weights_sha256':sha(loss),'sources':sources}
        for step,value in [(0,1.),(200,2.)]:
            folder=run/f'step_{step:06d}';folder.mkdir();(folder/'model.safetensors').write_bytes(b'old' if step==0 else b'updated')
            metrics={s['name']:{'flow_loss':value,'frames':50,'episodes':s['validation_episodes'],'chunks':s['validation_chunks'],'validation_weight':.5} for s in sources}
            write(folder/'metadata.json',{'schema':'pi05_g1_round02_training_checkpoint_v1','step':step,'round02_data_identity':identity,
                'loss_weights':[1.]*32,'validation_complete':True,'validation_flow_loss':value,'validation_sources':metrics,
                'warmstart':{'checkpoint':str(old),'source_optimizer_steps':1200,'weights_sha256':sha(old/'model.safetensors')}})
        write(run/'best_checkpoint.json',{'path':'step_000000','validation_flow_loss':1.})
        cases=p/'cases.json';write(cases,make_cases()['final_cases.json']);limits=p/'limits.json';write(limits,{'lower':[0.]*32,'upper':[1.]*32})
        rule=p/'comparator_adapter_rule.json';write(rule,{'pi05_common_norm_sha256':sha(norm),'normalization_override_disclosed':'synthetic'})
        gates={}
        for name in ('bounded','unbounded'):
            path=p/name/'result.json';write(path,{'status':'completed','success':True,'replay_passed':True,'source_interface_passed':True,
                'frozen_interface_sha256':'iface','acceptance_sha256':'accept','acceptance_version':'pi05_g1_placement_v2','initial_physical_state_schema':'pi05_g1_initial_kinematic_state_v2'})
            gates[name]=path
        write(p/'bounded/config.json',{'mode':'replay','observer_camera':{'test':'observer'},'assets':{},'expert_sequence':'unused','expert_sha256':'unused','action_limits_file':str(limits)})
        groot=p/'groot.bin';groot.write_bytes(b'groot');discovery=p/'gr00t_discovery.json'
        write(discovery,{'files':[{'path':str(groot),'sha256':sha(groot),'bytes':5}]})
        protocol={'schema':'pi05_g1_round02_frozen_final_protocol_v1','primary_models':['gr00t_n17','pi05_round01','pi05_round02'],
            'replan_interval_ticks':40,'native_action_horizons':{'pi05':50,'gr00t_n17':40},'pi_common_norm_file':str(norm),'pi_common_norm_sha256':sha(norm),
            'final_cases_file':str(cases),'final_cases_sha256':sha(cases),'data_config_sha256':sha(data),'limits_file':str(limits),'limits_file_sha256':sha(limits),
            'pi_compare_rule_sha256':sha(rule),'common_interface_sha256':'iface','acceptance_sha256':'accept','acceptance_version':'pi05_g1_placement_v2',
            'initial_state_schema':'pi05_g1_initial_kinematic_state_v2','camera_observer':{'test':'observer'},
            'simulator_sha256':sha(ROOT/'experiments/pi05_g1_round02/simulator.py')}
        for key,path in gates.items():protocol.update({key+'_replay_result':str(path),key+'_replay_sha256':sha(path)})
        protocol_path=p/'final_protocol.json';write(protocol_path,protocol)
        return {'training_dir':run,'protocol_path':protocol_path,'data_config':data,'loss_weights':loss,'output':p/'eval',
                'round01_checkpoint':old,'official_base':base,'gr00t_manifest':discovery},norm

    def recovery_fixture(self,p,old_loss=.7,new0_loss=.6,new200_loss=.9):
        from experiments.pi05_g1_round02.checkpoint_lineage import resolve_lineage
        args,norm=self.fixture(p);run01=args['training_dir'];parent=run01/'step_000200'
        meta=json.loads((parent/'metadata.json').read_text());meta['validation_flow_loss']=old_loss
        for metric in meta['validation_sources'].values():metric['flow_loss']=old_loss
        (parent/'metadata.json').write_text(json.dumps(meta))
        (run01/'best_checkpoint.json').write_text(json.dumps({'path':'step_000200','validation_flow_loss':old_loss}))
        (run01/'progress.jsonl').write_text(json.dumps({'event':'optimizer_step','step':343})+'\n')
        failure=p/'failure.json';write(failure,{'source_training_dir':str(run01),'source_step':200,'status':'host_memory_guard','exit_code':137,
            'source_metadata_sha256':sha(parent/'metadata.json'),'source_weights_sha256':sha(parent/'model.safetensors')})
        lineage=resolve_lineage(parent,verify_weights=False);run02=p/'recovery';run02.mkdir()
        for step,loss in [(0,new0_loss),(200,new200_loss)]:
            folder=run02/f'step_{step:06d}';folder.mkdir();(folder/'model.safetensors').write_bytes(b'updated' if step==0 else b'newer')
            child=copy.deepcopy(meta);child.update(step=step,local_step=step,inherited_round_updates=200,cumulative_round_updates=200+step,
                warmstart={'checkpoint':str(parent),'weights_sha256':sha(parent/'model.safetensors'),'optimizer_reset':True,'source_optimizer_payload_loaded':False},
                checkpoint_lineage={'schema':'pi05_g1_round02_checkpoint_lineage_v1','immediate_parent':{'path':str(parent),
                    'metadata_sha256':sha(parent/'metadata.json'),'weights_sha256':sha(parent/'model.safetensors')},
                    'genesis':lineage['genesis'],'inherited_round_updates':200,'local_step':step,'cumulative_round_updates':200+step,
                    'optimizer_reset':True,'source_failure_receipt':{'path':str(failure),'sha256':sha(failure)}},validation_flow_loss=loss)
            for metric in child['validation_sources'].values():metric['flow_loss']=loss
            write(folder/'metadata.json',child)
        write(run02/'candidate_training_dirs.json',{'schema':'pi05_g1_round02_candidate_training_dirs_v1',
             'training_dirs':[str(run01),str(run02)],'current_training_dir':str(run02)})
        path,value=min([(parent,old_loss),(run02/'step_000000',new0_loss),(run02/'step_000200',new200_loss)],key=lambda item:item[1])
        write(run02/'best_checkpoint.json',{'path':str(path),'validation_flow_loss':value})
        args['training_dir']=run02
        return args,norm,run01,parent

    def test_recovered_local_zero_has200_inherited_and_failed_run_candidates_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm,failed,parent=self.recovery_fixture(p)
            choice,candidates,identity,metadata,pointer=select_checkpoint(args['training_dir'],args['data_config'],norm,args['loss_weights'])
            self.assertEqual(choice['local_step'],0);self.assertEqual(choice['inherited_round_updates'],200)
            self.assertEqual(choice['cumulative_round_updates'],200);self.assertEqual(len(candidates),4)
            self.assertIn(str(parent),[c['path'] for c in candidates]);self.assertFalse((failed/'training_stopped.json').exists())
            plans=make_plan(**args);receipt=json.loads((p/'eval/selection_receipt.json').read_text())
            self.assertEqual(receipt['selected_round_optimizer_updates'],200);self.assertEqual(receipt['selected_local_optimizer_updates'],0)
            self.assertEqual(receipt['model_artifacts']['pi05_round01']['path'],str(args['round01_checkpoint']/'model.safetensors'))
            self.assertIn(str(p/'failure.json'),plans['primary']['frozen_files'])
            self.assertIn(str(parent/'model.safetensors'),[r['path'] for r in plans['primary']['ancestor_model_artifacts']])
            verify(p/'eval/primary_plan.json','start',p/'recovery_phase_start.json')
            (parent/'model.safetensors').write_bytes(b'changed only intermediate ancestor')
            with self.assertRaisesRegex(ValueError,'Phase verification'):verify(p/'eval/primary_plan.json','end',p/'recovery_phase_end.json')

    def test_global_min_can_be_failed_run200_or_lower_new_candidate(self):
        for new0,new200,expected_owner,expected_step,total in [(1.,1.1,'failed',200,200),(.8,.5,'new',200,400)]:
            with self.subTest(new200=new200),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);args,norm,failed,parent=self.recovery_fixture(p,new0_loss=new0,new200_loss=new200)
                choice,*_=select_checkpoint(args['training_dir'],args['data_config'],norm,args['loss_weights'])
                self.assertEqual(Path(choice['path']).parent,failed if expected_owner=='failed' else args['training_dir'])
                self.assertEqual(choice['local_step'],expected_step);self.assertEqual(choice['cumulative_round_updates'],total)

    def test_recovery_ancestor_tamper_and_pointer_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm,failed,parent=self.recovery_fixture(p)
            pointer=args['training_dir']/'best_checkpoint.json';original=pointer.read_bytes()
            pointer.write_text(json.dumps({'path':str(p/'outside/step_000000'),'validation_flow_loss':.6}))
            with self.assertRaisesRegex(ValueError,'roster'):select_checkpoint(args['training_dir'],args['data_config'],norm,args['loss_weights'])
            pointer.write_bytes(original);parent_meta=parent/'metadata.json';saved=parent_meta.read_bytes()
            parent_meta.write_text(parent_meta.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'metadata'):select_checkpoint(args['training_dir'],args['data_config'],norm,args['loss_weights'])
            parent_meta.write_bytes(saved);(parent/'model.safetensors').write_bytes(b'tampered ancestor')
            with self.assertRaisesRegex(ValueError,'ancestor weights'):make_plan(**args)

    def test_factory_step0_rotation_no_gpu_and_clone_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm=self.fixture(p);plans=make_plan(**args)
            self.assertEqual([len(plans[k]['jobs']) for k in ('primary','optional_base')],[60,20])
            jobs=plans['primary']['jobs'];policies=['gr00t_n17','pi05_round01','pi05_round02']
            for i in range(20):
                actual=[json.loads(Path(j['config_path']).read_text()) for j in jobs[i*3:i*3+3]]
                self.assertEqual([r['policy_id'] for r in actual],policies[i%3:]+policies[:i%3])
                for r in actual:
                    self.assertEqual(r['replan_interval_ticks'],40)
                    self.assertNotIn('expert_sequence',r)
                    if r['policy_id']!='gr00t_n17':self.assertEqual(r['norm_stats'],str(norm))
            receipt=json.loads((p/'eval/selection_receipt.json').read_text())
            self.assertEqual(receipt['selected_round_optimizer_updates'],0)
            self.assertEqual(receipt['selected']['step'],0)
            verify(p/'eval/primary_plan.json','start',p/'start.json')
            clone(p/'eval',p/'clone')
            cloned=json.loads((p/'clone/primary_plan.json').read_text());new=json.loads(Path(cloned['jobs'][0]['config_path']).read_text());old=json.loads(Path(jobs[0]['config_path']).read_text())
            self.assertEqual(new['replay_result'],old['replay_result']);self.assertEqual(new['policy_command'],old['policy_command'])
            self.assertTrue(new['output_dir'].startswith(str(p/'clone')))
            self.assertEqual((p/'clone/selection_receipt.json').read_bytes(),(p/'eval/selection_receipt.json').read_bytes())
            with self.assertRaises(FileExistsError):make_plan(**args)
            with self.assertRaises(FileExistsError):clone(p/'eval',p/'clone')
            (args['round01_checkpoint']/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(p/'eval/primary_plan.json','end',p/'end.json')
            self.assertFalse(json.loads((p/'end.json').read_text())['passed'])

    def test_best_pointer_must_be_actual_global_complete_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm=self.fixture(p);run=args['training_dir']
            pointer=run/'best_checkpoint.json';pointer.write_text(json.dumps({'path':'step_000200','validation_flow_loss':2.}))
            with self.assertRaisesRegex(ValueError,'minimum'):select_checkpoint(run,args['data_config'],norm,args['loss_weights'])
            pointer.write_text(json.dumps({'path':'step_000000','validation_flow_loss':1.}))
            meta=run/'step_000200/metadata.json';m=json.loads(meta.read_text());m['validation_sources']['original']['chunks']=1;meta.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'cover'):select_checkpoint(run,args['data_config'],norm,args['loss_weights'])

    def test_identity_mismatch_and_partial_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm=self.fixture(p);meta=args['training_dir']/'step_000000/metadata.json';m=json.loads(meta.read_text())
            m['round02_data_identity']['norm_sha256']='wrong';meta.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'identity'):make_plan(**args)
            self.assertFalse(args['output'].exists())
            args['output'].mkdir()
            with self.assertRaises(FileExistsError):make_plan(**args)

    def test_incomplete_validation_not_selected_and_outcomes_never_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args,norm=self.fixture(p);run=args['training_dir']
            meta=run/'step_000200/metadata.json';m=json.loads(meta.read_text())
            m.update(validation_complete=False,validation_flow_loss=None,validation_sources=None)
            meta.write_text(json.dumps(m))
            # A poisonous test/result artifact must have no effect on validation-only selection.
            (run/'offline_test_results.json').write_text('not valid JSON and forbidden for selection')
            choice,candidates,*_=select_checkpoint(run,args['data_config'],norm,args['loss_weights'])
            self.assertEqual(choice['step'],0);self.assertEqual(len(candidates),1)


if __name__=='__main__':unittest.main()
