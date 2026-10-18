"""Synthetic TRAIN-only prefix/scheduling tests, not physical validation."""
import ast,hashlib,json,random,tempfile,types,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from experiments.pi05_g1_mobile import carry_prefix_data as c
from experiments.pi05_g1_mobile import training_carry as t
from experiments.pi05_g1_mobile.contract import DATASET_NAMES,decode_to_source

class CarryTests(unittest.TestCase):
    def fixture(self,root):
        case=root/'case';case.mkdir();(case/'head_rgb').mkdir();rows=[]
        for i in range(5):
            q=np.zeros(43,np.float32);q[11]=i*.01;action=np.full(32,i*.001,np.float32);rgb=np.full((480,640,3),i,np.uint8);Image.fromarray(rgb).save(case/'head_rgb'/f'{i:05d}.png')
            rows.append({'control_tick':i,'physics_steps_since_reset':4*i,'source_joint_names':list(DATASET_NAMES),'joint_positions':q.tolist(),'applied_action32':action.tolist(),'head_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
        for i in range(1,5):rows[i]['applied_source_action50']=decode_to_source(rows[i]['applied_action32'],rows[i-1]['joint_positions'],DATASET_NAMES).tolist()
        (case/'trajectory.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        runtime={'source_joint_names':list(DATASET_NAMES),'wbc_config':{},'frozen_interface_sha256':'interface'};(case/'runtime_identity.json').write_text(json.dumps(runtime))
        result={**runtime,'case':{'case_id':'native_dev04','seed':201},'status':'completed','success':False,'policy':'gr00t_n16','acceptance':{'carry_qualified_control_tick':2,'grasped':True,'fell':False,'dropped':False},'trace_sha256':c.sha(case/'trajectory.jsonl'),'runtime_identity_sha256':c.sha(case/'runtime_identity.json'),'completed_control_ticks':4}
        (case/'result.json').write_text(json.dumps(result));(case/'config.json').write_text(json.dumps({'mode':'policy','policy_id':'gr00t_n16','case':result['case'],'prompt':c.PROMPT}))
        source=root/'source.json';source.write_text(json.dumps({'train':list(range(80)),'validation':list(range(80,90)),'test':list(range(90,100))}));norm=root/'norm.json';norm.write_text('{}');diagnostic=root/'diagnostic.json';diagnostic.write_text('{"synthetic":true}')
        audit=root/'audit.json';audit.write_text(json.dumps({'schema':c.AUDIT_SCHEMA,'status':'passed','prefix_no_fall_drop':True,'native_policy_only':True,'case_id':'native_dev04','seed':201,'carry_qualified_control_tick':2,'scope_partial_carry_not_task_success':True,'source_global_audit_passed':False,'video_origin_diagnostic':{'path':str(diagnostic),'sha256':c.sha(diagnostic)},'bind_sha256':{k:c.sha(case/f) for k,f in {'result':'result.json','trace':'trajectory.jsonl','config':'config.json','runtime_identity':'runtime_identity.json'}.items()}}))
        a=json.loads(audit.read_text())
        v2=root/'v2audit.json'
        inputs={str((case/f).resolve()):c.sha(case/f) for f in ['result.json','trajectory.jsonl','config.json','runtime_identity.json']}
        child={'schema':'pi05_g1_mobile_episode_cpu_audit_v2','result_path':str((case/'result.json').resolve()),'passed':True,'autonomous_policy_trial':True,'checks':{'actual_runtime_identity':{'passed':True},'videos':{'passed':True}},'inputs_sha256':inputs,'recomputed_acceptance':{'carry_qualified_control_tick':2,'success':False,'fell':False,'dropped':False}}
        v2.write_text(json.dumps({'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':[child]}))
        a.update(legacy_v1_audit_passed=False,independent_v2_audit_passed=True,independent_v2_audit={'path':str(v2),'sha256':c.sha(v2)},raw_rgb_frames=[{'path':str((case/'head_rgb'/f'{i:05d}.png').resolve()),'byte_sha256':c.sha(case/'head_rgb'/f'{i:05d}.png'),'rgb_sha256':rows[i]['head_rgb_sha256'],'control_tick':i,'timestamp_seconds':i/50} for i in range(2)])
        audit.write_text(json.dumps(a))
        return case,audit,source,norm
    def test_strict_prefix_no_post_boundary_or_fake_future_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,audit,source,norm=self.fixture(root);m=c.prepare(case,audit,source,norm,root/'prefix')
            self.assertEqual(m['length'],2);self.assertEqual(m['validation'],[]);self.assertFalse(m['task_success_teacher'])
            with np.load(root/'prefix/prefix.npz') as z:self.assertEqual(len(z['state']),2);self.assertTrue(np.allclose(z['actions'][:,0],[.001,.002]))
            normalizer=types.SimpleNamespace(normalize=lambda x,k:x)
            data=c.CarryPrefix(root/'prefix/manifest.json',normalizer);rgb,state,actions,mask=data.sample(1)
            self.assertEqual(mask.sum(),1);self.assertTrue(np.allclose(actions[:,0],.002));self.assertEqual(len(m['raw_rgb_frames']),2)
            receipt=c.admission(root/'prefix/manifest.json',source,norm);self.assertEqual(receipt['frames'],2);self.assertFalse(receipt['source_task_success'])
    def test_failed_audit_formal_seed_and_parent_change_reject(self):
        for change in ('failed','formal','parent'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);case,audit,source,norm=self.fixture(root);a=json.loads(audit.read_text())
                if change=='failed':a['status']='failed'
                elif change=='formal':a['seed']=1040
                else:a['bind_sha256']['trace']='wrong'
                audit.write_text(json.dumps(a))
                with self.assertRaises(ValueError):c.prepare(case,audit,source,norm,root/'prefix')
                self.assertFalse((root/'prefix').exists())
    def test_norm_and_png_tamper_fail_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,audit,source,norm=self.fixture(root);c.prepare(case,audit,source,norm,root/'prefix')
            norm.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'norm'):c.admission(root/'prefix/manifest.json',source,norm)
            norm.write_text('{}');Image.fromarray(np.ones((480,640,3),np.uint8)).save(case/'head_rgb/00000.png')
            with self.assertRaises(ValueError):c.CarryPrefix(root/'prefix/manifest.json',types.SimpleNamespace(normalize=lambda x,k:x))
    def test_v2_failed_tampered_or_wrong_case_rejected(self):
        for kind in ('failed','tampered','wrongcase'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);case,audit,source,norm=self.fixture(root);a=json.loads(audit.read_text());v2=Path(a['independent_v2_audit']['path']);value=json.loads(v2.read_text())
                if kind=='failed':value['passed']=False
                elif kind=='wrongcase':value['episodes'][0]['result_path']=str(root/'other/result.json')
                else:value['unexpected_change']=True
                v2.write_text(json.dumps(value))
                if kind!='tampered':a['independent_v2_audit']['sha256']=c.sha(v2);audit.write_text(json.dumps(a))
                with self.assertRaises(ValueError):c.prepare(case,audit,source,norm,root/'prefix')
                self.assertFalse((root/'prefix').exists())

    def test_sampling60_40_counts_and_defaults(self):
        data=types.SimpleNamespace(indices=list(range(5)),sample=lambda x:x);mix=c.CarryMixture(data,data);random.seed(42)
        for _ in range(1000):mix.sample(0)
        self.assertEqual(sum(mix.draw_counts.values()),1000);self.assertTrue(550<mix.draw_counts['original']<650)
        args=t.parser().parse_args(['--output','UNUSED']);self.assertEqual((args.hours,args.max_steps,args.checkpoint_every),(4.3,1600,400));self.assertFalse(args.save_optimizer);self.assertIsNone(args.carry_prefix_manifest)
        self.assertEqual(t.training_configuration(args)['data_strategy']['validation'],'original10complete190chunks_only')
    def test_synthetic_carry_runner_preserves_original_validation_and_typed_stop(self):
        import torch
        from experiments.pi05_g1_mobile.tests.test_training import Runtime,Dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir();(base/'model.safetensors').write_bytes(b'synthetic')
            args=t.parser().parse_args(['--output',str(root/'run'),'--checkpoint',str(base),'--carry-prefix-manifest','SYNTHETIC','--max-steps','2'])
            admission={'disk_estimate':{'model_bytes':10000,'final_optimizer_bytes_upper_bound':0,'reserve_bytes':0},'manifest_sha256':'manifest','norm_sha256':'norm','official_base':{'sha256':t.sha(base/'model.safetensors')},'frozen_files':{},'carry_prefix':{'manifest_sha256':'prefix'}}
            runtime=Runtime();runtime.dtype_summary={};module=types.ModuleType('experiments.pi05_g1_mobile.model_runtime');module.ModelRuntime=lambda *a,**kw:runtime
            values=[{'flow_loss':1.,'complete':True,'chunks':190,'frames':9000,'episodes':10},{'flow_loss':.9,'complete':True,'chunks':190,'frames':9000,'episodes':10}]
            valset=Dataset();trainset=Dataset();prefix=Dataset()
            with patch.dict(sys.modules,{module.__name__:module}),patch.object(t,'preflight',return_value=admission),patch.object(t,'QuantileNormalizer',return_value=None),patch.object(t,'EpisodeDataset',side_effect=[trainset,valset]),patch.object(c,'CarryPrefix',return_value=prefix),patch.object(t,'validate',side_effect=values) as validate,patch.object(t,'memory_snapshot',return_value={}):
                report=t.train(args)
            self.assertEqual(report['status'],'completed');self.assertEqual(report['training_stopped']['reason'],'max_steps');self.assertEqual(report['step'],2)
            self.assertFalse(report['test_loaded']);self.assertFalse(report['optimizer_payload_saved']);self.assertEqual(report['identity']['carry_family_id'],'native201')
            self.assertTrue(all(call.args[1] is valset for call in validate.call_args_list));self.assertEqual(len(validate.call_args_list),2)
            rows=[json.loads(x) for x in (root/'run/progress.jsonl').read_text().splitlines()];updates=[x for x in rows if x['event']=='optimizer_step']
            self.assertEqual(sum(updates[-1]['source_draw_counts'].values()),16)
            self.assertFalse((root/'run/final_000002/training_state.pt').exists())

    def test_update_validation_checkpoint_and_model_loader_unchanged(self):
        original=ast.parse(Path('experiments/pi05_g1_mobile/training.py').read_text());new=ast.parse(Path(t.__file__).read_text())
        functions=lambda tree:{n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(original),functions(new)
        for name in ['update','validate','checkpoint','StopRequest','precision','disk_estimate']:self.assertEqual(a[name],b[name])
        self.assertNotIn('monkeypatch',Path(t.__file__).read_text().replace('never monkeypatched',''))

if __name__=='__main__':unittest.main()
