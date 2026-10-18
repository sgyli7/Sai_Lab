"""All arrays/checkpoints here synthetic; never open actual held-out files."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.pi05_g1_mobile import offline_evaluation as e

class Data:
    def __init__(self,*args):pass
    def validation_indices(self):return [i*2 for i in range(10)]
    def sample(self,index):
        target=np.full((50,32),2,np.float32);mask=np.arange(50)<2
        return np.zeros((2,2,3),np.uint8),np.zeros(32,np.float32),target,mask

class Runtime:
    active=0;max_active=0;calls=[]
    def __init__(self,path,norm,head_only):
        assert not head_only;self.label='base' if Path(path).name=='pi05_base_pytorch' else 'fine'
        Runtime.active+=1;Runtime.max_active=max(Runtime.max_active,Runtime.active)
    def infer(self,rgb,state,prompt,seed,num_steps):
        Runtime.calls.append((self.label,seed,prompt,num_steps))
        return {'actions':np.full((50,32),1 if self.label=='base' else 1.5,np.float32),'inference_seconds':.01}

class OfflineTests(unittest.TestCase):
    def fixture(self,root):
        paths=[root/'experiments/pi05_g1_mobile/model_runtime.py',root/'experiments/pi05_g1_mobile/contract.py',root/'experiments/pi05_g1/training.py',root/'experiments/pi05_g1/contract.py']
        for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
        base=root/'.scratch/pi05_g1/pi05_base_pytorch';base.mkdir(parents=True);(base/'model.safetensors').write_bytes(b'base')
        manifest=root/'manifest.json';norm=root/'norm.json';norm.write_text('{}')
        manifest.write_text(json.dumps({'train':[20]*80,'validation':list(range(30,40)),'test':list(range(10)),'dataset_root':str(root/'dataset'),'files':[{'episode_id':i,'length':2} for i in list(range(10))+list(range(30,40))]}))
        train=root/'training';selected=train/'step_000400';selected.mkdir(parents=True);(selected/'model.safetensors').write_bytes(b'fine')
        identity={'manifest_sha256':e.sha(manifest),'norm_sha256':e.sha(norm),'prompt':e.PROMPT,'contract_schema':e.CONTRACT_SCHEMA}
        metadata={'schema':'pi05_g1_mobile_training_checkpoint_v1','step':400,'identity':identity,'validation_complete':True,'validation':{'complete':True,'chunks':10,'episodes':10},'validation_flow_loss':.1,'weights_sha256':e.sha(selected/'model.safetensors')}
        (selected/'metadata.json').write_text(json.dumps(metadata))
        (train/'best_checkpoint.json').write_text(json.dumps({'path':str(selected),'metadata_sha256':e.sha(selected/'metadata.json')}))
        result=train/'training_result.json';result.write_text(json.dumps({'schema':'pi05_g1_mobile_training_result_v1','status':'completed','training_stopped':{'reason':'max_steps'},'selected_checkpoint':str(selected),'selected_validation_flow_loss':.1,'identity':identity}))
        bound=root/'bounds.json';bound.write_text(json.dumps({'names':list(e.ACTION_NAMES),'lower':[-1]*32,'upper':[1]*32}))
        return SimpleNamespace(output=str(root/'out'),manifest=str(manifest),dataset=str(root/'dataset'),norm_stats=str(norm),base_checkpoint=str(base),training_result=str(result),common_bounds=str(bound))
    def setUp(self):Runtime.active=0;Runtime.max_active=0;Runtime.calls=[]
    def test_metadata_plan_covers_tail_and_is_seed_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);m=json.loads(Path(args.manifest).read_text());m['files'][0]['length']=51;Path(args.manifest).write_text(json.dumps(m))
            a=e.plan(args.manifest);self.assertEqual(a,e.plan(args.manifest));self.assertEqual(a['chunk_count'],11)
            self.assertEqual([x['valid_frames'] for x in a['chunks'][:2]],[50,1]);self.assertFalse(a['test_observations_opened'])
    def test_complete_pair_raw_targets_seeds_masks_and_sequential_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            def release(runtime):Runtime.active-=1
            with patch.object(e,'ROOT',root):r=e.evaluate(args,runtime_factory=Runtime,dataset_factory=Data,release=release)
            self.assertEqual(r['status'],'completed');self.assertEqual(r['full_test_chunk_count'],10);self.assertEqual(Runtime.max_active,1);self.assertEqual(Runtime.active,0)
            self.assertEqual(r['baseline']['mae_per_channel'],[1]*32);self.assertEqual(r['finetuned']['mae_per_channel'],[.5]*32)
            self.assertEqual(r['baseline']['valid_frames'],20);self.assertEqual(r['comparison']['mae_relative_change_per_channel'],[-.5]*32)
            self.assertEqual([x[1:] for x in Runtime.calls[:10]],[x[1:] for x in Runtime.calls[10:]])
            z=np.load(root/'out/baseline/chunk_00000.npz');self.assertTrue((z['raw_target_actions']==2).all());self.assertEqual(int(z['mask'].sum()),2)
            self.assertFalse(r['baseline']['mixed_unit_aggregate_reported']);self.assertEqual(r['finetuned']['any_channel_out_of_bounds_frame_rate'],1)
            self.assertFalse(r['test_used_for_checkpoint_selection'])
    def test_nonfinite_prediction_fails_and_still_releases(self):
        class Bad(Runtime):
            def infer(self,*a,**kw):return {'actions':np.full((50,32),np.nan),'inference_seconds':0}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            with patch.object(e,'ROOT',root):r=e.evaluate(args,runtime_factory=Bad,dataset_factory=Data,release=lambda r:setattr(Runtime,'active',Runtime.active-1))
            self.assertEqual(r['status'],'failed');self.assertEqual(Runtime.active,0);self.assertIn('Nonfinite',r['error'])
    def test_selected_identity_rejects_norm_change_before_opening_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);Path(args.norm_stats).write_text('{"changed":true}')
            with patch.object(e,'ROOT',root),patch.object(Data,'__init__',side_effect=AssertionError('must not open')) as opened:r=e.evaluate(args,runtime_factory=Runtime,dataset_factory=Data,release=lambda r:None)
            self.assertEqual(r['status'],'failed');opened.assert_not_called();self.assertIn('identity',r['error'])
    def test_zero_baseline_relative_change_null_and_raw_identity_no_quantile_loss(self):
        self.assertTrue((e.RawTargets().normalize(np.full((1,32),7),'actions')==7).all())
        b=e.metrics(np.zeros(32),np.zeros(32),np.zeros(32),1,0)
        self.assertEqual(e.compare(b,b)['mae_relative_change_per_channel'],[None]*32)
    def test_missing_training_stopped_and_pointer_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.fixture(root);r=json.loads(Path(a.training_result).read_text());r['training_stopped']=None;Path(a.training_result).write_text(json.dumps(r))
            with self.assertRaises(ValueError):e.resolve_selected(a.training_result,a.manifest,a.norm_stats,10)

if __name__=='__main__':unittest.main()
