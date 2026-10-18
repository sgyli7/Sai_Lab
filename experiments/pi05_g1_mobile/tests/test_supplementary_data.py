"""Synthetic eligibility/alignment tests; mocked acceptance is not physical proof."""
import hashlib,json,tempfile,unittest,random,ast
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.pi05_g1_mobile import supplementary_data as s
from experiments.pi05_g1_mobile.contract import DATASET_NAMES,decode_to_source,encode_state

class SupplementaryTests(unittest.TestCase):
    def cases(self):
        cases=[{'case_id':f'teacher{i}','seed':200+i,'family_id':f'family{i}','stratum':'flat','replay_origin_id':None} for i in range(10)]
        split=s.declared_splits(cases)
        for x in cases:x['split']=split[x['case_id']]
        return cases
    def plan(self,root):
        sourcefile=root/'pinned.py';sourcefile.write_text('synthetic head/controller/source')
        config={k:'synthetic' for k in s.CONFIG_FIELDS};config.update(prompt=s.PROMPT,policy_id='gr00t_n16',native_action_horizon=50,replan_interval_ticks=50)
        result={k:'synthetic' for k in s.RESULT_FIELDS};result['source_joint_names']=list(DATASET_NAMES)
        return {'schema':'pi05_g1_mobile_teacher_plan_v1','decision_frozen_before_collection':True,'split_seed':42,'declared_before_wall_time_ns':1,'formal_case_ids':['formal0'],'cases':self.cases(),'expected_config':config,'expected_result':result,'expected_frozen_files':{str(sourcefile):s.sha(sourcefile)},'frozen_file_roles':{k:[str(sourcefile)] for k in ('head_camera','controller','source')}}
    def fixture(self,root):
        plan=self.plan(root);results=root/'results';results.mkdir()
        for number,case in enumerate(plan['cases']):
            folder=results/case['case_id'];folder.mkdir();(folder/'head_rgb').mkdir();rows=[]
            for tick in range(3):
                q=np.zeros(43,np.float32);q[11]=number*.01+tick*.001;action=np.full(32,number*.001,np.float32)
                rgb=np.full((480,640,3),number*3+tick,np.uint8);Image.fromarray(rgb).save(folder/'head_rgb'/f'{tick:05d}.png')
                rows.append({'control_tick':tick,'physics_steps_since_reset':tick*4,'wall_time_ns':10+tick,'joint_positions':q.tolist(),'source_joint_names':list(DATASET_NAMES),'applied_action32':action.tolist(),'applied_source_action50':decode_to_source(action,q,DATASET_NAMES).tolist(),'head_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
            # Applied source command in row t+1 derives from state t, per controller contract.
            for tick in range(2):rows[tick+1]['applied_source_action50']=decode_to_source(rows[tick+1]['applied_action32'],rows[tick]['joint_positions'],DATASET_NAMES).tolist()
            trace=folder/'trajectory.jsonl';trace.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            config={**plan['expected_config'],'mode':'policy','case':{'case_id':case['case_id'],'seed':case['seed']},'frozen_files':plan['expected_frozen_files']}
            result={**plan['expected_result'],'status':'completed','success':True,'policy':'gr00t_n16','case':config['case'],'trace_sha256':s.sha(trace),'completed_control_ticks':2,'body_names':[],'contact_view':{}}
            (folder/'config.json').write_text(json.dumps(config));(folder/'result.json').write_text(json.dumps(result))
        planpath=root/'plan.json';planpath.write_text(json.dumps(plan));source=root/'source.json';source.write_text(json.dumps({'train':list(range(80)),'validation':list(range(80,90)),'test':list(range(90,100))}))
        return results,planpath,source
    def test_seed42_independent8_1_1_and_family_no_leak(self):
        cases=self.cases();split=s.declared_splits(cases)
        self.assertEqual([list(split.values()).count(k) for k in ('train','validation','test')],[8,1,1])
        self.assertEqual(split,s.declared_splits(list(reversed(cases))))
        variants=cases+[dict(cases[0],case_id='stretch',replay_origin_id=cases[0]['family_id'])]
        grouped=s.declared_splits(variants);self.assertEqual(grouped['stretch'],grouped[cases[0]['case_id']])
    def test_formal_seed_and_family_lie_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan(Path(tmp));plan['cases'][0]['seed']=1040
            with self.assertRaisesRegex(ValueError,'Formal'):s.validate_plan(plan)
            plan=self.plan(Path(tmp));plan['cases'][0]['replay_origin_id']='different'
            with self.assertRaisesRegex(ValueError,'family'):s.validate_plan(plan)
    def test_complete_synthetic_conversion_raw_png_nextrow_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results,plan,source=self.fixture(root)
            m=s.convert(results,plan,source,root/'converted',auditor=lambda *a:{'success':True,'policy_external_interventions':0})
            self.assertEqual(m['status'],'completed');self.assertEqual([len(m[k]) for k in ('train','validation','test')],[8,1,1])
            class Identity:
                def normalize(self,value,key):return value
            data=s.SupplementaryEpisodes(root/'converted/manifest.json','train',Identity());rgb,state,action,mask=data.sample(0)
            self.assertEqual(action.shape,(50,32));self.assertEqual(mask.sum(),2);self.assertTrue((action[2:]==action[1]).all())
            record=next(x for x in m['episodes'] if x['episode_id']==m['train'][0]);self.assertEqual(len(record['raw_rgb_frames']),2)
            trace=[json.loads(x) for x in (results/record['source_case']/'trajectory.jsonl').read_text().splitlines()]
            self.assertTrue(np.array_equal(action[0],trace[1]['applied_action32']))
    def test_failed_missing_and_pixel_tamper_partial_no_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results,plan,source=self.fixture(root)
            rp=results/'teacher0/result.json';r=json.loads(rp.read_text());r['success']=False;rp.write_text(json.dumps(r))
            (results/'teacher1/result.json').unlink();Image.fromarray(np.ones((480,640,3),np.uint8)).save(results/'teacher2/head_rgb/00000.png')
            m=s.convert(results,plan,source,root/'converted',auditor=lambda *a:{'success':True,'policy_external_interventions':0})
            self.assertEqual(m['status'],'partial');self.assertFalse(m['fresh_source_enabled']);self.assertEqual(len(m['rejected']),3)
            with self.assertRaises(ValueError):s.SupplementaryEpisodes(root/'converted/manifest.json','train',None)
    def test_independent_acceptance_rejection_not_raw_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results,plan,source=self.fixture(root)
            m=s.convert(results,plan,source,root/'converted',auditor=lambda *a:{'success':False,'policy_external_interventions':0})
            self.assertEqual(len(m['episodes']),0);self.assertEqual(len(m['rejected']),10)
    def test_union_norm_reads_only_declared_train_payloads(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'dataset';dataset.mkdir()
            rows=[{'observation.state':[0.]*43,'action':[1.]*43,'teleop.base_height_command':[.7],'teleop.navigate_command':[.1,.2,.3]}]
            original=dataset/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),original)
            source=root/'source.json';source.write_text(json.dumps({'dataset_root':str(dataset),'train':list(range(80)),'validation':list(range(80,90)),'test':list(range(90,100)),'files':[{'episode_id':i,'parquet':'train.parquet' if i<80 else 'FORBIDDEN_MISSING.parquet','parquet_sha256':s.sha(original) if i<80 else 'must_not_open'} for i in range(100)]}))
            fresh=root/'fresh.json';npz=root/'teacher_train.npz';np.savez(npz,state=np.full((1,32),2,np.float32),actions=np.full((1,32),3,np.float32))
            fresh.write_text(json.dumps({'status':'completed','fresh_source_enabled':True,'source_manifest_sha256':s.sha(source),'train':[0],'validation':[1],'test':[2],'episodes':[{'episode_id':0,'npz':npz.name,'npz_sha256':s.sha(npz)},{'episode_id':1,'npz':'FORBIDDEN_val.npz'},{'episode_id':2,'npz':'FORBIDDEN_test.npz'}]}))
            calls=[];real=pq.read_table
            def read(path,*a,**kw):calls.append(str(path));return real(path,*a,**kw)
            with patch.object(pq,'read_table',side_effect=read):stats=s.union_train_norm(source,fresh,root/'norm.json')
            self.assertEqual(len(calls),80);self.assertTrue(all(x==str(original) for x in calls));self.assertAlmostEqual(stats['actions']['mean'][0],83/81,places=5)
            receipt=json.loads((root/'norm.json.receipt.json').read_text());self.assertFalse(receipt['validation_or_test_payloads_opened']);self.assertEqual(receipt['frames'],81)

    def test_mixed_sampling_and_equal_source_validation(self):
        class Data:
            indices=list(range(10))
            def sample(self,i):return i
        mixed=s.MixedTraining(Data(),Data());random.seed(42)
        for _ in range(1000):mixed.sample(0)
        self.assertEqual(sum(mixed.draw_counts.values()),1000);self.assertTrue(400<mixed.draw_counts['original']<600)
        values=iter([{'flow_loss':2,'complete':True,'frames':100,'chunks':2,'episodes':10},{'flow_loss':4,'complete':True,'frames':200,'chunks':4,'episodes':1}])
        result=s.mixed_validate(None,{'original':None,'supplementary':None},None,0,lambda *a:next(values))
        self.assertEqual(result['flow_loss'],3);self.assertEqual(result['chunks'],6)
        self.assertIsNone(s.mixed_validate(None,{'original':None,'supplementary':None},None,0,lambda *a:None))

if __name__=='__main__':unittest.main()
