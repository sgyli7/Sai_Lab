import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE=Path(__file__).with_name('make_formal_segment_candidate_v2.py')
if not MODULE.is_file():MODULE=Path(__file__).resolve().parents[1]/'make_formal_segment_candidate_v2.py'
spec=importlib.util.spec_from_file_location('segment_factory_cpu',MODULE);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))

class SegmentTests(unittest.TestCase):
 def fixture(self,p):
  token=p/'proof.json';write(token,{'synthetic':True});adopt=p/'adoption.json';write(adopt,{'root_reviewed':True,'formal_adoption':True,'formal_trials_before_adoption':0,'pins':{str(token):f.sha(token)}})
  measurement={'module':'experiments.pi05_g1_mobile.acceptance_pose_speed','root_formal_adoption':str(adopt),'root_formal_adoption_sha256':f.sha(adopt)}
  weight=p/'synthetic.safetensors';weight.write_bytes(b'NO_ACTUAL_MODEL');s=weight.stat();entry={'path':str(weight),'sha256':'1'*64,'bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns};artifacts={k:dict(entry) for k in f.POLICIES}
  frozen={str(token):f.sha(token),str(adopt):f.sha(adopt),str(f.FACTORY.resolve()):f.sha(f.FACTORY)};selection=p/'selection.json';write(selection,{'factory_entry':'experiments.pi05_g1_mobile.make_benchmark_plan_pose_speed','model_artifacts':artifacts,'acceptance_measurement':measurement,'frozen_files':dict(frozen)});frozen[str(selection)]=f.sha(selection)
  cases=[{'case_id':f'c{x}_s{seed}','seed':seed,'cluster_id':x} for x in range(5) for seed in range(1040,1044)];jobs=[]
  for ci,case in enumerate(cases):
   order=f.POLICIES[ci%3:]+f.POLICIES[:ci%3]
   for policy in order:
    folder=p/'raw'/policy/case['case_id'];cp=p/'configs'/(policy+'_'+case['case_id']+'.json')
    config={'mode':'policy','policy_id':policy,'case':case,'output_dir':str(folder),'evaluate_acceptance':True,'acceptance_measurement':measurement,'ticks':3000,'episode_seconds':60,'native_action_horizon':50,'replan_interval_ticks':50,'frozen_files':{str(token):f.sha(token),str(adopt):f.sha(adopt)}};write(cp,config);frozen[str(cp)]=f.sha(cp)
    jobs.append({'policy':policy,'case_id':case['case_id'],'config_path':str(cp),'config_sha256':f.sha(cp),'result_path':str(folder/'result.json'),'estimated_wall_seconds':900})
  planpath=p/'plan.json';plan={'schema':'pi05_g1_mobile_benchmark_plan_v1','primary_policies':f.POLICIES,'case_ids':[c['case_id'] for c in cases],'cases':cases,'jobs':jobs,'frozen_files':frozen,'model_artifacts':artifacts,'selection_receipt':str(selection),'acceptance_measurement':measurement,'batch_entry':'experiments.pi05_g1_mobile.batch_pose_speed'};write(planpath,plan)
  phase=p/'phase.json';write(phase,{'schema':'pi05_g1_mobile_execution_phase_sha_v1','status':'verified','gpu_started':False,'plan':str(planpath),'plan_sha256':f.sha(planpath),'frozen_files':frozen,'model_artifacts_observed':{str(weight):entry}})
  pin=patch.object(f,'MOBILE',p);pin.start();self.addCleanup(pin.stop)
  return planpath,phase

 def prior(self,p,planpath):
  plan=f.read(planpath);records=[]
  for i,job in enumerate(plan['jobs'][:3]):
   cp=Path(job['config_path']);cfg=f.read(cp);folder=Path(cfg['output_dir']);write(folder/'config.json',cfg);trace=folder/'trajectory.jsonl';trace.write_text('{}\n')
   write(folder/'result.json',{'schema':'pi05_g1_mobile_case_result_v1','status':'completed','source_interface_passed':True,'policy':job['policy'],'case_id':job['case_id'],'success':False,'acceptance':{'success':False},'trace_sha256':f.sha(trace)})
   lp=cp.with_suffix('.launch.json');log=cp.with_suffix('.launch.log');write(lp,{'schema':'pi05_g1_mobile_case_launch_v1','exit_code':0,'config_sha256':f.sha(cp)});log.write_text('synthetic completed\n')
   records.append({'index':i,'policy':job['policy'],'case_id':job['case_id'],'status':'completed','success':False,'result_path':job['result_path'],'result_sha256':f.sha(job['result_path']),'config_sha256':f.sha(cp),'case_files_sha256':f.inventory(folder),'launch_path':str(lp),'launch_sha256':f.sha(lp),'launch_log':str(log),'launch_log_sha256':f.sha(log),'source_interface_passed':True})
  rp=p/'prior.json';write(rp,{'schema':f.SEGMENT_SCHEMA,'status':'completed','plan_sha256':f.sha(planpath),'segment_start_index':0,'segment_stop_index_exclusive':3,'expected_segment_jobs':3,'expected_total_jobs':60,'previous_completed_jobs':0,'completed_total_jobs':3,'benchmark_complete':False,'jobs':records,'resume_receipts_sha256':{}});return rp

 def test_segment0_pending_5stages_no_model_hash(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);plan,phase=self.fixture(p);original=f.sha
   def guarded(path):
    self.assertNotEqual(Path(path).suffix,'.safetensors');return original(path)
   with patch.object(f,'sha',side_effect=guarded):f.generate(plan,phase,0,[],p/'output')
   d=f.read(p/'output/candidate_plan.json');self.assertFalse(d['root_reviewed']);self.assertEqual(len(d['stages']),5)
   self.assertEqual(d['stages'][1]['argv'][d['stages'][1]['argv'].index('--start-index')+1],'0')
   third=d['stages'][2];self.assertIn('initial_hashes',third['argv'][third['argv'].index('-c')+1]);self.assertIn({'path':['actual_initial_state_match'],'op':'eq','value':True},third['result']['predicates'])
   self.assertNotIn('PENDING_',json.dumps(d));self.assertFalse((p/'raw').exists())

 def test_next_segment_prefix_and_resume_exact(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);plan,phase=self.fixture(p);prior=self.prior(p,plan);f.generate(plan,phase,1,[prior],p/'output');d=f.read(p/'output/candidate_plan.json');a=d['stages'][1]['argv'];self.assertEqual(a[a.index('--start-index')+1],'3');self.assertEqual(a[-2:],['--resume-receipt',str(prior)])
   # Candidate does not claim earlier independent physics audits passed.
   self.assertIn('earlier CPU audit pointers are not inputs',f.read(p/'output/preparation_receipt.json')['prefix_guard_scope'])

 def test_stale_phase_and_bad_order_rejected(self):
  for kind in ['phase','order','modelstat']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);plan,phase=self.fixture(p)
    if kind=='phase':d=f.read(phase);d['plan_sha256']='0'*64;write(phase,d)
    if kind=='order':d=f.read(plan);d['jobs'][0],d['jobs'][1]=d['jobs'][1],d['jobs'][0];write(plan,d);ph=f.read(phase);ph['plan_sha256']=f.sha(plan);write(phase,ph)
    if kind=='modelstat':(p/'synthetic.safetensors').write_bytes(b'CHANGED')
    with self.assertRaises(ValueError):f.generate(plan,phase,0,[],p/'output')
    self.assertFalse((p/'output').exists())

 def test_missing_duplicate_nonprefix_and_tampered_receipt_reject(self):
  for kind in ['missing','duplicate','nonprefix','changedfile','record']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);plan,phase=self.fixture(p);prior=self.prior(p,plan);paths=[prior];index=1
    if kind=='missing':paths=[]
    if kind=='duplicate':paths=[prior,prior];index=2
    if kind=='nonprefix':doc=f.read(prior);doc['segment_start_index']=3;write(prior,doc)
    if kind=='changedfile':Path(f.read(prior)['jobs'][0]['launch_log']).write_text('modified')
    if kind=='record':doc=f.read(prior);doc['jobs'][0]['result_sha256']='0'*64;write(prior,doc)
    with self.assertRaises(ValueError):f.generate(plan,phase,index,paths,p/'output')
    self.assertFalse((p/'output').exists())

 def test_final_segment_counts_and_pure_stdlib_imports(self):
  import sys
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);plan,phase=self.fixture(p)
   # Only final count/predicate plumbing; prior prefix verification tested above.
   with patch.object(f,'verify_prefix',return_value={}):f.generate(plan,phase,19,[],p/'output')
   d=f.read(p/'output/candidate_plan.json');gpu=d['stages'][1]
   values={tuple(x['path']):x['value'] for x in gpu['result']['predicates']}
   self.assertEqual(values[('segment_start_index',)],57);self.assertEqual(values[('completed_total_jobs',)],60);self.assertIs(values[('benchmark_complete',)],True)
   pointer={tuple(x['path']):x['value'] for x in d['stages'][4]['result']['predicates']}
   self.assertEqual(pointer[('next_start_index',)],60);self.assertIs(pointer[('producer60_completed',)],True)

 def test_snapshot_omits_transitive_payload_but_still_rejects_tamper(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);plan,phase=self.fixture(p);large=p/'historical.png';large.write_bytes(b'PNG_TOY_ONLY')
   doc=f.read(plan);doc['frozen_files'][str(large)]=f.sha(large);write(plan,doc);ph=f.read(phase);ph['frozen_files']=doc['frozen_files'];ph['plan_sha256']=f.sha(plan);write(phase,ph)
   result=f.generate(plan,phase,0,[],p/'output');d=f.read(p/'output/candidate_plan.json')
   self.assertNotIn(str(large),d['frozen_files']);self.assertNotIn(str(p/'proof.json'),d['frozen_files'])
   self.assertIn(str(p/'selection.json'),d['frozen_files']);self.assertIn(str(p/'adoption.json'),d['frozen_files'])
   self.assertEqual(sum(Path(name).stat().st_size for name in d['frozen_files']),result['driver_snapshot_source_bytes'])
   chosen={j['config_path'] for j in doc['jobs'][:3]};configs={name for name in d['frozen_files'] if '/configs/' in name};self.assertEqual(configs,chosen)
   large.write_bytes(b'TAMPERED')
   with self.assertRaisesRegex(ValueError,'frozen evidence'):f.generate(plan,phase,0,[],p/'output2')

 def test_pair_guard_matches_benchmark_exact_serialization(self):
  from types import SimpleNamespace
  import hashlib
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);jobs=[]
   for i in range(3):
    folder=p/str(i);folder.mkdir();row={key:[float(i==2)] for key in f.INITIAL_KEYS};(folder/'trajectory.jsonl').write_text(json.dumps(row)+'\n');digest=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();write(folder/'result.json',{'initial_physical_state_schema':'pi05_g1_mobile_initial_kinematic_v1','initial_physical_state_sha256':digest});jobs.append({'result_path':str(folder/'result.json')})
   context={'expected':jobs,'Path':Path,'json':json,'hashlib':hashlib,'batch':SimpleNamespace(read=f.read)}
   with self.assertRaises(AssertionError):exec(f.PAIR_GUARD,context)
   row={key:[0.] for key in f.INITIAL_KEYS};digest=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();(p/'2/trajectory.jsonl').write_text(json.dumps(row)+'\n');write(p/'2/result.json',{'initial_physical_state_schema':'pi05_g1_mobile_initial_kinematic_v1','initial_physical_state_sha256':digest});exec(f.PAIR_GUARD,context);self.assertEqual(len(set(context['initial_hashes'])),1)

if __name__=='__main__':unittest.main()
