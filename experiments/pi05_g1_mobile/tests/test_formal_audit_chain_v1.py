"""Synthetic smallJSON only; no realtest/model/physics/GPU."""
import copy,json,tempfile,unittest
from pathlib import Path
from experiments.pi05_g1_mobile import formal_audit_chain_v1 as c
from experiments.pi05_g1_mobile import make_formal_segment_candidate_v3 as g

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))

class ChainTests(unittest.TestCase):
 def fixture(self,root):
  plan=root/'plan.json';jobs=[{'policy':['a','b','c'][i%3],'case_id':'c'+str(i//3),'result_path':str(root/'raw'/str(i)/'result.json'),'config_sha256':'a'*64} for i in range(60)];write(plan,{'schema':'pi05_g1_mobile_benchmark_plan_v1','jobs':jobs});ph=c.sha(plan);tool=c.hashlib.sha256(c.AUDITOR.read_bytes()).hexdigest();pointers=[]
  for n in range(2):
   start=n*3;end=start+3;folder=root/str(n);rows=[];records=[]
   for i in range(start,end):
    job=jobs[i];rp=Path(job['result_path']);th='b'*64;write(rp,{'status':'completed','source_interface_passed':True,'case_id':job['case_id'],'policy':job['policy'],'success':False,'trace_sha256':th,'initial_physical_state_sha256':'e'*64});rh=c.sha(rp);inputs={str(rp):rh,str(rp.parent/'trajectory.jsonl'):th};row={'schema':'pi05_g1_mobile_episode_cpu_audit_v2','result_path':str(rp),'case_id':job['case_id'],'passed':True,'anomalies':[],'checks':{k:{'passed':True} for k in c.CHECKS},'tool_sha256':tool,'inputs_sha256':inputs};row['audit_sha256']=c.digest(row);rows.append(row);records.append({'index':i,'policy':job['policy'],'case_id':job['case_id'],'result_path':str(rp),'status':'completed','source_interface_passed':True,'config_sha256':job['config_sha256'],'result_sha256':rh,'case_files_sha256':inputs,'success':False})
   producer=folder/'segment.json';write(producer,{'schema':'pi05_g1_mobile_batch_segment_result_v1','status':'completed','plan_sha256':ph,'segment_start_index':start,'segment_stop_index_exclusive':end,'previous_completed_jobs':start,'completed_total_jobs':end,'expected_segment_jobs':3,'expected_total_jobs':60,'jobs':records})
   guard=folder/'guard.json';write(guard,{'schema':'pi05_g1_mobile_formal_segment_guard_v1','status':'passed','passed':True,'mode':'complete','actual_initial_state_match':True,'initial_physical_state_sha256':'e'*64,'plan_sha256':ph,'expected_start_index':start,'verified_completed_prefix':end,'checked_completed_segment_jobs':3,'case_dirs':[str(Path(j['result_path']).parent) for j in jobs[start:end]]})
   audit=folder/'audit.json';write(audit,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'tool_sha256':tool,'episodes':rows});bindings={str(p):c.sha(p) for p in (plan,producer,guard,audit)}
   if n:bindings[str(pointers[-1])]=c.sha(pointers[-1])
   pointer=folder/'pointer.json';write(pointer,{'schema':c.POINTER,'status':'segment_cpu_audit_passed','passed':True,'plan_sha256':ph,'segment_start_index':start,'next_start_index':end,'audited_this_segment':3,'bindings_sha256':bindings});pointers.append(pointer)
  return plan,pointers
 def test_valid_two_nodes_records_only_no_rawtrace_files_required(self):
  with tempfile.TemporaryDirectory() as tmp:
   plan,ptr=self.fixture(Path(tmp));r=c.verify_prior_chain(ptr[-1],6,plan);self.assertTrue(r['passed']);self.assertEqual(len(r['nodes']),2);self.assertFalse(r['raw_payloads_rehashed'])
 def test_changed_audit_or_selfSHA_rejected(self):
  for kind in ('boundchange','selfSHA'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    plan,ptr=self.fixture(Path(tmp));audit=ptr[-1].parent/'audit.json';a=c.read(audit);a['episodes'][0]['checks']['actions']['extra']='changed';write(audit,a)
    if kind=='selfSHA':p=c.read(ptr[-1]);p['bindings_sha256'][str(audit)]=c.sha(audit);write(ptr[-1],p)
    with self.assertRaises(ValueError):c.verify_prior_chain(ptr[-1],6,plan)
 def test_false_initial_guard_rejected_even_when_newbinding_valid(self):
  with tempfile.TemporaryDirectory() as tmp:
   plan,ptr=self.fixture(Path(tmp));guard=ptr[-1].parent/'guard.json';x=c.read(guard);x['actual_initial_state_match']=False;write(guard,x);p=c.read(ptr[-1]);p['bindings_sha256'][str(guard)]=c.sha(guard);write(ptr[-1],p)
   with self.assertRaises(ValueError):c.verify_prior_chain(ptr[-1],6,plan)
 def test_missing_previous_gap_mixedplan_cycle_rejected(self):
  for kind in ('missing','gap','mixed','cycle'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    plan,ptr=self.fixture(Path(tmp));p=c.read(ptr[-1])
    if kind=='missing':del p['bindings_sha256'][str(ptr[0])]
    elif kind=='gap':p['segment_start_index']=0
    elif kind=='mixed':p['plan_sha256']='0'*64
    else:p['bindings_sha256'][str(ptr[-1])]=c.sha(ptr[-1])
    write(ptr[-1],p)
    with self.assertRaises(ValueError):c.verify_prior_chain(ptr[-1],6,plan)
 def test_latest_snapshot_only_three_CPU_evidence_and_GPU_unchanged(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan,ptr=self.fixture(root);chain=c.verify_prior_chain(ptr[-1],6,plan)
   prefix="start=6\noutput=Path(out);report={'mode':mode,}\n";pointer="previous=None\nfinished=end==60\nreport={'bindings_sha256':bindings}\n"
   stages=[{'argv':['python','-c',prefix],'result':{'predicates':[]}}, {'argv':['env','-u','CUDA_VISIBLE_DEVICES','python','GPU_UNCHANGED'],'max_seconds':3000}, {'argv':['CPU'],'result':{'predicates':[]}}, {'argv':['AUDIT'],'result':{'predicates':[]}}, {'argv':['python','-c',pointer,'NONE_FIRST_SEGMENT_ONLY','out'],'result':{'predicates':[]}}]
   original={'stages':stages,'frozen_files':{},'root_reviewed':False};new=g.upgrade(original,2,ptr[-1],chain);self.assertEqual(original['stages'][1],new['stages'][1]);self.assertEqual(len(new['frozen_files']),5)
   self.assertEqual(set(new['frozen_files'])&set(chain['all_small_binding_sha256']),set(chain['latest_direct_CPU_paths']));self.assertNotIn(str(ptr[0]),new['frozen_files']);self.assertFalse(new['root_reviewed']);self.assertIn('verify_prior_chain',new['stages'][0]['argv'][2]);self.assertEqual(new['stages'][4]['argv'][-2],str(ptr[-1]))
 def test_missing_onecheck_or_anomalies_rejected(self):
  for kind in ('check','anomaly'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    plan,ptr=self.fixture(Path(tmp));ap=ptr[-1].parent/'audit.json';a=c.read(ap);row=a['episodes'][0]
    if kind=='check':row['checks'].pop('videos')
    else:row['anomalies']=['bad']
    row['audit_sha256']=c.digest({k:v for k,v in row.items() if k!='audit_sha256'});write(ap,a);p=c.read(ptr[-1]);p['bindings_sha256'][str(ap)]=c.sha(ap);write(ptr[-1],p)
    with self.assertRaises(ValueError):c.verify_prior_chain(ptr[-1],6,plan)

if __name__=='__main__':unittest.main()
