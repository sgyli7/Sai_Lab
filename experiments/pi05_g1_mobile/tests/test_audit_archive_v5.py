"""New synthetic tiny archives only; never actualarchive/model/test/GPU."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import archive_mobile_v5 as v
from experiments.pi05_g1_mobile import audit_archive_v5 as a

class AuditTests(unittest.TestCase):
 def fixture(self,root,final=False):
  src=root/'source';src.mkdir();(src/'trace.jsonl').write_bytes(b'{"sample":"synthetic"}\n'*4000);(src/'view.mp4').write_bytes(b'NOT_A_REAL_VIDEO');selected=src/'selected';selected.mkdir();model=selected/'model.safetensors';model.write_bytes(b'NOT_A_REAL_MODEL');os.link(model,src/'same_model.safetensors')
  training=src/'training_result.json';benchmark=src/'benchmark_summary.json';offline=src/'offline_result.json'
  if final:
   training.write_text(json.dumps({'schema':'pi05_g1_mobile_training_result_v1','status':'completed','training_stopped':{'complete_optimizer_update_boundary':True},'selected_checkpoint':str(selected)}));cases=['c'+str(i) for i in range(20)];policies=['a','b','c'];benchmark.write_text(json.dumps({'schema':'pi05_g1_mobile_benchmark_summary_v1','complete':True,'evidence_qualified':True,'formal_verdict_available':True,'errors':[],'case_ids':cases,'policies':{p:{'completed_evidence_cases':20} for p in policies},'episodes':[{'policy':p,'case_id':c} for p in policies for c in cases]}));offline.write_text(json.dumps({'schema':'pi05_g1_mobile_offline_error_result_v1','status':'completed','full_test_episode_count':10,'full_test_chunk_count':200,'baseline':{'chunks':200},'finetuned':{'chunks':200},'test_used_for_checkpoint_selection':False,'identity':{'selected':{'training_result_sha256':v.sha(training)},'weights_sha256':{'finetuned':v.sha(model)}}}))
  plan=src/'original_plan.json';plan.write_text(json.dumps({'schema':'pi05_g1_mobile_archive_plan_v1','sources':[{'category':'test','path':str(src),'destination':'tiny','immutable':True,'reason':'synthetic'}],'required_files':[{'path':str(src/'trace.jsonl'),'sha256':v.sha(src/'trace.jsonl')}],'reconstruction_commands':['synthetic'],'docker_image_id':'sha256:synthetic','training_result':str(training),'benchmark':str(benchmark)}));archive=root/'archive'
  with patch.object(v,'BACKUP_ROOT',root),patch.object(v,'CATEGORIES',{'test'}),patch.object(v,'readiness',return_value=({},{})):v.archive(plan,archive,'partial')
  if final:
   r=v.read(archive/'archive_receipt.json');r.update(phase='final',status='retained_final');(archive/'archive_receipt.json').write_text(json.dumps(r))
  return archive
 def rows(self,arc):return [json.loads(x) for x in (arc/'inventory.jsonl').read_text().splitlines()]
 def change_rows(self,arc,rows):
  inv=arc/'inventory.jsonl';inv.write_text(''.join(json.dumps(r)+'\n' for r in rows));receipt=v.read(arc/'archive_receipt.json');receipt['inventory_sha256']=v.sha(inv);receipt['files']=sum(r['type']!='directory' for r in rows);(arc/'archive_receipt.json').write_text(json.dumps(receipt))
 def test_gzip_copy_link_correct_and_shared_inode_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);arc=self.fixture(root);r=a.audit(arc,root/'report.json');self.assertEqual(r['status'],'passed');self.assertEqual(r['counts']['gzip_files'],1);self.assertTrue(r['shared_hardlinks_are_not_independent_copies']);self.assertTrue(r['all_stored_and_logical_SHA_verified'])
   if os.getuid()!=0:self.assertGreaterEqual(r['counts']['shared_inode_hardlinks'],3);self.assertLess(r['physical_inode_SHA_cache_entries'],r['counts']['files']+2)
 def test_gzip_and_plain_damage_rejected(self):
  for kind in ('gzip','copy'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);arc=self.fixture(root);row=next(x for x in self.rows(arc) if x['type']==kind);p=arc/row['relative_path']
    with p.open('r+b') as f:f.write(b'BAD')
    self.assertEqual(a.audit(arc,root/'report.json')['status'],'failed')
 def test_escape_duplicate_symlink_fileancestor_rejected(self):
  for kind in ('escape','duplicate','symlink','ancestor'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);arc=self.fixture(root);rows=self.rows(arc);row=next(x for x in rows if x['type']=='gzip')
    if kind=='escape':row['logical_relative_path']='../escape'
    elif kind=='duplicate':rows.append(copy.deepcopy(row))
    elif kind=='ancestor':row['logical_relative_path']='retained/tiny/original_plan.json/child.jsonl'
    else:
     p=arc/row['relative_path'];other=root/'other.gz';p.rename(other);p.symlink_to(other)
    self.change_rows(arc,rows);self.assertEqual(a.audit(arc,root/'report.json')['status'],'failed')
 def test_size_closure_and_gzip_name_metadata_rejected(self):
  for kind in ('size','closure','name','mtime'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);arc=self.fixture(root);rows=self.rows(arc);row=next(x for x in rows if x['type']=='gzip')
    if kind=='size':row['bytes']+=1
    elif kind=='closure':row['source_path']='/UNRELATED';row['source_resolved_path']='/UNRELATED'
    elif kind=='name':old=arc/row['relative_path'];new=arc/'text_gzip/WRONG.gz';old.rename(new);row['relative_path']=row['stored_relative_path']='text_gzip/WRONG.gz'
    else:
     p=arc/row['relative_path'];data=bytearray(p.read_bytes());data[4]=1;p.write_bytes(data);row['stored_sha256']=v.sha(p)
    self.change_rows(arc,rows);self.assertEqual(a.audit(arc,root/'report.json')['status'],'failed')
 def test_root_uid_cannot_be_labeled_hardlink(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);arc=self.fixture(root);rows=self.rows(arc);row=next(x for x in rows if x['type']!='directory' and x['source_path'].endswith('.safetensors'));row['type']='hardlink';row['source_uid']=0;self.change_rows(arc,rows);self.assertEqual(a.audit(arc,root/'report.json')['status'],'failed')
 def test_final_light3x20_and400_only_no_physics(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);arc=self.fixture(root,True);r=a.audit(arc,root/'report.json');self.assertEqual(r['status'],'passed',r);self.assertTrue(r['final_light_fields']['paired_offline400_complete']);self.assertFalse(r['final_light_fields']['physical_acceptance_recomputed'])
 def test_report_never_inside_archive_or_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);arc=self.fixture(root)
   with self.assertRaises(ValueError):a.audit(arc,arc/'BAD_REPORT.json')
   out=root/'report.json';out.write_text('unchanged')
   with self.assertRaises(ValueError):a.audit(arc,out)
   self.assertEqual(out.read_text(),'unchanged')

if __name__=='__main__':unittest.main()
