"""New tiny aliases fixtures only; all original source files removed beforeaudit."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import archive_mobile_v5 as v
from experiments.pi05_g1_mobile import audit_archive_v5 as old
from experiments.pi05_g1_mobile import audit_archive_v5_aliases_v2 as new

class AliasTests(unittest.TestCase):
 def fixture(self,root,kind='valid'):
  resolved=root/'frozen_assets';resolved.mkdir();(resolved/'asset.jsonl').write_bytes(b'{"tiny":"aliases"}\n'*5000);alias=root/'old_asset_alias';alias.symlink_to(resolved,target_is_directory=True);meta=root/'meta';meta.mkdir();plan=meta/'original_plan.json';entry={'alias':str(alias),'resolved':str(resolved),'destination':'assets'}
  declared=[entry];required=str(alias/'asset.jsonl')
  if kind=='similar':required=str(alias)+'_SIMILAR/asset.jsonl'
  elif kind=='destination':entry['destination']='WRONG'
  elif kind=='resolved':entry['resolved']=str(root/'not_the_source')
  elif kind=='duplicate':declared.append(copy.deepcopy(entry))
  elif kind=='overlap':declared.append({'alias':str(alias/'sub'),'resolved':str(root/'other'),'destination':'different'})
  elif kind=='escape':entry['destination']='../escape'
  elif kind=='noncanonical':entry['alias']=str(alias)+'/../old_asset_alias'
  data={'schema':'pi05_g1_mobile_archive_plan_v1','sources':[{'category':'test','path':str(resolved),'destination':'assets','original_alias_path':str(alias),'immutable':False},{'category':'test','path':str(meta),'destination':'provenance','immutable':False}],'explicit_directory_alias_sources':declared,'required_files':[{'path':str(resolved/'asset.jsonl'),'sha256':v.sha(resolved/'asset.jsonl')}],'reconstruction_commands':['synthetic'],'docker_image_id':'sha256:fixture'}
  # Build a valid tiny archive first; emulate an actualalias-based dependency in
  # both archivedplan copies, sinceinvalidmappingfixtures are auditoradversaries.
  plan.write_text(json.dumps(data));archive=root/'archive'
  with patch.object(v,'BACKUP_ROOT',root),patch.object(v,'CATEGORIES',{'test'}),patch.object(v,'readiness',return_value=({},{})):v.archive(plan,archive,'partial')
  data['required_files'][0]['path']=required;plan_copy=archive/'retained/provenance/original_plan.json';plan_copy.write_text(json.dumps(data));(archive/'archive_plan.json').write_text(json.dumps(data,indent=2));rows=[json.loads(l) for l in (archive/'inventory.jsonl').read_text().splitlines()]
  for row in rows:
   if row.get('source_path')==str(plan):row['bytes']=plan_copy.stat().st_size;row['sha256']=v.sha(plan_copy)
  inv=archive/'inventory.jsonl';inv.write_text(''.join(json.dumps(row)+'\n' for row in rows));receipt=v.read(archive/'archive_receipt.json');receipt['inventory_sha256']=v.sha(inv);receipt['input_bindings'][str(plan)]=v.sha(plan_copy);(archive/'archive_receipt.json').write_text(json.dumps(receipt))
  alias.unlink();shutil.rmtree(resolved);shutil.rmtree(meta)
  return archive,alias,resolved,meta
 def test_v1_refuses_v2_passes_without_original_sources(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);archive,alias,resolved,meta=self.fixture(root);self.assertEqual(old.audit(archive,root/'v1.json')['status'],'failed');result=new.audit(archive,root/'v2.json');self.assertEqual(result['status'],'passed',result);self.assertEqual(result['explicit_directory_alias_count'],1);self.assertEqual(result['component_alias_lookups'],1);self.assertFalse(result['original_source_FS_resolved'])
 def test_never_resolves_or_stats_original_alias_or_frozen_source(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);archive,alias,resolved,meta=self.fixture(root);real_resolve=Path.resolve;real_stat=Path.stat
   def denied(p):return any(p==r or p.is_relative_to(r) for r in (alias,resolved,meta))
   def resolve(p,*args,**kwargs):
    if denied(p):raise AssertionError('Originalsource filesystem resolve')
    return real_resolve(p,*args,**kwargs)
   def stat(p,*args,**kwargs):
    if denied(p):raise AssertionError('Originalsource filesystem stat')
    return real_stat(p,*args,**kwargs)
   with patch.object(Path,'resolve',resolve),patch.object(Path,'stat',stat):r=new.audit(archive,root/'report.json')
   self.assertEqual(r['status'],'passed',r)
 def test_similar_component_prefix_does_not_map(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);archive,*_=self.fixture(root,'similar');r=new.audit(archive,root/'report.json');self.assertEqual(r['status'],'failed');self.assertIn('Required source absent',r['errors'][0])
 def test_wrong_destination_or_resolved_pair_rejected(self):
  for kind in ('destination','resolved'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);archive,*_=self.fixture(root,kind);self.assertEqual(new.audit(archive,root/'report.json')['status'],'failed')
 def test_duplicate_overlap_escape_noncanonical_rejected(self):
  for kind in ('duplicate','overlap','escape','noncanonical'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);archive,*_=self.fixture(root,kind);self.assertEqual(new.audit(archive,root/'report.json')['status'],'failed')

if __name__=='__main__':unittest.main()
