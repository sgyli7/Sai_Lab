"""Synthetic small files only: never touches real training/evidence/model trees."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.pi05_g1_round02.archive_round02 import archive,sha,link_eligible,MAIN,reproduction_snapshot,external_asset_snapshot

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
class RetentionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.root=self.base/'round02';self.back=self.base/'new_archive';self.source=self.base/'source';self.first=self.base/'first';self.root.mkdir();self.source.mkdir();self.first.mkdir()
  (self.source/'README.md').write_text('Synthetic final documentation');(self.source/'tool.py').write_text('pass\n');write(self.first/'archive_manifest.json',{'retained':True})
  write(self.root/'gpu_budget.json',{'runs':[{'status':'completed'}]});progress=self.root/'training_run01/progress.jsonl';progress.parent.mkdir();progress.write_text('{"event":"training_stopped"}\n')
  write(self.root/'benchmark_summary.json',{'schema':'pi05_g1_round02_multi_policy_benchmark_v2','complete':True,'evidence_qualified':True,'errors':[],'infrastructure_incomplete':[],'policies':{k:{'completed':20} for k in MAIN}})
  jobs=[]
  for policy in sorted(MAIN):
   for i in range(20):
    case=self.root/'final'/policy/str(i);write(case/'result.json',{'policy':policy,'case_id':str(i),'success':False,'status':'completed'});cfg=self.root/'configs'/f'{policy}_{i}.json';write(cfg,{'output_dir':str(case)});jobs.append({'config_path':str(cfg),'config_sha256':sha(cfg)})
  write(self.root/'primary_plan.json',{'schema':'pi05_g1_round02_frozen_benchmark_plan_v1','phase':'primary','policies':sorted(MAIN),'jobs':jobs})
  external=[]
  for i in range(14):
   f=self.base/'external_original'/str(i)/('payload.onnx' if i else 'wrapper.py');f.parent.mkdir(parents=True);f.write_bytes(b'small external frozen bytes');external.append({'path':str(f),'sha256':sha(f)})
  write(self.root/'gr00t_discovery.json',{'files':external})
  self.payload=self.root/'frame.png';self.payload.write_bytes(b'synthetic payload not decoded');write(self.root/'immutable_receipt.json',{'path':str(self.payload),'sha256':sha(self.payload)})
  self.metadata=self.root/'metadata.json';write(self.metadata,{'mutable_notes':'copy independently'});write(self.root/'plot_venv/ignored.txt',{});write(self.root/'part.incomplete/ignored.txt',{})
 def tearDown(self):self.tmp.cleanup()
 def run_archive(self):
  receipt=self.root/'human_payload_freeze.json';write(receipt,{'schema':'pi05_g1_round02_immutable_payload_receipt_v1','permanently_frozen':True,'round_root':str(self.root),'files':[{'path':str(self.payload),'bytes':self.payload.stat().st_size,'sha256':sha(self.payload),'reason':'Human permanently freezes this synthetic image after all work'}]})
  return archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt)
 def test_retention_hashes_metadata_copy_and_no_delete(self):
  originals={p:str(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()};m=self.run_archive();self.assertEqual(m['status'],'complete');self.assertTrue((self.back/'retention_manifest.sha256').is_file());self.assertNotEqual(self.metadata.stat().st_ino,(self.back/'round02/metadata.json').stat().st_ino)
  for f in m['files']:self.assertEqual(sha(self.back/f['relative_path']),f['sha256'])
  for p,b in originals.items():self.assertTrue(p.exists());self.assertEqual(str(p.read_bytes()),b)
  self.assertFalse((self.back/'round02/plot_venv').exists());self.assertEqual(len(m['exclusions']),2)
 def test_confirmed_user_payload_same_inode(self):
  if os.getuid()==0:self.skipTest('Root-owned payload must never hardlink')
  m=self.run_archive();f=next(x for x in m['files'] if x['source_path']==str(self.payload));self.assertEqual(f['method'],'hardlink');self.assertEqual(self.payload.stat().st_ino,(self.back/'round02/frame.png').stat().st_ino)
 def test_no_implicit_freeze_from_existing_hash_metadata(self):
  m=archive(self.root,self.back,source_root=self.source,first_round_backup=self.first)
  f=next(x for x in m['files'] if x['source_path']==str(self.payload));self.assertEqual(f['method'],'copy');self.assertFalse(f['immutability_receipt_confirmed'])
 def test_root_owned_rule_and_copy_fallback(self):
  info=SimpleNamespace(st_uid=0,st_dev=self.base.stat().st_dev)
  self.assertFalse(link_eligible(info,self.base.stat().st_dev,True,True))
  with patch('experiments.pi05_g1_round02.archive_round02.link_eligible',return_value=False):m=self.run_archive()
  f=next(x for x in m['files'] if x['source_path']==str(self.payload));self.assertEqual(f['method'],'copy');self.assertNotEqual(self.payload.stat().st_ino,(self.back/'round02/frame.png').stat().st_ino)
 def test_reject_active_and_accounting_open(self):
  for run in [{'status':'running'},{'status':'cleanup_pending'},{'status':'completed','accounting_open':True}]:
   write(self.root/'gpu_budget.json',{'runs':[run]})
   with self.assertRaises(ValueError):self.run_archive()
   self.assertFalse(self.back.exists())
 def test_reject_unstopped_training(self):
  (self.root/'training_run01/progress.jsonl').write_text('{"event":"optimizer_step"}\n')
  with self.assertRaises(ValueError):self.run_archive()
  self.assertFalse(self.back.exists())
 def test_reject_incomplete_main60(self):
  (self.root/'final'/sorted(MAIN)[0]/'0/result.json').unlink()
  with self.assertRaises(ValueError):self.run_archive()
  self.assertFalse(self.back.exists())
 def frozen_receipt(self):
  payload=self.root/'training_run01/step_000200/training_state.pt';payload.parent.mkdir(parents=True);payload.write_bytes(b'synthetic optimizer state')
  receipt=self.root/'explicit_payload_receipt.json';write(receipt,{'schema':'pi05_g1_round02_immutable_payload_receipt_v1','permanently_frozen':True,'round_root':str(self.root),'gpu_budget_sha256':sha(self.root/'gpu_budget.json'),'files':[{'path':str(payload),'bytes':payload.stat().st_size,'sha256':sha(payload),'reason':'Final closed run: permanently immutable optimizer payload'}]})
  return payload,receipt
 def test_explicit_optimizer_freeze_links_and_copies_receipt(self):
  payload,receipt=self.frozen_receipt();m=archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt)
  f=next(x for x in m['files'] if x['source_path']==str(payload));self.assertTrue(f['immutability_receipt_confirmed']);self.assertEqual(f['method'],'hardlink' if os.getuid()!=0 else 'copy')
  self.assertEqual(sha(self.back/'references/immutable_payload_receipt.json'),sha(receipt));self.assertNotEqual((self.back/'references/immutable_payload_receipt.json').stat().st_ino,receipt.stat().st_ino)
 def test_explicit_receipt_tampered_payload_rejected(self):
  payload,receipt=self.frozen_receipt();payload.write_bytes(b'tampered optimizer payload')
  with self.assertRaises(ValueError):archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt)
  self.assertFalse(self.back.exists())
 def test_explicit_missing_freeze_or_wrong_budget_rejected(self):
  payload,receipt=self.frozen_receipt();a=json.loads(receipt.read_text());a['permanently_frozen']=False;write(receipt,a)
  with self.assertRaises(ValueError):archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt)
  self.assertFalse(self.back.exists());a['permanently_frozen']=True;a['gpu_budget_sha256']='0'*64;write(receipt,a)
  with self.assertRaises(ValueError):archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt)
  self.assertFalse(self.back.exists())
 def test_local_harness_openpi_and_recipe_snapshot(self):
  repo=self.base/'synthetic_repo';rr=repo/'.scratch/pi05_g1/round02';rr.mkdir(parents=True);harness=self.base/'temporary_worktree/scripts';harness.mkdir(parents=True)
  seeds=['unitree_g1_t1_source_task','unitree_g1_t1_source_contacts','unitree_g1_t1_source_placement','unitree_g1_static_onnx','unitree_g1_static_server','unitree_g1_source_policy_bridge']
  for name in seeds:(harness/(name+'.py')).write_text('from adjacent_custom import value\n' if name==seeds[0] else '# pinned source\n')
  (harness/'adjacent_custom.py').write_text('value=123\n');(harness/'adjacent.yaml').write_text('fixed: true\n')
  gate=rr/'gate/result.json';write(gate.parent/'config.json',{'source_harness_dir':str(harness)});write(rr/'final_protocol.json',{'bounded_replay_result':str(gate)})
  for name in ['experiments/pi05_g1/Dockerfile.pi05_g1','experiments/pi05_g1/README.md','.scratch/pi05_g1/ENVIRONMENT.md']:
   f=repo/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('reproduction text')
  op=repo/'.scratch/pi05_g1/openpi';src=op/'src/openpi/local.py';src.parent.mkdir(parents=True);src.write_text('LOCAL_MODIFICATION=True\n');write(op/'models/must_not_copy.safetensors',{'not':'source'})
  (op/'pyproject.toml').write_text('[project]\nname="synthetic"\n');(op/'uv.lock').write_text('version=1\n')
  def fakegit(command,**kwargs):
   output='synthetic-commit\n' if 'rev-parse' in command else (' M src/openpi/local.py\n' if 'status' in command else ('LOCAL PATCH\n' if 'diff' in command else 'H src/openpi/local.py\n'))
   self.assertEqual(kwargs['env']['GIT_OPTIONAL_LOCKS'],'0');return SimpleNamespace(stdout=output)
  self.back.mkdir();entries=[]
  with patch('experiments.pi05_g1_round02.archive_round02.subprocess.run',side_effect=fakegit):meta=reproduction_snapshot(rr,repo,self.back,entries,[])
  self.assertIn('adjacent_custom',meta['runtime_local_import_closure']);self.assertTrue(meta['openpi_dirty']);self.assertEqual(meta['openpi_commit'],'synthetic-commit')
  mapped=self.back/'source_snapshot/absolute'/str(src).lstrip('/');self.assertEqual(mapped.read_text(),src.read_text());self.assertNotEqual(mapped.stat().st_ino,src.stat().st_ino)
  self.assertTrue((self.back/'source_snapshot/openpi_local_changes.patch').is_file());self.assertFalse(any('must_not_copy' in e['source_path'] for e in entries))
  for e in entries:self.assertEqual(sha(Path(e['relative_path'])),e['sha256'])
 def test_explicit_external_assets_link_and_urdf_copy_no_environments(self):
  assetroot=self.base/'physical_assets';assetroot.mkdir();usd=assetroot/'robot.usd';usd.write_bytes(b'frozen USD payload');urdf=assetroot/'robot.urdf';urdf.write_text('<robot/>');mesh=assetroot/'meshes/body.STL';mesh.parent.mkdir();mesh.write_bytes(b'frozen mesh');write(assetroot/'tools_env/must_not_copy.json',{'env':True})
  receipt=self.root/'external_freeze.json';write(receipt,{'schema':'pi05_g1_round02_immutable_payload_receipt_v1','permanently_frozen':True,'round_root':str(self.root),'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'reason':'Permanently frozen physical asset'} for p in [usd,urdf,mesh]]})
  m=archive(self.root,self.back,source_root=self.source,first_round_backup=self.first,immutable_payload_receipt=receipt,external_artifact_roots=[assetroot],external_artifact_files=[usd])
  items=[e for e in m['files'] if e['kind']=='explicit_external_physical_asset'];self.assertEqual(len(items),3)
  u=next(e for e in items if e['source_path']==str(usd));self.assertEqual(u['method'],'hardlink' if os.getuid()!=0 else 'copy')
  text=next(e for e in items if e['source_path']==str(urdf));self.assertEqual(text['method'],'copy');self.assertNotEqual((self.back/text['relative_path']).stat().st_ino,urdf.stat().st_ino)
  self.assertTrue(any(e['reason']=='external_environment_or_git_excluded' for e in m['exclusions']));self.assertEqual(m['external_artifact_roots'][1]['overlap_files_already_retained'],1)
 def test_external_asset_symlink_is_rejected(self):
  folder=self.base/'asset_with_link';folder.mkdir();(folder/'bad.usd').symlink_to(self.payload);self.back.mkdir()
  with self.assertRaises(ValueError):external_asset_snapshot([folder],self.back,{},[],[])
 def test_existing_archive_never_overwritten(self):
  self.back.mkdir();sentinel=self.back/'previous';sentinel.write_text('KEEP')
  with self.assertRaises(FileExistsError):self.run_archive()
  self.assertEqual(sentinel.read_text(),'KEEP')
if __name__=='__main__':unittest.main()
