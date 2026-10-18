"""Tiny independent text/media fixtures only; no actualarchive or GPU."""
import ast,copy,gzip,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from experiments.pi05_g1_mobile import archive_mobile_v5 as v
from experiments.pi05_g1_mobile import archive_mobile_v4 as old

class ArchiveV5Tests(unittest.TestCase):
 def fixture(self,root):
  src=root/'src';src.mkdir();(src/'foo.json').write_bytes((b'{"line":"evidence"}\n'*5000));(src/'foo.json.gz').write_bytes(b'EXISTING_GZ_SUFFIX_PLAIN');(src/'video.mp4').write_bytes(b'TINY_NOT_REAL_VIDEO');(src/'model.safetensors').write_bytes(b'TINY_NOT_REAL_MODEL');p=root/'plan.json';p.write_text(json.dumps({'schema':'pi05_g1_mobile_archive_plan_v1','sources':[{'category':'test','path':str(src),'destination':'source','immutable':True,'reason':'syntheticfixture'}],'required_files':[{'path':str(src/'foo.json'),'sha256':v.sha(src/'foo.json')}],'reconstruction_commands':['syntheticfixture'],'docker_image_id':'sha256:fixture'}));return src,p
 def arch(self,root,src,p,name='archive'):
  with patch.object(v,'BACKUP_ROOT',root),patch.object(v,'CATEGORIES',{'test'}),patch.object(v,'readiness',return_value=({},{})):
   return v.archive(p,root/name,'partial')
 def test_same_v4_readiness_immutable_and_coverage_AST_unchanged(self):
  a=ast.parse(Path(v.__file__).read_text());b=ast.parse(Path(old.__file__).read_text());names=('explicit_immutable_payloads','readiness','enumerate_sources','check_hashes')
  for name in names:self.assertEqual(ast.dump(next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name)),ast.dump(next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name)))
  self.assertEqual(v.RESERVE,24*1024**3);self.assertFalse(v.TEXT_GZIP_POLICY['adopted'])
  with self.assertRaises(ValueError):v.final_gzip_adoption({},'final')
 def test_roundtrip_collision_media_hardlink_and_closure(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root);(src/'nested.json').write_bytes(b'A'*70000);(src/'nested.json.gz').mkdir();(src/'nested.json.gz/inner.json').write_bytes(b'B'*70000);receipt=self.arch(root,src,p);archive=root/'archive';rows=[json.loads(line) for line in (archive/'inventory.jsonl').read_text().splitlines()];text=next(r for r in rows if r.get('type')=='gzip');self.assertEqual(text['bytes'],(src/'foo.json').stat().st_size);self.assertEqual(text['roundtrip_original_sha256'],v.sha(src/'foo.json'));self.assertEqual(text['sha256'],v.sha(src/'foo.json'));self.assertNotEqual(text['relative_path'],text['logical_relative_path']);self.assertEqual((archive/'retained/source/foo.json.gz').read_bytes(),b'EXISTING_GZ_SUFFIX_PLAIN');self.assertEqual((archive/'retained/source/video.mp4').stat().st_ino,(src/'video.mp4').stat().st_ino);self.assertTrue(receipt['archive_complete']);result=v.restore(archive,root/'restore');self.assertEqual(result['status'],'completed')
   for f in src.rglob('*'):
    if f.is_file():self.assertEqual(v.sha(f),v.sha(root/'restore/retained/source'/f.relative_to(src)))
 def test_symlink_text_not_compressed_preserves_resolved_copy(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root);(src/'linked.json').symlink_to(src/'foo.json');self.arch(root,src,p);rows=[json.loads(line) for line in (root/'archive/inventory.jsonl').read_text().splitlines()];row=next(x for x in rows if x.get('source_path')==str(src/'linked.json'));self.assertEqual(row['type'],'copy');self.assertEqual(row['symlink_target'],str(src/'foo.json'));self.assertFalse((root/'archive/retained/source/linked.json').is_symlink())
 def test_uid_mismatch_payload_stays_independent_copy(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root)
   with patch.object(v.os,'getuid',return_value=999999):self.arch(root,src,p)
   for name in ('video.mp4','model.safetensors'):self.assertNotEqual((src/name).stat().st_ino,(root/'archive/retained/source'/name).stat().st_ino)
 def test_gzip_every_header_footer_write_guarded_and_deterministic(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root);source=src/'foo.json';calls=[];real=v.FloorWriter.write
   def observed(self,data):calls.append(len(data));return real(self,data)
   with patch.object(v.FloorWriter,'write',observed):v.gzip_text(source,root/'a.gz',source.stat(),root,lambda:0)
   v.gzip_text(source,root/'b.gz',source.stat(),root,lambda:0);self.assertEqual(v.sha(root/'a.gz'),v.sha(root/'b.gz'));self.assertGreaterEqual(len(calls),8);self.assertEqual(calls[-2:],[4,4]);self.assertEqual((root/'a.gz').read_bytes()[4:8],bytes(4))
 def test_changed_compression_source_rejected_and_partial_retained(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root);source=src/'foo.json';real=v.gzip.GzipFile.write;changed=False
   def mutated(obj,data):
    nonlocal changed
    out=real(obj,data)
    if not changed:
     changed=True
     with source.open('ab') as f:f.write(b'CHANGED')
    return out
   with patch.object(v.gzip.GzipFile,'write',mutated),self.assertRaises(ValueError):self.arch(root,src,p)
   receipt=v.read(root/'archive/archive_receipt.json');self.assertEqual(receipt['status'],'archive_failed_partial_retained');self.assertTrue(list((root/'archive/text_gzip').rglob('*.gz')))
 def test_floor_failure_preserves_staging_typedpartial(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src,p=self.fixture(root);(src/'z.txt').write_bytes(b'mandatoryplain');real=v.shutil.disk_usage
   def available(path):
    if (root/'archive/text_gzip').exists():return SimpleNamespace(free=v.RESERVE+v.CONTROL_HEADROOM)
    return SimpleNamespace(free=v.RESERVE+10*v.CONTROL_HEADROOM)
   with patch.object(v.shutil,'disk_usage',side_effect=available),self.assertRaises(OSError):self.arch(root,src,p)
   self.assertEqual(v.read(root/'archive/archive_receipt.json')['status'],'archive_failed_partial_retained');self.assertTrue((root/'archive/text_gzip').exists())
 def test_restore_escape_duplicate_symlink_bomb_and_no_overwrite(self):
  for kind in ('escape','duplicate','symlink','bomb','existing'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);src,p=self.fixture(root);self.arch(root,src,p);arc=root/'archive';inv=arc/'inventory.jsonl';rows=[json.loads(l) for l in inv.read_text().splitlines()];text=next(r for r in rows if r.get('type')=='gzip');out=root/'restore'
    if kind=='escape':text['logical_relative_path']='../escape'
    elif kind=='duplicate':rows.append(copy.deepcopy(text))
    elif kind=='symlink':stored=arc/text['relative_path'];data=stored.read_bytes();stored.unlink();other=root/'other.gz';other.write_bytes(data);stored.symlink_to(other)
    elif kind=='bomb':text['bytes']=1
    else:out.mkdir()
    if kind!='existing':
     inv.write_text(''.join(json.dumps(r)+'\n' for r in rows));receipt=v.read(arc/'archive_receipt.json');receipt['inventory_sha256']=v.sha(inv);(arc/'archive_receipt.json').write_text(json.dumps(receipt))
    with self.assertRaises((ValueError,FileExistsError)):v.restore(arc,out)
    if kind!='existing':self.assertEqual(v.read(out/'restore_receipt.json')['status'],'restore_failed_partial_retained')

if __name__=='__main__':unittest.main()
