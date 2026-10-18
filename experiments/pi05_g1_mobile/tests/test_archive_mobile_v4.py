"""Small CPU-only reserve-policy regression. No real model payload reads."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import archive_mobile_v3 as old,archive_mobile_v4 as new


class NormalizeAllowedMetadata(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace('pi05_g1_mobile_archive_receipt_v4','pi05_g1_mobile_archive_receipt_v3').replace('plus24GiB reserve','plus32GiB reserve')
        return node
    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value=='archive_reserve_policy')]
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
        return self.generic_visit(node)


class ArchiveV4Tests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();payload=source/'weights.npz';payload.write_bytes(b'ABC');metadata=source/'notes.json';metadata.write_text('{}')
        plan={'schema':'pi05_g1_mobile_archive_plan_v1','sources':[{'category':'official_base','path':str(source),'destination':'base','immutable':True,'reason':'toy permanent freeze'}],
            'reconstruction_commands':['toy reconstruct'],'docker_image_id':'sha256:toy','required_files':[{'path':str(payload),'sha256':hashlib.sha256(b'ABC').hexdigest()}]}
        pp=root/'plan.json';pp.write_text(json.dumps(plan));backup=root/'backup';backup.mkdir();return pp,backup,payload,metadata

    def test_ast_all_functions_identical_except_explicit_receipt_metadata(self):
        def functions(module):
            tree=ast.parse(Path(module.__file__).read_text())
            return {n.name:ast.dump(NormalizeAllowedMetadata().visit(n),include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(functions(old),functions(new))
        for name in ('PAYLOAD','EXCLUDED','CATEGORIES','ORBAX_SHARDS','ORBAX_REASON','BACKUP_ROOT'):
            self.assertEqual(getattr(old,name),getattr(new,name))
        self.assertEqual(old.RESERVE,32*2**30);self.assertEqual(new.RESERVE,24*2**30)

    def test_30GiB_toy_rejects_v3_passes_v4_and_preserves_link_copy_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            pp,backup,payload,metadata=self.fixture(Path(tmp));free=SimpleNamespace(free=30*2**30)
            def bindings(module):
                return (patch.object(module,'CATEGORIES',{'official_base'}),patch.object(module,'BACKUP_ROOT',backup),patch.object(module,'readiness',return_value=({}, {'owned_closed':True})),patch.object(module.shutil,'disk_usage',return_value=free))
            a,b,c,d=bindings(old)
            with a,b,c,d,self.assertRaises(OSError):old.archive(pp,backup/'v3','partial')
            a,b,c,d=bindings(new)
            with a,b,c,d:receipt=new.archive(pp,backup/'v4','partial')
            self.assertTrue(receipt['archive_complete']);self.assertEqual(receipt['schema'],'pi05_g1_mobile_archive_receipt_v4')
            self.assertEqual(receipt['archive_reserve_policy']['archive_reserve_bytes'],24*2**30)
            self.assertEqual(receipt['disk_preflight']['reserve_bytes'],24*2**30)
            self.assertEqual(payload.stat().st_ino,(backup/'v4/retained/base/weights.npz').stat().st_ino)
            self.assertNotEqual(metadata.stat().st_ino,(backup/'v4/retained/base/notes.json').stat().st_ino)
            self.assertEqual(payload.read_bytes(),b'ABC');self.assertEqual(metadata.read_text(),'{}')

    def test_failure_retains_typed_partial_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            pp,backup,payload,metadata=self.fixture(Path(tmp))
            with patch.object(new,'CATEGORIES',{'official_base'}),patch.object(new,'BACKUP_ROOT',backup),patch.object(new,'readiness',return_value=({}, {'owned_closed':True})),patch.object(new.shutil,'disk_usage',return_value=SimpleNamespace(free=30*2**30)),patch.object(new.shutil,'copyfileobj',side_effect=OSError('toy copy failure')):
                with self.assertRaises(OSError):new.archive(pp,backup/'failed','partial')
            receipt=json.loads((backup/'failed/archive_receipt.json').read_text())
            self.assertEqual(receipt['status'],'archive_failed_partial_retained');self.assertFalse(receipt['archive_complete']);self.assertFalse(receipt['experiment_complete_claim'])
            self.assertEqual(payload.read_bytes(),b'ABC');self.assertEqual(metadata.read_text(),'{}')

    def test_root_owned_payload_remains_mandatory_copy_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pp,backup,payload,metadata=self.fixture(root);plan=json.loads(pp.read_text())
            original_stat=Path.stat
            def root_uid(path,*args,**kwargs):
                info=original_stat(path,*args,**kwargs)
                if str(path)==str(payload):
                    fields=list(info);fields[4]=0;return os.stat_result(fields)
                return info
            with patch.object(new,'CATEGORIES',{'official_base'}),patch.object(Path,'stat',root_uid):
                records,_=new.enumerate_sources(plan,root.stat().st_dev)
            actual=next(r for r in records if r.get('source')==payload)
            self.assertEqual(actual['info'].st_uid,0);self.assertFalse(actual['link_eligible'])

if __name__=='__main__':unittest.main()
