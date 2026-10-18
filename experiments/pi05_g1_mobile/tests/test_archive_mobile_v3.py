import hashlib,json,os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import archive_mobile_v3 as subject


class ArchiveV3Tests(unittest.TestCase):
    def setup_plan(self,root,immutable=True,cache=False):
        source=root/('cache' if cache else 'source');directory=source/'params/ocdbt.process_0/d';directory.mkdir(parents=True)
        first=directory/('a'*32);first.write_bytes(b'ABC')
        second=directory/('b'*32);second.write_bytes(b'XYZ')
        metadata=source/'config.json';metadata.write_text('{}')
        plan={'schema':'pi05_g1_mobile_archive_plan_v1','sources':[{'category':'official_base','path':str(source),'destination':'base','immutable':immutable,'reason':'permanent freeze'}],
            'immutable_payload_files':[{'path':str(first),'bytes':3,'reason':subject.ORBAX_REASON}],
            'reconstruction_commands':['reconstruct pinned source'],'docker_image_id':'sha256:test',
            'required_files':[{'path':str(first),'sha256':hashlib.sha256(b'ABC').hexdigest()}]}
        return plan,first,second,metadata

    def test_real_toy_archive_explicit_links_unlisted_copies_metadata_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,first,second,metadata=self.setup_plan(root);backup=root/'backup';backup.mkdir();pp=root/'plan.json';pp.write_text(json.dumps(plan))
            with patch.object(subject,'ORBAX_SHARDS',{str(first):3}),patch.object(subject,'CATEGORIES',{'official_base'}),patch.object(subject,'BACKUP_ROOT',backup),patch.object(subject,'RESERVE',0),patch.object(subject,'readiness',return_value=({}, {'owned_closed':True})):
                receipt=subject.archive(pp,backup/'new','partial')
            self.assertTrue(receipt['archive_complete']);self.assertEqual(receipt['schema'],'pi05_g1_mobile_archive_receipt_v3')
            retained=backup/'new/retained/base'
            self.assertEqual(first.stat().st_ino,(retained/'params/ocdbt.process_0/d'/first.name).stat().st_ino)
            self.assertNotEqual(second.stat().st_ino,(retained/'params/ocdbt.process_0/d'/second.name).stat().st_ino)
            self.assertNotEqual(metadata.stat().st_ino,(retained/'config.json').stat().st_ino)
            self.assertEqual(first.read_bytes(),b'ABC');self.assertEqual(second.read_bytes(),b'XYZ')

    def test_wrong_size_nonimmutable_cache_and_unknown_rejected(self):
        for mode in ('size','nonimmutable','cache','unknown'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,first,_,_=self.setup_plan(root,immutable=mode!='nonimmutable',cache=mode=='cache')
                if mode=='size':plan['immutable_payload_files'][0]['bytes']=4
                allowed={} if mode=='unknown' else {str(first):3}
                with patch.object(subject,'ORBAX_SHARDS',allowed):
                    with self.assertRaises(ValueError):subject.explicit_immutable_payloads(plan,root.stat().st_dev)

    def test_root_uid_and_cross_filesystem_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,first,_,_=self.setup_plan(root)
            with patch.object(subject,'ORBAX_SHARDS',{str(first):3}):
                with self.assertRaises(ValueError):subject.explicit_immutable_payloads(plan,root.stat().st_dev+1)
                with patch.object(subject.os,'getuid',return_value=0):
                    with self.assertRaises(ValueError):subject.explicit_immutable_payloads(plan,root.stat().st_dev)


if __name__=='__main__':unittest.main()
