import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('archive_checkpoints', Path(__file__).resolve().parents[1]/'archive_checkpoints.py')
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)

class ArchiveCheckpointsTests(unittest.TestCase):
    def checkpoint(self, path):
        path.mkdir(parents=True)
        metadata = {'step':100,'identity':{'manifest_sha256':'example'}}
        (path/'metadata.json').write_text(json.dumps(metadata))
        for name in archive.PAYLOADS:
            (path/name).write_bytes(b'immutable test payload')
        return metadata

    def test_payload_links_survive_pruning_and_metadata_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, backup = Path(tmp)/'source', Path(tmp)/'backup'
            metadata = self.checkpoint(source)
            row = archive.archive_checkpoint(source,backup)
            for name in archive.PAYLOADS:
                self.assertEqual((source/name).stat().st_ino,(backup/name).stat().st_ino)
            self.assertNotEqual((source/'metadata.json').stat().st_ino,(backup/'metadata.json').stat().st_ino)
            for name in archive.PAYLOADS:
                (source/name).unlink()  # Simulates trainer pruning only disposable source paths.
            (source/'metadata.json').write_text(json.dumps({**metadata,'payload_retention':{'status':'pruned'}}))
            self.assertEqual(json.loads((backup/'metadata.json').read_text()),metadata)
            self.assertEqual(archive.archive_checkpoint(source,backup,previous=row),row)
            self.assertTrue(all((backup/name).is_file() for name in archive.PAYLOADS))

    def test_conflicting_existing_payload_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, backup = Path(tmp)/'source',Path(tmp)/'backup'
            self.checkpoint(source)
            backup.mkdir()
            conflicting = backup/archive.PAYLOADS[0]
            conflicting.write_bytes(b'older historical payload')
            with self.assertRaises(FileExistsError):
                archive.archive_checkpoint(source,backup)
            self.assertEqual(conflicting.read_bytes(),b'older historical payload')

    def test_atomic_incomplete_is_skipped_and_manifest_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, backup = Path(tmp)/'training',Path(tmp)/'backups'
            backup.mkdir()
            self.checkpoint(source/'step_000100.incomplete')
            manifest=backup/'manifest.jsonl';entries={}
            self.assertEqual(archive.scan([source],backup,entries,manifest),0)
            (source/'step_000100.incomplete').rename(source/'step_000100')
            self.assertEqual(archive.scan([source],backup,entries,manifest),1)
            initial=manifest.read_bytes()
            self.assertEqual(archive.scan([source],backup,entries,manifest),0)
            self.assertEqual(manifest.read_bytes(),initial)
            self.assertEqual(len(archive.read_manifest(manifest)),1)

class ArchiveInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('archive_inputs',Path(__file__).resolve().parents[1]/'archive_inputs.py')
        cls.inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.inputs)

    def test_large_payload_link_never_reads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source/model.safetensors';source.parent.mkdir()
            source.write_bytes(b'payload')
            destination=Path(tmp)/'backup/model.safetensors'
            with patch.object(Path,'open',side_effect=AssertionError('Payload must never be read')):
                row=self.inputs.preserve_file(source,destination)
            self.assertEqual(row['method'],'hardlink_no_payload_read')
            self.assertEqual(source.stat().st_ino,destination.stat().st_ino)

    def test_input_metadata_is_copied_and_existing_snapshot_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.json';source.write_text('{"v":1}')
            destination=Path(tmp)/'backup/source.json'
            row=self.inputs.preserve_file(source,destination)
            self.assertEqual(row['method'],'independent_metadata_copy')
            self.assertNotEqual(source.stat().st_ino,destination.stat().st_ino)
            source.write_text('{"v":2}')
            with self.assertRaises(FileExistsError):self.inputs.preserve_file(source,destination)
            self.assertEqual(destination.read_text(),'{"v":1}')

if __name__=='__main__':
    unittest.main()
