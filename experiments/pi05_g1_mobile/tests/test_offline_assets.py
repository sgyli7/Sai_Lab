import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.pi05_g1_mobile import offline_assets as subject


class OfflineAssetsTests(unittest.TestCase):
    def fixture(self):
        return ('VALUE = 7\nclass BrownBox:\n    physical = 42\n' + ''.join(
            'class '+name+':\n    raise RuntimeError("must never execute remote asset")\n'
            for name in subject.EXCLUDED_CLASSES)).encode()

    def test_official_source_retains_every_other_node_exactly(self):
        path = Path(__file__).resolve().parents[3]/'upstream/unitree_g1/isaaclab_arena_static/isaaclab_arena/assets/object_library.py'
        raw = path.read_bytes()
        original = ast.parse(raw, filename=str(path))
        tree, receipt = subject.filtered_source(raw, str(path))
        expected = [n for n in original.body if not (isinstance(n, ast.ClassDef) and n.name in subject.EXCLUDED_CLASSES)]
        self.assertEqual([ast.dump(n, include_attributes=True) for n in expected],
                         [ast.dump(n, include_attributes=True) for n in tree.body])
        self.assertTrue({'BrownBox', 'BlueSortingBin', 'LibraryObject'} <= receipt['retained_class_ast_sha256'].keys())
        self.assertEqual(receipt['official_source_sha256'], subject.SOURCE_SHA256)

    def test_pinned_sha_rejects_changed_source(self):
        with self.assertRaises(ValueError):subject.filtered_source(self.fixture(), 'fixture.py')

    def test_finder_loader_exact_module_and_no_removed_execution(self):
        raw = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(subject, 'SOURCE_SHA256', hashlib.sha256(raw).hexdigest()):
            path = Path(tmp)/'official.py';path.write_bytes(raw)
            finder = subject.OfflineObjectFinder(path)
            self.assertIsNone(finder.find_spec(subject.FULLNAME+'.other'))
            spec = finder.find_spec(subject.FULLNAME)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.VALUE,7)
            self.assertEqual(module.BrownBox.physical,42)
            self.assertFalse(any(hasattr(module,n) for n in subject.EXCLUDED_CLASSES))
            self.assertEqual(list(Path(tmp).iterdir()),[path])
            path.write_bytes(raw+b'\nchanged = 1\n')
            with self.assertRaises(ValueError):spec.loader.get_code(subject.FULLNAME)

    def test_install_fails_preloaded_and_rechecks_on_load(self):
        raw = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(subject, 'SOURCE_SHA256',hashlib.sha256(raw).hexdigest()):
            path=Path(tmp)/'official.py';path.write_bytes(raw)
            with patch.dict(sys.modules,{subject.FULLNAME:None}):
                with self.assertRaises(RuntimeError):subject.install_offline_assets(path)
            before=list(sys.meta_path)
            try:
                receipt=subject.install_offline_assets(path)
                self.assertEqual(receipt['excluded_classes'],list(subject.EXCLUDED_CLASSES))
                with self.assertRaises(RuntimeError):subject.install_offline_assets(path)
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):sys.meta_path[0].find_spec(subject.FULLNAME).loader.get_code(subject.FULLNAME)
            finally:sys.meta_path[:]=before

    def test_missing_duplicate_exclusions_fail(self):
        for raw in (b'VALUE=1\n',self.fixture()+b'class Microwave: pass\n'):
            with patch.object(subject,'SOURCE_SHA256',hashlib.sha256(raw).hexdigest()):
                with self.assertRaises(ValueError):subject.filtered_source(raw,'fixture.py')

    def test_hook_precedes_all_arena_imports(self):
        source=(Path(subject.__file__).parent/'simulator.py').read_text()
        self.assertLess(source.index("result['offline_unused_assets'] = install_offline_assets()"),
                        source.index('from isaaclab_arena.cli'))


if __name__ == '__main__':unittest.main()
