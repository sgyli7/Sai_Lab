"""Opt-in import isolation for six unused eager remote assets; no upstream writes.

All retained AST nodes, including their locations, are the official source nodes.
No registry, API, USD path, or selected asset attribute is substituted.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

FULLNAME = 'isaaclab_arena.assets.object_library'
SOURCE_PATH = '/workspace/Arena/isaaclab_arena/assets/object_library.py'
SOURCE_SHA256 = '102c7b55cf9af05dfec92112cfd14c5bb4e51b3dba733359e1e3cff071e2182f'
EXCLUDED_CLASSES = ('Microwave', 'CoffeeMachine', 'Broccoli', 'SweetPotato', 'Jug', 'BeerBottle')


def ast_identity(node):
    return hashlib.sha256(ast.dump(node, include_attributes=True).encode()).hexdigest()


def filtered_source(source, filename):
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError('Official object_library source SHA mismatch: ' + actual)
    tree = ast.parse(source, filename=filename)
    removed = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in EXCLUDED_CLASSES]
    if sorted(n.name for n in removed) != sorted(EXCLUDED_CLASSES):
        raise ValueError('Each excluded class must occur exactly once at module top level')
    retained = [n for n in tree.body if n not in removed]
    before = [ast_identity(n) for n in retained]
    tree.body = retained
    if before != [ast_identity(n) for n in tree.body]:
        raise AssertionError('Retained AST changed')
    receipt = {
        'schema': 'pi05_g1_mobile_unused_asset_import_v1',
        'module': FULLNAME, 'official_source_path': str(filename),
        'official_source_sha256': actual,
        'excluded_classes': list(EXCLUDED_CLASSES),
        'excluded_class_ast_sha256': {n.name: ast_identity(n) for n in removed},
        'retained_node_ast_sha256': before,
        'retained_module_ast_sha256': ast_identity(tree),
        'retained_class_ast_sha256': {n.name: ast_identity(n) for n in retained if isinstance(n, ast.ClassDef)},
        'semantics': 'Only six complete unused class definitions omitted; selected asset AST unchanged; no API/registry response or USD substitute',
    }
    return tree, receipt


class OfflineObjectLoader(importlib.abc.SourceLoader):
    def __init__(self, path):
        self.path = str(path)

    def get_filename(self, fullname):
        if fullname != FULLNAME:
            raise ImportError(fullname)
        return self.path

    def get_data(self, path):
        if path != self.path:
            raise OSError('Only pinned official source may be read')
        return Path(path).read_bytes()

    def get_code(self, fullname):
        # Deliberately bypass SourceLoader's bytecode cache reads and writes.
        filename = self.get_filename(fullname)
        tree, _ = filtered_source(self.get_data(filename), filename)
        return compile(tree, filename, 'exec', dont_inherit=True)


class OfflineObjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = str(path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != FULLNAME:
            return None
        return importlib.util.spec_from_loader(fullname, OfflineObjectLoader(self.path))


def install_offline_assets(source_path=SOURCE_PATH):
    if FULLNAME in sys.modules:
        raise RuntimeError('object_library already loaded; refusing late asset isolation')
    if any(isinstance(f, OfflineObjectFinder) for f in sys.meta_path):
        raise RuntimeError('Asset isolation already installed')
    source_path = str(Path(source_path).resolve(strict=True))
    _, receipt = filtered_source(Path(source_path).read_bytes(), source_path)
    sys.meta_path.insert(0, OfflineObjectFinder(source_path))
    return receipt
