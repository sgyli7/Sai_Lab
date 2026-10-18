"""Explicit geometry body-index/name adapter; original physical math unchanged.

Geometry collision.body indexes the exported geometry bodies list, whose order
can differ from SDK body poses. Map integer -> geometry body name, then let the
original acceptance resolve that name in actual SDK body_names. No new physical
predicate, shape, transform, threshold or simulator input is introduced.

CPU: evaluate_episode(rows, audit_config). Runtime entry (only root may run GPU):
python -m experiments.pi05_g1_mobile.acceptance_body_ids --config CONFIG
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import threading
from . import acceptance as original

ORIGINAL_ACCEPTANCE_SHA256='303cf7d1ba59ad0e055ad8dd6933638449b6533c31319a6b351d6414db1b3b7e'
_ORIGINAL_GEOMETRY=original.geometry
_ORIGINAL_EVALUATE=original.evaluate_episode
_LOCK=threading.RLock()
APPROVED_CPU_MODULES=frozenset('experiments.pi05_g1_mobile.'+name for name in (
    'audit_episodes','audit_episodes_v2','audit_episodes_v3','benchmark','benchmark_v2',
    'report','report_v2','transport_analysis','transport_report'))


def adapt_robot_geometry(robot):
    names=[body['name'] for body in robot['bodies']]
    if not names or any(not isinstance(name,str) or not name for name in names) or len(set(names))!=len(names):
        raise ValueError('Unique nonempty geometry body names required')
    collisions=[]
    for collider in robot['collisions']:
        index=collider['body']
        if type(index) is not int or not 0<=index<len(names):
            raise ValueError('Collision body must be an in-range geometry integer index')
        collisions.append({**collider,'body':names[index]})
    return {**robot,'collisions':collisions}


def _geometry_with_names(*args,**kwargs):
    bundle=_ORIGINAL_GEOMETRY(*args,**kwargs)
    if len(bundle)!=7:raise ValueError('Frozen original geometry tuple contract differs')
    return (*bundle[:4],adapt_robot_geometry(bundle[4]),*bundle[5:])


@contextmanager
def geometry_body_names_context():
    """Foreground process-local binding, always restored, no upstream writes."""
    with _LOCK:
        source=Path(original.__file__)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=ORIGINAL_ACCEPTANCE_SHA256:
            raise ValueError('Frozen original acceptance source changed')
        if original.geometry is not _ORIGINAL_GEOMETRY:
            raise RuntimeError('Unexpected or nested original geometry binding')
        original.geometry=_geometry_with_names
        try:yield
        finally:original.geometry=_ORIGINAL_GEOMETRY


def evaluate_episode(rows,config):
    with geometry_body_names_context():return _ORIGINAL_EVALUATE(rows,config)


body_name_geometry=geometry_body_names_context


def main(argv=None):
    """Default --config runtime, or --module APPROVED_CPU -- original args."""
    args=list(sys.argv[1:] if argv is None else argv)
    cpu='--module' in args
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity-output',type=Path)
    if cpu:
        parser.add_argument('--module',choices=sorted(APPROVED_CPU_MODULES),required=True)
        if '--' not in args:parser.error('CPU module requires explicit -- separator')
        split=args.index('--');options=parser.parse_args(args[:split]);forwarded=args[split+1:]
        if not forwarded:parser.error('Explicit target module arguments required')
        target=options.module
    else:
        parser.add_argument('--config',required=True)
        options=parser.parse_args(args);target='experiments.pi05_g1_mobile.simulator'
        forwarded=['--config',options.config]
    spec=importlib.util.find_spec(target)
    if spec is None or not spec.origin:raise ValueError('Approved module source not found')
    identity={'schema':'pi05_g1_mobile_body_ids_context_identity_v1',
        'adapter_path':str(Path(__file__).resolve()),'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_acceptance_sha256':ORIGINAL_ACCEPTANCE_SHA256,
        'target_module':target,'target_source':spec.origin,'target_sha256':hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
        'scope':'Only geometry collision integer-to-body-name representation; original math and simulator unchanged',
        'cpu_module_mode':cpu}
    if options.identity_output:
        with options.identity_output.open('x') as f:f.write(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity),file=sys.stderr,flush=True)
    previous=sys.argv
    try:
        sys.argv=[target,*forwarded]
        with body_name_geometry():
            if cpu:return runpy.run_module(target,run_name='__main__')
            # Original CLI/parser/runtime, including action/RGB IPC, unchanged.
            from . import simulator
            return simulator.main()
    finally:sys.argv=previous

if __name__=='__main__':main()
