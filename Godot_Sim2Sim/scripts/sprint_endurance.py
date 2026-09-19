"""Run a turning endurance tape inside the unchanged finite game arena."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from sim2sim.paths import sim2sim_root
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.suite import run


def case(seconds):
    if seconds<40 or seconds%20:raise ValueError('Duration must be a multiple of 20 seconds, at least 40')
    segments=[];active=[]
    for at in range(0,seconds,20):
        # Keep turning in one direction across blocks. A repeated all-forward
        # tape eventually walks off the finite arena and tests a different task.
        segments.extend(dict(at=at+t,held=keys) for t,keys in [
            (0,[]),(1,['sprint','fwd']),(4,['sprint','fwd','left']),
            (12,['fwd','left']),(14,['fwd']),(16,[])])
        active.append([at+1,at+12])
    return dict(case='sprint_endurance',skill='walking',mode='walk',seconds=seconds,
        seed=None,protocol='walking_sprint_v1',segments=segments,sprint_intervals=active,
        ordinary_control=False,pair_id='sprint_endurance',scoring=dict(start=0,end=seconds))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--executable',type=Path);source.add_argument('--project',type=Path)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--timeout',type=float,default=300)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    project=args.project or sim2sim_root()/'godot'
    scene=project/'main.tscn';scene_sha=hashlib.sha256(scene.read_bytes()).hexdigest()
    if args.executable:
        build=json.loads((args.executable.parent/'build.json').read_text())
        if build['source_sha256']['godot/main.tscn']!=scene_sha:
            raise RuntimeError('Local arena specification does not identify the exported arena')
    match=re.search(r'\[sub_resource type="BoxShape3D" id="floor_box"\]\s*size = Vector3\(([^)]+)\)',scene.read_text())
    if not match:raise ValueError('Unrecognised finite arena; inspect before benchmarking')
    size=[float(v) for v in match[1].split(',')]
    if size!=[40.,.02,40.]:raise ValueError('This protocol targets the frozen 40 m game floor')
    replay=case(args.seconds);path=args.out/'case.json';atomic_json(path,replay)
    summary=run([path],args.out/'suite',workers=1,project=args.project,
                executable=args.executable,timeout=args.timeout)
    episode=summary['episodes'][0]
    if not episode['completed']:raise RuntimeError(episode['error'])
    data=json.loads(Path(episode['trace']).read_text());rows=data['rows']
    # Use a conservative half-metre margin for the robot footprint. Leaving it
    # invalidates the endurance scenario; it never produces a quality pass.
    outside=next((r for r in rows if max(abs(float(v)) for v in r['body']['base_pos'][:2])>=19.5),None)
    metrics=episode['task_metrics']
    report=dict(protocol='finite_arena_sprint_endurance_v1',completed=True,arena_source_sha256=scene_sha,
        seconds=args.seconds,models=summary['models'],runtime_id=summary['runtime_id'],
        max_abs_xy=max(max(abs(float(v)) for v in r['body']['base_pos'][:2]) for r in rows),
        first_arena_exit_s=None if outside is None else outside['t'],
        arena_valid=outside is None,resets=data['summary']['resets'],
        physical_metrics=metrics,passed=outside is None and metrics['success'] and data['summary']['resets']==0,
        trace=episode['trace'],trace_sha256=hashlib.sha256(Path(episode['trace']).read_bytes()).hexdigest())
    atomic_json(args.out/'result.json',report)
    print({k:v for k,v in report.items() if k not in ['models','physical_metrics']},flush=True)
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
