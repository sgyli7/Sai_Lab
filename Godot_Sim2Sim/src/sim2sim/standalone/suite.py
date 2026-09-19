"""Bounded native replay suites with explicit completion evidence and resumable failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from sim2sim.godot_proc import _headless_overlay
from sim2sim.paths import sim2sim_root
from sim2sim.research.queue import atomic_json
from .score import score


def verify_clean_environment(image, package, output):
    """Record actual container properties before a suite can claim clean runtime."""
    name='microduck-environment-'+uuid.uuid4().hex
    script='''set -eu
. /etc/os-release
printf 'os=%s version=%s arch=%s\\n' "$ID" "$VERSION_ID" "$(uname -m)"
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] && [ "$(uname -m)" = aarch64 ]
! command -v python
! command -v python3
test ! -e /repo && test ! -e /workspace
test "$(wc -l < /proc/net/route)" -eq 1
printf 'python=absent repository=absent network_routes=none\\n'
'''
    command=['docker','run','--rm','--pull','never','--name',name,'--network','none',
        '--cpuset-cpus',','.join(map(str,sorted(os.sched_getaffinity(0)))),
        '--read-only','--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp',
        '--tmpfs','/tmp:rw,size=128m','--mount',f'type=bind,source={package},target=/app,readonly',
        image,'/bin/sh','-c',script]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        evidence=dict(image=image,command=command,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,passed=result.returncode==0)
        atomic_json(Path(output)/'container_environment.json',evidence)
        if not evidence['passed']:raise RuntimeError('Clean container preflight failed; see container_environment.json')
    finally:
        subprocess.run(['docker','rm','--force',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)


def run_case(case,output,models,executable=None,timeout=45.,resume=False,project=None,container_image=None,runtime_id=None):
    case=Path(case).resolve();directory=Path(output)/case.stem
    checksum=hashlib.sha256(case.read_bytes()).hexdigest()
    marker=directory/'completed.json'
    if resume and marker.exists():
        result=json.loads(marker.read_text())
        if (result['case_sha256']==checksum and result['models']==models
                and result.get('runtime_id')==runtime_id and result.get('container_image')==container_image):return result
        raise RuntimeError('Resume source/case mismatch: '+str(case))
    directory.mkdir(parents=True,exist_ok=True)
    previous=list(directory.glob('attempt_*'))
    attempt=directory/f'attempt_{len(previous)+1:02d}'
    attempt.mkdir(exist_ok=False)
    trace=attempt/'trace.json';overlay=None;container=None;start=time.monotonic()
    try:
        if container_image:
            if not executable:raise ValueError('Container evaluation requires a standalone executable')
            binary=Path(executable).resolve();container='microduck-case-'+uuid.uuid4().hex
            command=['docker','run','--rm','--pull','never','--name',container,
                '--network','none','--read-only','--user',f'{os.getuid()}:{os.getgid()}',
                '--cpuset-cpus',','.join(map(str,sorted(os.sched_getaffinity(0)))),
                '--env','HOME=/tmp','--tmpfs','/tmp:rw,size=128m',
                '--mount',f'type=bind,source={binary.parent},target=/app,readonly',
                '--mount',f'type=bind,source={case},target=/case.json,readonly',
                '--mount',f'type=bind,source={attempt.resolve()},target=/evidence',
                container_image,'/app/'+binary.name,'--headless','--fixed-fps','200',
                '--','--replay=/case.json','--trace=/evidence/trace.json']
        elif executable:
            command=[str(Path(executable).resolve()),'--headless','--fixed-fps','200']
        else:
            overlay=_headless_overlay(Path(project) if project else sim2sim_root()/'godot')
            command=['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn']
        if not container_image:command.extend(['--',f'--replay={case}',f'--trace={trace}'])
        with (attempt/'player.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if process.returncode!=0:raise RuntimeError(f'Player exit {process.returncode}')
        result=score(trace,case)
        if result['models']!=models:raise RuntimeError('Player model hashes differ from frozen suite inputs')
        result.update(case_sha256=checksum,completed=True,returncode=process.returncode,
                      elapsed_s=time.monotonic()-start,command=command,case_path=str(case),
                      container_image=container_image,runtime_id=runtime_id)
        atomic_json(attempt/'metrics.json',result)
        atomic_json(marker,result)
        return result
    except Exception as error:
        result=dict(case=case.stem,completed=False,error=f'{type(error).__name__}: {error}',
                    case_sha256=checksum,elapsed_s=time.monotonic()-start,attempt=str(attempt))
        atomic_json(attempt/'failed.json',result)
        return result
    finally:
        if overlay:shutil.rmtree(overlay)
        if container:
            # subprocess timeout kills the Docker client, not its container.
            # Remove only this attempt's generated container name.
            subprocess.run(['docker','rm','--force',container],stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=15)


def run(cases,output,workers=4,executable=None,timeout=45.,resume=False,project=None,container_image=None):
    if container_image and not executable:raise ValueError('A standalone executable is required for clean-container evaluation')
    cases=[Path(case).resolve() for case in cases]
    if not cases:raise ValueError('A suite requires at least one replay case')
    if len({case.stem for case in cases})!=len(cases):raise ValueError('Replay case filenames must be unique')
    for case in cases:
        if not case.is_file():raise FileNotFoundError(case)
    output=Path(output).resolve()
    if output.exists() and not resume:raise FileExistsError('Use a new suite directory or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'runtime'
    if executable:
        package=Path(executable).resolve().parent
        manifest=(package/'SHA256SUMS').read_bytes()
        for line in manifest.decode().splitlines():
            expected,name=line.split('  ',1);path=(package/name).resolve()
            if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Package integrity mismatch: '+name)
        runtime_id=hashlib.sha256(manifest).hexdigest()
        models={k:v['sha256'] for k,v in json.loads((package/'models.json').read_text()).items()}
        if container_image:verify_clean_environment(container_image,package,output)
    else:
        # Each suite owns a physical snapshot. Workspace edits / model promotion
        # cannot change a later episode while frozen evaluations run in parallel.
        marker=output/'runtime_snapshot.json'
        if not (resume and marker.exists()):
            if snapshot.exists():raise RuntimeError('Incomplete runtime snapshot; preserve it and use a new suite directory')
            source=Path(project) if project else sim2sim_root()/'godot'
            subprocess.run(['cp','--reflink=auto','-a',str(source),str(snapshot)],timeout=60,check=True)
            inputs=runtime_inputs(snapshot)
            atomic_json(marker,dict(schema_version=2,directory=str(snapshot),input_sha256=inputs))
        recorded=json.loads(marker.read_text())
        if recorded.get('schema_version')!=2:
            raise RuntimeError('Older suite lacks full runtime fingerprints; preserve it and start a new suite directory')
        inputs=runtime_inputs(snapshot)
        if inputs!=recorded['input_sha256']:raise RuntimeError('Frozen runtime changed since suite creation')
        runtime_id=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        deployment=json.loads((snapshot/'runtime_assets/deployment.json').read_text())
        models={k:v['sha256'] for k,v in deployment['policies'].items()}
    start=time.time();results=[]
    atomic_json(output/'running.json',dict(started_unix=start,models=models,cases=[str(p) for p in cases]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_case,case,output,models,executable,timeout,resume,snapshot,container_image,runtime_id):case for case in cases}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            short={key:result.get(key) for key in ['case','skill','seed','completed','error','brake_success']}
            short['task_success']=result.get('task_metrics',{}).get('success')
            print(json.dumps(short),flush=True)
    results.sort(key=lambda row:row['case']+str(row.get('seed')))
    summary=dict(started_unix=start,elapsed_s=time.time()-start,models=models,
        runtime_directory=None if executable else str(snapshot),
        container_image=container_image,runtime_id=runtime_id,
        errors=sum(not row['completed'] for row in results),episodes=results)
    atomic_json(output/'summary.json',summary)
    (output/'running.json').unlink()
    return summary


def runtime_inputs(directory):
    directory=Path(directory)
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file() and '.godot' not in p.relative_to(directory).parts and
            (p.suffix in ['.gd','.tscn','.gdextension','.onnx','.json','.godot','.cfg','.so','.bin'] or p.name.endswith('.so.1'))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cases',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--executable',type=Path);parser.add_argument('--timeout',type=float,default=45)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--container-image',help='Run the exported executable in a cached, network-disabled container image')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--select',nargs='*')
    args=parser.parse_args();cases=sorted(args.cases.glob('*.json'))
    if args.select:cases=[case for case in cases if json.loads(case.read_text())['case'] in args.select]
    if not cases:parser.error('No replay cases selected')
    summary=run(cases,args.out,args.workers,args.executable,args.timeout,args.resume,args.project,args.container_image)
    print(json.dumps(dict(episodes=len(summary['episodes']),errors=summary['errors'],elapsed_s=summary['elapsed_s'])))
    if summary['errors']:raise SystemExit(1)
