#!/usr/bin/env python3
"""Download verified build inputs and build matching debug/release ARM64 extensions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
PIN=json.loads((ROOT/'dependencies.json').read_text())


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def download(item):
    path=ROOT/'.deps/downloads'/item['archive'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and sha256(path)==item['sha256']:return path
    temporary=path.with_suffix(path.suffix+'.partial')
    for attempt in range(2):
        try:
            start=time.monotonic()
            request=urllib.request.Request(item['url'],headers={'User-Agent':'MicroDuck-build'})
            with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as stream:
                while chunk:=response.read(1024*1024):
                    if time.monotonic()-start>300:raise TimeoutError('Dependency download exceeded 300 seconds')
                    stream.write(chunk)
            if sha256(temporary)!=item['sha256']:raise ValueError('Dependency checksum mismatch: '+item['archive'])
            temporary.replace(path);return path
        except Exception:
            if attempt:raise
    raise RuntimeError('Dependency download failed')


def command(arguments,timeout=180,cwd=None):
    subprocess.run([str(value) for value in arguments],check=True,timeout=timeout,cwd=cwd)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--skip-downloads',action='store_true')
    parser.add_argument('--double-engine',type=Path)
    parser.add_argument('--double-godotcpp',type=Path)
    parser.add_argument('--double-only',action='store_true')
    args=parser.parse_args()
    if bool(args.double_engine)!=bool(args.double_godotcpp):
        parser.error('--double-engine and --double-godotcpp must be provided together')
    if args.double_only and not args.double_engine:
        parser.error('--double-only requires a matching double engine and godot-cpp checkout')
    if platform.machine() not in ('aarch64','arm64'):raise SystemExit('This deployment target requires Linux ARM64')
    if platform.system()!='Linux':raise SystemExit('This deployment target requires Linux')
    dependencies=ROOT/'.deps';dependencies.mkdir(exist_ok=True)
    runtime=dependencies/PIN['onnxruntime']['directory']
    if not runtime.is_dir():
        if args.skip_downloads:raise FileNotFoundError(runtime)
        with tarfile.open(download(PIN['onnxruntime'])) as archive:archive.extractall(dependencies,filter='data')
    builds=[]
    if not args.double_only:
        bindings=dependencies/'godot-cpp'
        if not bindings.exists():
            if args.skip_downloads:raise FileNotFoundError(bindings)
            command(['git','init',bindings])
            command(['git','-C',bindings,'remote','add','origin',PIN['godot_cpp']['url']])
            command(['git','-C',bindings,'fetch','--depth','1','origin',PIN['godot_cpp']['commit']])
            command(['git','-C',bindings,'checkout','--detach','FETCH_HEAD'])
        commit=subprocess.check_output(['git','-C',str(bindings),'rev-parse','HEAD'],text=True).strip()
        if commit!=PIN['godot_cpp']['commit']:raise RuntimeError('godot-cpp commit differs from the pinned SDK')
        template_dir=Path.home()/'.local/share/godot/export_templates'/PIN['godot_version']
        missing=[name for name in ['linux_debug.arm64','linux_release.arm64'] if not (template_dir/name).is_file()]
        if missing:
            if args.skip_downloads:raise FileNotFoundError(template_dir)
            template_dir.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(download(PIN['templates'])) as archive:
                for name in missing:
                    with archive.open('templates/'+name) as source,(template_dir/name).open('wb') as target:
                        shutil.copyfileobj(source,target)
                    (template_dir/name).chmod(0o755)
        for name in ['linux_debug.arm64','linux_release.arm64']:
            version=subprocess.check_output([str(template_dir/name),'--version'],text=True).strip()
            if not version.startswith(PIN['godot_version']+'.'):raise RuntimeError('Mismatched Godot export template: '+version)
        builds += [('template_debug','build','single',[]),('template_release','build-release','single',[])]
    if args.double_engine:
        double_api=ROOT/'build-double-api';double_api.mkdir(exist_ok=True)
        command([args.double_engine,'--headless','--dump-extension-api'],cwd=double_api)
        api=double_api/'extension_api.json'
        if not api.is_file():raise RuntimeError('Double engine did not produce extension_api.json')
        extra=['-DGODOTCPP_DIR='+str(args.double_godotcpp.resolve()),'-DGODOTCPP_CUSTOM_API_FILE='+str(api.resolve())]
        builds += [('template_debug','build-double','double',extra),('template_release','build-release-double','double',extra)]
    for target,directory,precision,extra in builds:
        command(['cmake','-S',ROOT,'-B',ROOT/directory,'-DCMAKE_BUILD_TYPE=RelWithDebInfo',
                 '-DGODOTCPP_TARGET='+target,'-DMICRODUCK_PRECISION='+precision,*extra])
        command(['cmake','--build',ROOT/directory,'-j',args.jobs])


if __name__=='__main__':main()
