"""One-shot CPU retention after completed GPU work and all final delivery artifacts.

No deletions, replacement, chmod, daemons or GPU imports. Metadata copies are
independent; hardlinked immutable user-owned payloads remain shared inodes and
must never be edited in place. Unknown immutability falls back to independent copy.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
PAYLOAD={'.safetensors','.pt','.onnx','.data','.png','.jpg','.jpeg','.mp4','.npz','.npy','.pdf','.usd','.usdc','.stl','.obj','.ply','.glb','.gltf','.exr','.hdr','.dds','.ktx','.ktx2','.tga','.tif','.tiff','.bmp','.webp','.usdz','.bin'}
EXCLUDE={'venv','plot_venv','__pycache__','.pytest_cache'}
MAIN={'gr00t_n17','pi05_round01','pi05_round02'}

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load(path):return json.loads(Path(path).read_text())
def stamp(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def excluded(path):return any(p in EXCLUDE or '.incomplete' in p for p in path.parts)

def readiness(root,summary=None,progress=None):
 budget=load(root/'gpu_budget.json')
 for run in budget.get('runs',[]):
  if run.get('status') in {'running','cleanup','cleanup_pending','accounting_open','starting'} or run.get('accounting_open') or run.get('launcher_cleanup_pending'):
   raise ValueError('Active/unreconciled GPU budget refuses archive')
 progress=Path(progress) if progress else root/'training_run01/progress.jsonl'
 events=[json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
 if not events or events[-1].get('event')!='training_stopped':raise ValueError('Formal training has no terminal training_stopped receipt')
 candidates=[Path(summary)] if summary else [p for p in root.rglob('*.json') if not excluded(p.relative_to(root)) and 'summary' in p.name]
 good=[]
 for p in candidates:
  try:a=load(p)
  except (ValueError,OSError):continue
  if (a.get('schema')=='pi05_g1_round02_multi_policy_benchmark_v2' and a.get('complete') is True
      and a.get('evidence_qualified') is True and not a.get('errors') and not a.get('infrastructure_incomplete')
      and MAIN<=set(a.get('policies',{})) and all(a['policies'][k].get('completed')==20 for k in MAIN)):good.append(p)
 if len(good)!=1:raise ValueError('Require exactly one explicit/unique complete qualified main60 benchmark summary')
 # An independent primary60 plan must point at60 actual completed immutable results.
 plans=[]
 for p in root.rglob('*plan*.json'):
  if excluded(p.relative_to(root)):continue
  try:a=load(p)
  except (ValueError,OSError):continue
  if a.get('schema')=='pi05_g1_round02_frozen_benchmark_plan_v1' and a.get('phase')=='primary' and set(a.get('policies',[]))==MAIN and len(a.get('jobs',[]))==60:
   valid=True;counts={k:set() for k in MAIN}
   for job in a['jobs']:
    try:
     cp=Path(job['config_path']);cfg=load(cp);rp=Path(cfg['output_dir'])/'result.json';res=load(rp)
     if sha(cp)!=job['config_sha256'] or res.get('status')!='completed' or type(res.get('success')) is not bool or res['policy'] not in MAIN:valid=False;break
     counts[res['policy']].add(res['case_id'])
    except (ValueError,OSError,KeyError):valid=False;break
   if valid and all(len(ids)==20 for ids in counts.values()) and len({tuple(sorted(v)) for v in counts.values()})==1:plans.append(p)
 if not plans:raise ValueError('Main60 physical trial plan is missing/incomplete')
 return {'gpu_budget_sha256':sha(root/'gpu_budget.json'),'training_progress':str(progress),'training_progress_sha256':sha(progress),'benchmark_summary':str(good[0]),'benchmark_summary_sha256':sha(good[0]),'completed_main60_plans':[str(p) for p in plans]}

def walk(root):
 """Yield entries without following symlinks; excluded directory itself is listed."""
 for path in sorted(root.iterdir()):
  yield path
  if path.is_dir() and not path.is_symlink() and not excluded(path.relative_to(root)):yield from walk_inner(path,root)

def walk_inner(parent,root):
 for path in sorted(parent.iterdir()):
  yield path
  if path.is_dir() and not path.is_symlink() and not excluded(path.relative_to(root)):yield from walk_inner(path,root)

def explicit_immutable_receipt(path,root,ready):
 """An explicit root/user-authorized freeze declaration is required in addition to byte verification."""
 if path is None:return {},None
 path=Path(path).resolve();before=path.stat();receipt=load(path)
 if (receipt.get('schema')!='pi05_g1_round02_immutable_payload_receipt_v1'
     or receipt.get('permanently_frozen') is not True
     or Path(receipt.get('round_root','')).resolve()!=root):
  raise ValueError('Explicit immutable payload receipt schema/freeze/root mismatch')
 if receipt.get('gpu_budget_sha256',ready['gpu_budget_sha256'])!=ready['gpu_budget_sha256']:
  raise ValueError('Immutable receipt belongs to a different quiescent budget')
 files=receipt.get('files');known={}
 if not isinstance(files,list) or not files:raise ValueError('Immutable payload receipt is empty')
 for entry in files:
  source=Path(entry['path'])
  if not source.is_absolute() or source.is_symlink():raise ValueError('Immutable payload requires absolute nonsymlink path')
  source=source.resolve(strict=True);info=source.stat()
  if str(source) in known:raise ValueError('Duplicate immutable payload path')
  if not isinstance(entry.get('reason'),str) or not entry['reason'].strip():raise ValueError('Immutable payload reason missing')
  if not stat.S_ISREG(info.st_mode) or info.st_size!=entry['bytes'] or sha(source)!=entry['sha256']:
   raise ValueError('Immutable payload receipt bytes/SHA mismatch: '+str(source))
  if stamp(info)!=stamp(source.stat()):raise ValueError('Payload changed during freeze-receipt verification')
  known[str(source)]=entry['sha256']
 if stamp(before)!=stamp(path.stat()):raise ValueError('Freeze receipt changed while checking payloads')
 return known,{'path':str(path),'sha256':sha(path),'entries':len(known),'permanently_frozen':True}

def link_eligible(info,destination_device,payload,confirmed):
 return bool(payload and confirmed and info.st_uid!=0 and info.st_uid==os.getuid() and info.st_dev==destination_device)

def save_file(source,target,known,entries,kind):
 if source.is_symlink():raise ValueError('Unexpected symbolic link: '+str(source))
 info=source.stat()
 if not stat.S_ISREG(info.st_mode):raise ValueError('Nonregular source file: '+str(source))
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():raise FileExistsError(target)
 h=sha(source);confirmed=known.get(str(source.resolve()))==h
 if str(source.resolve()) in known and not confirmed:raise ValueError('Pinned immutable receipt mismatch: '+str(source))
 method='copy'
 if link_eligible(info,target.parent.stat().st_dev,source.suffix.lower() in PAYLOAD,confirmed):
  try:os.link(source,target);method='hardlink'
  except OSError:method='copy'
 if method=='copy':
  with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,16*1024*1024)
 if (stamp(source.stat())[:4] != stamp(info)[:4] if method=='hardlink' else stamp(source.stat()) != stamp(info)) or sha(target)!=h:raise ValueError('Source changed or retained SHA mismatch: '+str(source))
 entries.append({'relative_path':str(target),
  'source_path':str(source.resolve()),'bytes':info.st_size,'sha256':h,'source_device':info.st_dev,'source_inode':info.st_ino,'source_uid':info.st_uid,'source_mtime_ns':info.st_mtime_ns,'source_ctime_ns':info.st_ctime_ns,'destination_inode':target.stat().st_ino,'method':method,'kind':kind,'immutability_receipt_confirmed':confirmed})

def reproduction_snapshot(root,repo,backup,entries,omitted):
 """Independent small-source snapshots, including local TempWorktree dependencies."""
 protocol=load(root/'final_protocol.json');gate=Path(protocol['bounded_replay_result'])
 config=load(gate.parent/'config.json');harness=Path(config['source_harness_dir']).resolve()
 if not harness.is_dir():raise ValueError('Required local source harness vanished')
 closure=set();pending=['unitree_g1_t1_source_task','unitree_g1_t1_source_contacts','unitree_g1_t1_source_placement',
                         'unitree_g1_static_onnx','unitree_g1_static_server','unitree_g1_source_policy_bridge']
 while pending:
  name=pending.pop()
  if name in closure:continue
  source=harness/(name+'.py')
  if not source.is_file():raise ValueError('Local custom source dependency missing: '+str(source))
  closure.add(name)
  for node in ast.walk(ast.parse(source.read_text())):
   imports=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else
            [x.name.split('.')[0] for x in node.names] if isinstance(node,ast.Import) else [])
   pending.extend(x for x in imports if (harness/(x+'.py')).is_file() and x not in closure)
 paths=set(harness.glob('*.py'))
 for source in harness.iterdir():
  if source.is_file() and source.suffix.lower() in {'.json','.yaml','.yml','.toml','.ini','.cfg','.txt','.md'}:paths.add(source)
 necessary_configs={Path(config[key]).resolve() for key in ('geometry_definition','robot_geometry','expert_sequence') if key in config}
 paths.update(necessary_configs)
 for value in config.get('assets',{}).values():
  asset=Path(value)
  if asset.suffix.lower() in {'.urdf','.xml','.json','.yaml','.yml','.toml'}:paths.add(asset.resolve())
 history=harness.parents[3]/'docs/g1_t1_source_task.md' if len(harness.parents)>3 else None
 if history is not None and history.is_file():paths.add(history)
 for source in [repo/'experiments/pi05_g1/Dockerfile.pi05_g1',repo/'experiments/pi05_g1/README.md',
                repo/'.scratch/pi05_g1/ENVIRONMENT.md']:
  if not source.is_file():raise ValueError('Required reproduction document missing: '+str(source))
  paths.add(source)
 openpi=repo/'.scratch/pi05_g1/openpi'
 if not openpi.is_dir():raise ValueError('Local OpenPI checkout missing')
 for prefix in ('src','packages/openpi-client/src'):
  paths.update(p for p in (openpi/prefix).rglob('*') if p.is_file() and not excluded(p.relative_to(openpi)))
 for pattern in ('README*','pyproject.toml','uv.lock','requirements*.txt'):
  paths.update(p for p in openpi.glob(pattern) if p.is_file())
 # No .git, LFS stores, model payload, environments or non-source repository trees.
 for source in sorted(paths):
  if source.is_symlink():raise ValueError('Small source symlink needs explicit retention handling: '+str(source))
  maximum_bytes = 64*1024*1024 if source in necessary_configs else 4*1024*1024
  if source.stat().st_size>maximum_bytes:raise ValueError('Unexpected large reproduction source/config: '+str(source))
  save_file(source,backup/'source_snapshot/absolute'/str(source.resolve()).lstrip('/'),{},entries,'local_reproduction_source_or_config')
 env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
 def git(*args):
  return subprocess.run(['git','-c','core.fsmonitor=false','-C',str(openpi),*args],check=True,
                        capture_output=True,text=True,env=env).stdout
 commit=git('rev-parse','HEAD').strip();status=git('status','--porcelain=v1','--untracked-files=all')
 diff=git('diff','--no-ext-diff','--binary','HEAD','--','src','packages/openpi-client/src','pyproject.toml','uv.lock')
 flags=git('ls-files','-v','--','src','packages/openpi-client/src','pyproject.toml','uv.lock')
 if len(diff.encode())>16*1024*1024:raise ValueError('OpenPI source diff unexpectedly large')
 generated={'openpi_source_identity.json':json.dumps({'schema':'pi05_g1_openpi_local_source_snapshot_v1',
  'original_checkout':str(openpi),'commit':commit,'dirty':bool(status),'porcelain_status':status,
  'tracked_file_flags':flags,'scope':'src and openpi-client/src + root README/pyproject/uv.lock captured as independent files, including local ignored source files. Git tree/LFS/models/env not copied.'},indent=2)+'\n',
  'openpi_local_changes.patch':diff}
 for name,data in generated.items():
  target=backup/'source_snapshot'/name
  with target.open('x') as stream:stream.write(data)
  entries.append({'relative_path':str(target),'source_path':str(openpi)+' [read-only git metadata]',
   'bytes':target.stat().st_size,'sha256':sha(target),'source_device':None,'source_inode':None,'source_uid':None,
   'source_mtime_ns':None,'source_ctime_ns':None,'destination_inode':target.stat().st_ino,
   'method':'generated_independent_metadata','kind':'openpi_git_provenance','immutability_receipt_confirmed':False})
 return {'harness_dir':str(harness),'runtime_local_import_closure':sorted(closure),'all_harness_python_files_copied':len(list(harness.glob('*.py'))),
  'source_files_copied':len(paths),'openpi_commit':commit,'openpi_dirty':bool(status),
  'absolute_path_mapping':'source_snapshot/absolute/<original absolute path without leading slash>',
  'scope_limit':'Does not export Isaac OCI image, installed environments, .git/LFS or full SDK source. Simulation assets are preserved separately through explicit external_artifact_roots; unselected external assets remain references only.'}

def external_asset_snapshot(roots,backup,known,entries,omitted):
 """Explicit physical-asset roots only; never recursively include environments/git."""
 blocked=EXCLUDE|{'.git','.venv','env','envs','tools_env','usd_env','conda_cache','node_modules'}
 seen=set();summaries=[]
 def files(source):
  if source.is_symlink():raise ValueError('External artifact symlink requires explicit resolved target selection: '+str(source))
  if source.name in blocked:
   omitted.append({'source_path':str(source),'reason':'external_environment_or_git_excluded','directory':source.is_dir()});return
  if source.is_file():yield source;return
  if not source.is_dir():raise ValueError('External artifact root missing: '+str(source))
  for child in sorted(source.iterdir()):yield from files(child)
 for value in roots or []:
  original=Path(value)
  if not original.is_absolute() or original.is_symlink():raise ValueError('External artifact roots must be absolute nonsymlink paths')
  source=original.resolve(strict=True);count=total=duplicates=0
  for item in files(source):
   key=str(item.resolve());info=item.stat();count+=1;total+=info.st_size
   target=backup/'external'/key.lstrip('/')
   if key in seen or target.exists():
    previous=next((e for e in entries if e['relative_path']==str(target)),None)
    if previous is None or previous['source_path']!=key:raise ValueError('Conflicting external destination: '+str(target))
    duplicates+=1;seen.add(key);continue
   save_file(item,target,known,entries,'explicit_external_physical_asset');seen.add(key)
  summaries.append({'source_root':str(source),'file_count':count,'bytes':total,'overlap_files_already_retained':duplicates})
 return summaries

def archive(round_root,backup_root,*,benchmark_summary=None,training_progress=None,first_round_backup=None,source_root=None,immutable_payload_receipt=None,external_artifact_roots=None,external_artifact_files=None):
 root=Path(round_root).resolve();backup=Path(backup_root).resolve()
 if backup.exists() or backup.is_symlink():raise FileExistsError('Archive destination must be new: '+str(backup))
 if backup.is_relative_to(root):raise ValueError('Backup cannot be inside source round')
 ready=readiness(root,benchmark_summary,training_progress)
 discovery=load(root/'gr00t_discovery.json')
 if len(discovery.get('files',[]))!=14:raise ValueError('Complete14-file GR00T discovery required')
 first=Path(first_round_backup) if first_round_backup else Path('/home/ethan/ProjectBackups/2026-10-16/pi05_g1_first_round')
 if not first.is_dir():raise ValueError('Existing first-round backup reference missing; pass --first-round-backup')
 source=Path(source_root) if source_root else REPO/'experiments/pi05_g1_round02'
 if not (source/'README.md').is_file():raise ValueError('Source README missing')
 # Frozen release discovery explicitly authorizes immutable external model bytes;
 # per-run payloads require the separate human permanent-freeze declaration.
 known={str(Path(e['path']).resolve()):e['sha256'] for e in discovery['files']}
 explicit,freeze_receipt=explicit_immutable_receipt(immutable_payload_receipt,root,ready)
 known.update(explicit)
 backup.mkdir(parents=True,exist_ok=False);entries=[];omitted=[]
 try:
  for path in walk(root):
   rel=path.relative_to(root)
   if excluded(rel):omitted.append({'relative_path':str(rel),'reason':'excluded_environment_cache_or_incomplete','directory':path.is_dir()});continue
   if path.is_dir() and not path.is_symlink():continue
   if path.is_symlink():raise ValueError('Unspecified symlink in evidence; archive refuses: '+str(path))
   save_file(path,backup/'round02'/rel,known,entries,'round02_evidence')
  for path in sorted(source.rglob('*')):
   rel=path.relative_to(source)
   if excluded(rel):
    omitted.append({'relative_path':'source_snapshot/experiments/pi05_g1_round02/'+str(rel),'reason':'excluded_source_cache_or_incomplete','directory':path.is_dir()});continue
   if path.is_file() and (path.suffix=='.py' or path.name.startswith(('README','requirements')) or 'lock' in path.name or path.name=='pyproject.toml'):
    save_file(path,backup/'source_snapshot/experiments/pi05_g1_round02'/rel,{},entries,'source_snapshot')
  # The source imports first-round runtime/contract/acceptance. Preserve those modules too.
  dependency=REPO/'experiments/pi05_g1'
  if source_root is None:
   for path in sorted(dependency.glob('*.py')):save_file(path,backup/'source_snapshot/experiments/pi05_g1'/path.name,{},entries,'inherited_source_dependency')
  reproduction = reproduction_snapshot(root,REPO,backup,entries,omitted) if source_root is None else {'scope':'Synthetic source_root override; reproduction collector tested separately'}
  for entry in discovery['files']:
   path=Path(entry['path'])
   if not path.is_absolute():raise ValueError('External model discovery path must be absolute')
   path=path.resolve(strict=True)
   save_file(path,backup/'external'/str(path).lstrip('/'),known,entries,'pinned_gr00t_external')
  for value in external_artifact_files or []:
   candidate=Path(value)
   if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
    raise ValueError('External artifact file must be an absolute regular nonsymlink file')
  external_roots = external_asset_snapshot([*(external_artifact_roots or []),*(external_artifact_files or [])],backup,known,entries,omitted)
  if freeze_receipt is not None:
   save_file(Path(freeze_receipt['path']),backup/'references/immutable_payload_receipt.json',{},entries,'explicit_freeze_receipt')
   if sha(backup/'references/immutable_payload_receipt.json')!=freeze_receipt['sha256']:
    raise ValueError('Immutable freeze receipt changed during archive')
  references=[]
  for path in sorted(first.iterdir()):
   if path.is_file() and (path.suffix in {'.json','.jsonl'} or path.name=='ARCHIVE_STOP'):
    save_file(path,backup/'references/first_round'/path.name,{},entries,'first_round_reference_manifest');references.append({'path':str(path),'sha256':sha(path)})
  if not references:raise ValueError('First-round reference manifests missing')
  # Recheck quiescence and terminal receipts after lengthy copying. Any failure leaves
  # an explicit partial archive, never removes it or silently retries over its files.
  if readiness(root,ready['benchmark_summary'],ready['training_progress'])!=ready:raise ValueError('Terminal inputs changed during archive')
  for entry in entries:entry['relative_path']=str(Path(entry['relative_path']).relative_to(backup))
  manifest={'schema':'pi05_g1_round02_retention_manifest_v1','status':'complete','created_unix':time.time(),'source_round_root':str(root),'backup_root':str(backup),'readiness':ready,'external_artifact_roots':external_roots,'reproduction_source_snapshot':reproduction,'explicit_immutable_payload_receipt':freeze_receipt,'first_round_reference':{'backup_root':str(first.resolve()),'manifests':references,'scope':'Reference and independent metadata copies only; first-round payloads remain in their original retained backup.'},'files':entries,'exclusions':omitted,'hardlink_warning':'Hardlinks share source inode. Only receipt-confirmed user-owned immutable payloads link; do not edit or overwrite them in place. Metadata, unknown-immutability and root-owned files are independent copies.','no_source_deletions':True}
  data=json.dumps(manifest,indent=2,allow_nan=False)+'\n';(backup/'retention_manifest.json').write_text(data);(backup/'retention_manifest.sha256').write_text(hashlib.sha256(data.encode()).hexdigest()+'  retention_manifest.json\n');return manifest
 except BaseException as exc:
  (backup/'ARCHIVE_INCOMPLETE.json').write_text(json.dumps({'status':'incomplete','error':repr(exc),'files_retained':entries,'exclusions':omitted},indent=2)+'\n');raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--round-root',required=True);p.add_argument('--backup-root',required=True);p.add_argument('--benchmark-summary');p.add_argument('--training-progress');p.add_argument('--first-round-backup');p.add_argument('--immutable-payload-receipt');p.add_argument('--external-artifact-root',action='append',default=[]);p.add_argument('--external-artifact-file',action='append',default=[]);a=p.parse_args();m=archive(a.round_root,a.backup_root,benchmark_summary=a.benchmark_summary,training_progress=a.training_progress,first_round_backup=a.first_round_backup,immutable_payload_receipt=a.immutable_payload_receipt,external_artifact_roots=a.external_artifact_root,external_artifact_files=a.external_artifact_file);print(json.dumps({'status':m['status'],'files':len(m['files']),'bytes':sum(f['bytes'] for f in m['files']),'hardlinks':sum(f['method']=='hardlink' for f in m['files']),'backup_root':m['backup_root']}))
if __name__=='__main__':main()
