"""Keep compact reproducibility evidence and hashes of large local trajectories."""
from pathlib import Path
import gzip,hashlib,json,shutil,subprocess,tarfile
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913');D=Path('docs/sprint_joint_identification_20260913')
EXCLUDE={'runtime','prepared','export_project','main_game_runtime','.godot'}
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,target):
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
def compact(p,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if p.stat().st_size>131072:
  target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 else:shutil.copy2(p,target)

def main():
 D.mkdir(exist_ok=True)
 paths=[p for p in R.rglob('*') if p.is_file() and not EXCLUDE.intersection(p.relative_to(R).parts)]
 local=[];archived=[]
 for p in sorted(paths):
  rel=p.relative_to(R)
  if p.name=='trace.json' or p.name.startswith('lifecycle_') and p.suffix=='.json' or p.suffix in ['.npz','.avi','.mp4','.png']:
   local.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)));continue
  if p.suffix in ['.py','.gd']:
   copy(p,D/'harness'/rel);archived.append(p)
  elif p.suffix=='.json':
   # Full suite summaries already contain each completed episode's score.
   if 'suite' in rel.parts or 'clean_container' in rel.parts:
    if p.name not in ['summary.json','runtime_snapshot.json','container_environment.json']:continue
   if 'cases' in rel.parts or 'regressions' in rel.parts:continue
   compact(p,D/'evidence'/rel)
  elif p.suffix=='.log' and ('supervisors' in rel.parts or p.stat().st_size<40000):
   target=D/'evidence'/rel.with_name(p.name+'.gz');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 # Capture replay programs as one compressed archive, avoiding hundreds of duplicate Git entries.
 casepaths=[p for p in paths if p.suffix=='.json' and ('cases' in p.relative_to(R).parts or 'regressions' in p.relative_to(R).parts)]
 target=D/'evidence/replay_cases.tar.gz'
 with target.open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
   with tarfile.open(fileobj=z,mode='w') as tar:
    for p in sorted(casepaths):tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
 atomic_json(D/'evidence/local_artifacts.json',dict(files=local,count=len(local),total_bytes=sum(p['bytes'] for p in local),scope='Original trajectories, diagnostics and media remain local; exact SHA256 and absolute path retained. No raw trajectory is silently discarded.'))
 base='d357400b56df25d5f7a7482efeccafe9245d6249'
 for name in ['src/sim2sim/play_input.py','godot/standalone/play_brain.gd','godot/standalone/driver.gd','scripts/sprint_gpu_world.py','src/sim2sim/research/joint_compliance.py']:
  p=D/'harness/base'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',base+':'+name]));archived.append(p)
 protected=json.loads(Path('results/sprint_contact_calibration_20260913/closeout/defaults_and_physics.json').read_text())
 for v in protected['files']:v['actual']=digest(v['path']);v['unchanged']=v['actual']==v['expected']
 promotion=json.loads((R/'default_promotion.json').read_text()) if (R/'default_promotion.json').exists() else {}
 for v in protected['files']:
  v['accepted_default_update']=bool(v['path']=='godot/runtime_assets/policies/Walk_Godot.onnx' and v['actual']==promotion.get('after',{}).get('walking'))
 protected['passed']=all(v['unchanged'] or v['accepted_default_update'] for v in protected['files']);assert protected['passed']
 atomic_json(R/'closeout/defaults_and_physics.json',protected);copy(R/'closeout/defaults_and_physics.json',D/'evidence/closeout/defaults_and_physics.json')
 # A mutated prototype needs its earlier exact snapshot, not just its latest filename.
 hashes={digest(p) for p in archived};sources=[];missing=[]
 for p in paths:
  if p.name!='protocol.json' and not p.name.endswith('_protocol.json'):continue
  data=json.loads(p.read_text());codes=data.get('code_sha256',{})
  if not isinstance(codes,dict):continue
  for name,sha in codes.items():
   found=sha in hashes or (Path(name).is_file() and digest(name)==sha)
   sources.append(dict(protocol=str(p),source=name,sha256=sha,verified=found))
   if not found:missing.append(sources[-1])
 atomic_json(R/'closeout/source_validation.json',dict(passed=not missing,protected_files=len(protected['files']),protocol_source_checks=sources,missing=missing,local_artifact_count=len(local),local_artifact_bytes=sum(p['bytes'] for p in local)))
 copy(R/'closeout/source_validation.json',D/'evidence/closeout/source_validation.json')
 assert not missing,missing
 print(dict(local_artifacts=len(local),bytes=sum(p['bytes'] for p in local),source_checks=len(sources),archived_sources=len(archived)))
if __name__=='__main__':main()
