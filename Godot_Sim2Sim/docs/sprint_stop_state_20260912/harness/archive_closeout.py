from pathlib import Path
import json,hashlib,shutil,time
r=Path('results/sprint_stop_state_20260912');d=Path('docs/sprint_stop_state_20260912');d.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def tree(p):
 return {str(f.relative_to(p)):('link',str(f.readlink())) if f.is_symlink() else ('file',sha(f)) for f in p.rglob('*') if f.is_file() or f.is_symlink()}
cleanup=[]
for rel,base in [('native_joint_fd/prepared',r/'native_joint_fd/suite/runtime'),('diagnostic/runtime',Path('results/sprint_joint_gpu_20260912/native_trained_tracking_split/suite/runtime')),('target_sampler/runtime',r/'native_joint_fd/suite/runtime'),('target_sampler_retry/runtime',r/'native_joint_fd/suite/runtime')]:
 p=r/rel;a=tree(p);b=tree(base);removed=set(b)-set(a);changed={k for k,v in a.items() if b.get(k)!=v};assert not removed,removed
 allowed={'standalone/stop_state_probe.gd','standalone/main.tscn'} if rel=='diagnostic/runtime' else {'standalone/driver.gd'} if rel.startswith('target_sampler') else set()
 assert changed<=allowed,(rel,changed)
 patch=r/'runtime_patches'/rel
 for key in changed:copy(p/key,patch/key)
 logical=sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
 record=dict(path=str(p),canonical=str(base),changed_files=sorted(changed),source_patch=str(patch),canonical_sha256=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest(),logical_bytes=logical,deleted=False)
 assert p.resolve().is_relative_to(r.resolve());shutil.rmtree(p);record['deleted']=True;cleanup.append(record)
save(r/'closeout/runtime_cleanup.json',cleanup)
old=json.loads(Path('results/sprint_joint_gpu_20260912/closeout/defaults_and_physics.json').read_text());files=[]
for item in old['files']:
 actual=sha(Path(item['path']));assert actual==item['expected'],item['path'];files.append({**item,'actual':actual,'unchanged':True})
save(r/'closeout/defaults_and_physics.json',dict(passed=True,files=files))
for p in r.iterdir():
 if p.is_file() and p.suffix=='.py':copy(p,d/'harness'/p.name)
 elif p.is_file() and p.suffix in ('.json','.md') and p.name!='active_budget.json':copy(p,d/'evidence'/p.name)
for name in ['training_code','full_observer_prototype','runtime_patches']:
 shutil.copytree(r/name,d/'harness'/name)
for p in (r/'closeout').glob('*.json'):copy(p,d/'evidence/closeout'/p.name)
for p in (r/'supervisors').glob('*.log'):copy(p,d/'evidence/supervisors'/p.name)
raw=[];models=[]
for p in r.iterdir():
 if not p.is_dir():continue
 if p.name.startswith(('train_','smoke_','gpu_')) or p.name in ['native_joint_fd','target_sampler_retry']:
  for name in ['completed.json','config.json','metrics.jsonl','parity.json','paired_acceptance.json','control.json']:
   if (p/name).exists():copy(p/name,d/'evidence'/p.name/name)
  for f in [*p.glob('*.onnx'),*p.glob('*.pt'),*p.glob('*.manifest.json')]:models.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
 # Traverse evidence only, not the retained canonical runtime/model bank.
 for f in p.rglob('*'):
  if not f.is_file() or any(k in f.relative_to(p).parts for k in ['runtime','models','training_code','runtime_patches','full_observer_prototype']):continue
  if f.name=='trace.json' or f.suffix in ('.npz','.bin') or f.name.endswith('.sensors.json'):
   raw.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
  elif f.name.endswith('state.bin.json') or f.name=='state.bin.json':copy(f,d/'evidence'/f.relative_to(r))
for name in ['reproduce','cold_probe','roundtrip','native_sensor_trace','target_sampler','target_sampler_retry']:
 for f in (r/name).rglob('player.log'):copy(f,d/'evidence'/f.relative_to(r))
s=json.loads((r/'native_joint_fd/suite/summary.json').read_text());episodes=[]
for e in s['episodes']:episodes.append({k:e[k] for k in ['case','seed','completed','returncode','case_sha256','case_path','trace','task_metrics']})
save(d/'evidence/native_joint_fd/episodes.json',dict(source=str(r/'native_joint_fd/suite/summary.json'),source_sha256=sha(r/'native_joint_fd/suite/summary.json'),models=s['models'],errors=s['errors'],episodes=episodes))
save(d/'evidence/local_artifacts.json',dict(models=models,raw=raw,raw_bytes=sum(x['bytes'] for x in raw),notes='Weights, snapshots, exploratory samples and raw traces retained locally; only hashes committed.'))
save(r/'closeout/archive_completed.json',dict(completed=True,files=sum(f.is_file() for f in d.rglob('*')),raw_files=len(raw),raw_bytes=sum(x['bytes'] for x in raw),finished_unix=time.time()))
print((r/'closeout/archive_completed.json').read_text())
