from pathlib import Path
import json,hashlib,shutil,time
r=Path('results/sprint_target_gpu_20260912');d=Path('docs/sprint_target_gpu_20260912');prior=Path('results/sprint_stop_state_20260912/native_joint_fd/suite/runtime')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def tree(p):return {str(f.relative_to(p)):('link',str(f.readlink())) if f.is_symlink() else ('file',sha(f)) for f in p.rglob('*') if f.is_file() or f.is_symlink()}
arms=[f'native_{i:03d}' for i in [6,12]]+[f'native_warm_{i:03d}' for i in [6,12]]+[f'native_initial_teacher_{i:03d}' for i in [6,12]]
for arm in arms:
 result=json.loads((r/arm/'comparison.json').read_text());assert not any(x['development_eligible'] for x in result.values())
assert not (r/'delivery').exists()
records=[];record_file=r/'closeout/runtime_cleanup.json'
if record_file.exists():records=json.loads(record_file.read_text())
tasks=[(name+'/prepared',r/name/'suite/runtime',set()) for name in arms]
tasks += [(name+'/runtime',prior,{'standalone/driver.gd','runtime_assets/deployment.json','runtime_assets/policies/Sprint_Godot.onnx'}) for name in ['train','train_warm','train_initial_teacher','noise_probe','prefix_probe']]
for rel,canonical,allowed in tasks:
 p=r/rel
 if not p.exists():
  assert any(x['path']==str(p) and x['deleted'] for x in records),str(p)
  continue
 actual,base=tree(p),tree(canonical);assert not(set(base)-set(actual)),rel
 changed={k for k,v in actual.items() if base.get(k)!=v};assert changed<=allowed,(rel,changed)
 patch=r/'runtime_patches'/rel;restore=[]
 for name in sorted(changed):
  f=p/name
  if f.suffix=='.onnx':
   actor=r/rel.split('/')[0]/'iteration_011/actor.onnx';assert sha(f)==sha(actor)
   restore.append(dict(path=name,source=str(actor),sha256=sha(actor)))
  else:
   copy(f,patch/name);restore.append(dict(path=name,source=str(patch/name),sha256=sha(patch/name)))
 logical=sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
 record=dict(path=str(p),canonical=str(canonical),changed_files=sorted(changed),restore=restore,logical_bytes=logical,deleted=False,canonical_tree_sha256=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest())
 assert p.resolve().is_relative_to(r.resolve());shutil.rmtree(p);record['deleted']=True;records.append(record);save(record_file,records)
old=json.loads(Path('results/sprint_stop_state_20260912/closeout/defaults_and_physics.json').read_text())
for item in old['files']:
 item['actual']=sha(Path(item['path']));item['unchanged']=item['actual']==item['expected'];assert item['unchanged'],item['path']
save(r/'closeout/defaults_and_physics.json',old)
for p in r.iterdir():
 if p.is_file() and p.suffix=='.py':copy(p,d/'harness'/p.name)
 elif p.is_file() and p.suffix in ['.json','.md'] and p.name!='active_budget.json':copy(p,d/'evidence'/p.name)
for name in ['training_code','warm_prefix_code','initial_teacher_code','runtime_patches']:
 shutil.copytree(r/name,d/'harness'/name,dirs_exist_ok=True)
for p in (r/'supervisors').glob('*.log'):copy(p,d/'evidence/supervisors'/p.name)
for p in (r/'closeout').glob('*.json'):copy(p,d/'evidence/closeout'/p.name)
for arm in arms:
 folder=r/arm
 for name in ['completed.json','control.json','comparison.json','paired_acceptance.json']:copy(folder/name,d/'evidence'/arm/name)
 summary=json.loads((folder/'suite/summary.json').read_text())
 episodes=[{k:e[k] for k in ['case','seed','completed','returncode','case_sha256','case_path','trace','task_metrics']} for e in summary['episodes']]
 save(d/'evidence'/arm/'episodes.json',dict(source=str(folder/'suite/summary.json'),source_sha256=sha(folder/'suite/summary.json'),models=summary['models'],errors=summary['errors'],episodes=episodes))
 for e in summary['episodes']:
  if not e['task_metrics']['success']:
   trace=Path(e['trace']);log=trace.parent/'player.log'
   if log.exists():copy(log,d/'evidence'/arm/'failed_logs'/f"{e['case']}_{e['seed']}.log")
for name in ['train','train_warm','train_initial_teacher']:
 folder=r/name
 for f in folder.glob('*.json'):copy(f,d/'evidence'/name/f.name)
 for iteration in folder.glob('iteration_*'):
  for f in iteration.glob('*.json'):copy(f,d/'evidence'/name/iteration.name/f.name)
  copy(iteration/'rollouts/completed.json',d/'evidence'/name/iteration.name/'rollouts.json')
for name in ['audit','noise_probe','prefix_probe']:
 for f in (r/name).glob('*.json'):copy(f,d/'evidence'/name/f.name)
artifacts=[]
for f in r.rglob('*'):
 if not f.is_file() or any(x in f.relative_to(r).parts for x in ['runtime','models','training_code','warm_prefix_code','initial_teacher_code','runtime_patches']):continue
 if f.name=='trace.json' or f.suffix in ['.onnx','.pt','.npz'] or f.name.endswith('.manifest.json'):
  artifacts.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
save(d/'evidence/local_artifacts.json',dict(files=artifacts,bytes=sum(x['bytes'] for x in artifacts),scope='Raw trajectories, policy graphs, checkpoints and datasets stay local. Hashes and compact evaluation evidence are committed.'))
save(r/'closeout/archive_completed.json',dict(completed=True,files=sum(p.is_file() for p in d.rglob('*')),local_artifacts=len(artifacts),local_bytes=sum(x['bytes'] for x in artifacts),finished_unix=time.time()))
print((r/'closeout/archive_completed.json').read_text(),flush=True)
