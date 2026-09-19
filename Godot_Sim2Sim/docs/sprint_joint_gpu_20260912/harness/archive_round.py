from pathlib import Path
import json,hashlib,shutil,time
R=Path('results/sprint_joint_gpu_20260912');D=Path('docs/sprint_joint_gpu_20260912')
D.mkdir(exist_ok=True)  # Explicit bounded retry after archive-only missing-path failure.
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for f in R.iterdir():
 if f.is_file() and f.suffix in ('.py','.md'):
  cp(f,D/'harness'/f.name)
 elif f.is_file() and f.suffix=='.json' and f.name!='active_budget.json':cp(f,D/'evidence'/f.name)
for name in ['frozen_code','tracking_code','paired_code','feedback_code']:
 shutil.copytree(R/name,D/'harness'/name,dirs_exist_ok=True)
for f in (R/'closeout').glob('*.json'):cp(f,D/'evidence/closeout'/f.name)
for f in (R/'teacher').glob('*.json'):cp(f,D/'evidence/teacher'/f.name)
comparison=json.loads((R/'comparison.json').read_text());feedback=json.loads((R/'feedback_comparison.json').read_text());comparison['native_feedback']=feedback
raw=[];case_index={}
for arm in [*comparison,'feedback_disabled_canary']:
 p=R/arm
 for name in ['completed.json','paired_acceptance.json','control.json']:
  if (p/name).exists():cp(p/name,D/'evidence'/arm/name)
 s=json.loads((p/'suite/summary.json').read_text());rows=[]
 for e in s['episodes']:
  row={k:e[k] for k in ['case','seed','completed','returncode','case_path','case_sha256','task_metrics']}
  tr=Path(e['trace'])
  if not tr.exists():raise FileNotFoundError(tr)
  row['trace']={'path':str(tr),'sha256':sha(tr),'bytes':tr.stat().st_size}
  case=Path(e['case_path']);case_index[str(case)]={'sha256':sha(case),'case':json.loads(case.read_text())}
  rows.append(row)
 put(D/'evidence'/arm/'episodes.json',dict(source=str(p/'suite/summary.json'),source_sha256=sha(p/'suite/summary.json'),errors=s['errors'],models=s['models'],episodes=rows))
 raw.extend(x['trace'] for x in rows)
put(D/'evidence/cases.json',dict(schema_version=1,source_paths={k:a['sha256'] for k,a in case_index.items()},cases_by_sha256={a['sha256']:a['case'] for a in case_index.values()}))
trains={};model_index=[]
for p in R.iterdir():
 if not p.is_dir() or not (p.name.startswith('train_') or p.name.startswith('smoke_')):continue
 for name in ['config.json','completed.json','metrics.jsonl','parity.json']:
  if (p/name).exists():cp(p/name,D/'evidence'/p.name/name)
 for f in [*p.glob('*.onnx'),*p.glob('*.pt'),*p.glob('*.manifest.json')]:model_index.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
 if p.name.startswith('train_'):trains[p.name]=json.loads((p/'completed.json').read_text())
gpu={}
for p in sorted(R.glob('post_*')):
 if not p.is_dir() or not (p/'completed.json').exists():continue
 v=json.loads((p/'completed.json').read_text());rows=v.pop('results');v['ordinary_pass']=sum(x['metrics']['success'] for x in rows if x['ordinary']);v['ordinary_count']=sum(x['ordinary'] for x in rows);v['source_sha256']=sha(p/'completed.json');gpu[p.name]=v
 cp(p/'completed.json',D/'evidence'/p.name/'completed.json')
 for f in p.glob('*.npz'):raw.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
for f in (R/'stop_state_probe').iterdir():
 if f.suffix=='.json':cp(f,D/'evidence/stop_state_probe'/f.name)
 if f.suffix=='.npz':raw.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
raw.append(dict(path=str(R/'teacher/observations.npz'),bytes=(R/'teacher/observations.npz').stat().st_size,sha256=sha(R/'teacher/observations.npz')))
put(D/'evidence/local_artifacts.json',dict(models=model_index,raw=raw,raw_bytes=sum(x['bytes'] for x in raw),note='Retained locally; SHA index only, large weights and trajectories excluded from Git.'))
put(D/'RESULT.json',dict(status='not_promoted',goal_complete=False,session=str(R),predecessor_commit='1c80048dc71b1c7ef608080b3409fc15c3e6a9e2',development_seeds=[927000,927015],holdout_seeds=[928000,928049],holdout_used=False,prior_unused_holdouts=[[924000,924049],[926000,926049]],native=comparison,training=trains,gpu_post_action=gpu,tests=json.loads((R/'closeout/tests.json').read_text()),defaults_and_physics_unchanged=True,package_rebuilt=False,os_keyboard_acceptance='not verified; previously locked desktop',source_snapshots='harness/{frozen_code,tracking_code,paired_code,feedback_code}',scope='walking sprint only; no claim of nine-skill completion'))
put(R/'closeout/archive_completed.json',dict(completed=True,files=sum(f.is_file() for f in D.rglob('*')),raw_files=len(raw),raw_bytes=sum(x['bytes'] for x in raw),finished_unix=time.time()))
print((R/'closeout/archive_completed.json').read_text())
