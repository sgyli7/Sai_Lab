"""Verify only owned process groups, actual package hashes, and owned containers."""
import hashlib,json,subprocess,time
from pathlib import Path
from sim2sim.research.budget import cleanup_group,live_group_members,process_start_ticks
p=Path('results/jolt_learning_20260911');d=json.loads((p/'active_budget.json').read_text())
assert list(d['actors'])==['primary'],d['actors']
owners={818888:16740356}
for e in d['events']:
 if e['kind']=='job_started':owners[e['pid']]=e.get('process_start_ticks',owners.get(e['pid']))
records=[]
for pgid,expected in owners.items():
 current=process_start_ticks(pgid);members=live_group_members(pgid)
 if current is not None and expected is not None and current!=expected:
  records.append(dict(pgid=pgid,status='pid_reused_not_our_job',signaled=False));continue
 if members:
  if expected is None:raise RuntimeError('Unverified ownership, inspect manually: '+str(pgid))
  result=cleanup_group(pgid,expected)
 else:result=dict(found=[],remaining=[])
 records.append(dict(pgid=pgid,expected_start_ticks=expected,**result))
 assert not result['remaining']
# This inventory is read-only; unrelated applications are not signaled.
runtimes=[]
for stat in Path('/proc').glob('[0-9]*/stat'):
 try:
  raw=stat.read_text();name=raw.split('(',1)[1].rsplit(')',1)[0];state=raw.rsplit(') ',1)[1].split()[0]
  if state!='Z' and (name.lower()=='godot' or name.startswith('MicroDuck')):
   runtimes.append(dict(pid=int(stat.parent.name),name=name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
clean=json.loads((p/'final_validation_retry/candidate_clean/summary.json').read_text())
names={e['command'][e['command'].index('--name')+1] for e in clean['episodes']}
env=json.loads((p/'final_validation_retry/candidate_clean/container_environment.json').read_text());names.add(env['command'][env['command'].index('--name')+1])
running=set(subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True,timeout=15).splitlines());left=sorted(running&names);assert not left,left
archives={}
expected={'dist/MicroDuck-ARM64-20260911-candidate.tar.gz':'b175c0d59786e1da7fa05605c38767efd0050c49d6cbdd336695f8588943da9b','dist/MicroDuck-ARM64-20260911-incumbent.tar.gz':'d4a9cad342ba3c40f510c455bd14e0a0c446da2b02e7e26124ef080002ae5f07'}
for name,sha in expected.items():
 with Path(name).open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
 assert actual==sha,name
 archives[name]=dict(expected=sha,actual=actual,unchanged=True)
bank=json.loads((p/'final_validation_retry/baseline_models/bank.json').read_text())
models={}
for skill,model in bank['models'].items():
 source=Path(model['source']);actual=hashlib.sha256(source.read_bytes()).hexdigest();assert actual==model['sha256']
 models[skill]=dict(source=str(source),sha256=actual,unchanged=True)
result=dict(unix=time.time(),owned_process_groups=records,owned_remaining=[],live_godot_or_microduck_inventory=runtimes,owned_container_count=len(names),owned_containers_remaining=left,old_archives=archives,old_models=models,unrelated_applications_not_signaled=True)
(p/'closeout_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(groups_checked=len(records),owned_remaining=[],live_runtime_inventory=runtimes,owned_containers_remaining=[],old_archives_unchanged=True,old_models_unchanged=True)))
