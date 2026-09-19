from pathlib import Path
import os,json,time,subprocess
r=Path('results/sprint_observability_20260913');ledger=json.loads((r/'active_budget.json').read_text())
started={e['actor']:e for e in ledger['events'] if e['kind']=='job_started'}
finished={e['actor']:e for e in ledger['events'] if e['kind']=='job_finished'}
missing=sorted(set(started)-set(finished));cleanup={k:v['process_cleanup']['remaining'] for k,v in finished.items() if v['process_cleanup']['remaining']}
def proc(pid):
 try:
  p=Path('/proc')/str(pid);st=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,state=st[0],ppid=int(st[1]),pgrp=int(st[2]),session=int(st[3]),start_ticks=int(st[19]),comm=(p/'comm').read_text().strip(),args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:700])
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
ancestors=set();n=os.getpid()
while n and n not in ancestors:
 ancestors.add(n);p=proc(n);n=p['ppid'] if p else 0
owned_pids=[]
for e in ledger['events']:
 if e['kind'] not in ['job_started','detached_supervisor_started']:continue
 pid=e.get('pid',e.get('supervisor_pid'));p=proc(pid)
 if p and p['start_ticks']==e['process_start_ticks'] and p['state'] not in ('Z','X'):owned_pids.append(p)
groups={e['pid'] for e in started.values()};live_groups=[];marker=[];godots=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 p=proc(int(d.name))
 if not p or p['state'] in ('Z','X') or p['pid'] in ancestors:continue
 if p['pgrp'] in groups or p['session'] in groups:live_groups.append(p)
 try:owned_env=('SIM2SIM_ACTIVE_BUDGET_DIR='+str(r.resolve())).encode() in (d/'environ').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):owned_env=False
 if str(r) in p['args'] or owned_env:marker.append(p)
 if 'godot' in p['comm'].lower():godots.append(p)
def command(argv):
 try:
  s=subprocess.run(argv,capture_output=True,text=True,timeout=15)
  return dict(command=argv,returncode=s.returncode,stdout=s.stdout.strip(),stderr=s.stderr.strip())
 except (OSError,subprocess.TimeoutExpired) as e:return dict(command=argv,error=str(e))
docker=command(['docker','ps','--format','{{.ID}}\t{{.Names}}\t{{.Image}}'])
gpu=command(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'])
apps=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
v=dict(unix=time.time(),started=len(started),finished=len(finished),missing_terminal=missing,cleanup_remaining=cleanup,live_owned_pids=owned_pids,live_owned_groups=live_groups,live_session_marker_processes=marker,all_godot_processes=godots,docker=docker,gpu=gpu,gpu_apps=apps,job_results=[dict(actor=k,returncode=v['returncode'],reason=v['reason']) for k,v in finished.items()],passed=not(missing or cleanup or owned_pids or live_groups or marker),scope='Only this round ownership checked for cleanup. Other applications are inventoried, never killed.')
(r/'closeout/resources.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v[k] for k in ["started","finished","missing_terminal","cleanup_remaining","live_owned_pids","live_owned_groups","live_session_marker_processes","all_godot_processes","passed"]}));assert v['passed']
