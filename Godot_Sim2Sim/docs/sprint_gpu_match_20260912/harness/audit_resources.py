"""Read-only ownership audit after all supervised work has returned."""
from pathlib import Path
import json,os,subprocess,time
from sim2sim.research.queue import atomic_json
from sim2sim.research.budget import process_start_ticks
root=Path('results/sprint_gpu_match_20260912').resolve();budget=json.loads((root/'active_budget.json').read_text());events=budget['events']
starts=[e for e in events if e['kind']=='job_started'];launches=[e for e in events if e['kind']=='detached_supervisor_started']
known={e['pid']:e for e in starts}
supervisors=[e['supervisor_pid'] for e in launches if process_start_ticks(e['supervisor_pid'])==e['process_start_ticks']]
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  pid=int(p.name);pgid=os.getpgid(pid)
  if pgid not in known:continue
  rec=known[pgid];leader=process_start_ticks(pgid)
  if leader is not None and leader!=rec['process_start_ticks']:continue
  command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  stat=(p/'stat').read_text();state=stat.split(') ',1)[1].split()[0]
  if state!='Z':processes.append(dict(pid=pid,pgid=pgid,state=state,command=command))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
container_ids=subprocess.check_output(['docker','ps','--quiet'],text=True,timeout=10).split()
containers=[]
if container_ids:
 for item in json.loads(subprocess.check_output(['docker','inspect',*container_ids],text=True,timeout=15)):
  sources=[m.get('Source','') for m in item.get('Mounts',[])]
  if any(source==str(root) or source.startswith(str(root)+'/') for source in sources):
   containers.append(dict(id=item['Id'],name=item['Name'],mounts=sources))
complete_events=[e for e in events if e['kind']=='job_finished']
result=dict(created_unix=time.time(),owned_live_processes=processes,owned_live_supervisors=supervisors,owned_running_containers=containers,
 jobs_started=len(starts),jobs_finished=len(complete_events),actors=list(budget['actors']),passed=not(processes or supervisors or containers) and len(starts)==len(complete_events),
 note='Read-only audit using recorded process groups/start ticks and container bind-mount ownership. Unrelated workloads were not stopped.')
(root/'closeout').mkdir(exist_ok=True);atomic_json(root/'closeout/resources.json',result);print(result,flush=True)
assert result['passed']
