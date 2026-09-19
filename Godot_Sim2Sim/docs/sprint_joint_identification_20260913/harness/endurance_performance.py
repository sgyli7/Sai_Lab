"""Frozen 30-minute arena course, then isolated-from-own-jobs latency/RSS run."""
from pathlib import Path
import json,os,subprocess,time
from sim2sim.standalone.benchmark import benchmark
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913');out=R/'endurance_performance';out.mkdir(exist_ok=False)
assert (R/'integrated_final/completed.json').is_file()
binary=Path(json.loads((R/'package_completed.json').read_text())['executable'])
ledger=json.loads((R/'active_budget.json').read_text())
started={e['actor'] for e in ledger['events'] if e['kind']=='job_started'};finished={e['actor'] for e in ledger['events'] if e['kind']=='job_finished'}
assert len(started-finished)==1,started-finished
with (out/'endurance.log').open('w') as log:
 process=subprocess.run(['.venv/bin/python','scripts/sprint_endurance.py','--executable',str(binary),'--out',str(out/'physical'),'--seconds','1800','--timeout','300'],stdout=log,stderr=subprocess.STDOUT,timeout=330)
assert process.returncode in [0,2],process.returncode
physical=json.loads((out/'physical/result.json').read_text())
(out/'processes_before_benchmark.txt').write_text(subprocess.check_output(['ps','-eo','pid,ppid,pgid,comm,pcpu','--sort=-pcpu'],text=True))
atomic_json(out/'host.json',dict(affinity=sorted(os.sched_getaffinity(0)),load=list(os.getloadavg()),unix=time.time(),scope='No concurrent jobs owned by this session. Other applications remain running; not an exclusive-host performance claim.'))
metrics=benchmark(binary,out/'performance',seconds=1800,replay=out/'physical/case.json',timeout=300)
assert metrics['completed'],metrics['reason']
atomic_json(out/'completed.json',dict(completed=True,physical_passed=physical['passed'],arena_valid=physical['arena_valid'],falls=physical['physical_metrics']['fell'],performance_completed=True,simulation_speed=metrics['simulation_speed'],peak_rss_kib=metrics['peak_rss_kib'],elapsed_s=metrics['elapsed_s']))
print(json.loads((out/'completed.json').read_text()),flush=True)
