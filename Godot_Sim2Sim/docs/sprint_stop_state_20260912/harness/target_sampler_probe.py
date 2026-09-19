"""Bounded exploratory Jolt rollout throughput probe; no policy update."""
from pathlib import Path
import concurrent.futures,json,subprocess,shutil,time,hashlib
from sim2sim.standalone.sprint import write_cases
from sim2sim.research.queue import atomic_json
from sim2sim.godot_proc import _headless_overlay
r=Path('results/sprint_stop_state_20260912').resolve();out=r/'target_sampler_retry';out.mkdir(exist_ok=False);project=out/'runtime'
subprocess.run(['cp','--reflink=auto','-a',str(r/'native_joint_fd/suite/runtime'),str(project)],check=True,timeout=60)
f=project/'standalone/driver.gd';s=f.read_text();s=s.replace('var forced_skill := ""','var forced_skill := ""\nvar sample_std := 0.02')
s=s.replace('  ', '  ')
s=s.replace('\tseed(int(session.seed))','\tfor arg in OS.get_cmdline_user_args():\n\t\tif arg.begins_with("--sample-std="): sample_std=float(arg.trim_prefix("--sample-std="))\n\tif sample_std not in [0.0,0.02]:\n\t\t_fatal("Unsupported training exploration")\n\t\treturn\n\tseed(int(session.seed))')
s=s.replace('\tready_to_run = true','\tbank["sprint"] = TrainingPolicy.new(bank["sprint"],sample_std)\n\tready_to_run = true',1)
s=s.replace('var result := {"schema_version":1,"mode":session.mode', 'var result := {"training_exploration":sample_std,"schema_version":1,"mode":session.mode')
s+='''
# [DEBUG-target-sampler] Research-only stochastic rollout, never an acceptance.
class TrainingPolicy:
 extends RefCounted
 var inner
 var sigma: float
 func _init(policy, standard_deviation: float):
  inner=policy
  sigma=standard_deviation
 func infer(obs):
  var action=inner.infer(obs)
  if sigma>0.0:
   for i in range(action.size()): action[i]+=randfn(0.0,sigma)
  return action
 func get_last_error(): return inner.get_last_error()
 func get_last_infer_usec(): return inner.get_last_infer_usec()
 func unload(): inner.unload()
'''
s='\n'.join(('\t'*(len(line)-len(line.lstrip(' ')))+line.lstrip(' ')) if line.startswith(' ') else line for line in s.splitlines())+'\n'
f.write_text(s);control=json.loads((r/'native_joint_fd/control.json').read_text());cases=write_cases(out/'cases',range(954100,954102),.3,paired=False,control=control)
for case in cases:
 x=json.loads(Path(case).read_text());x['training_exploration']=True;atomic_json(case,x)
def run(item):
 i,case,sigma=item;case=Path(case).resolve();folder=out/f'rollout_{i:02d}';folder.mkdir();overlay=_headless_overlay(project);started=time.monotonic();command=['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(case),'--trace='+str(folder/'trace.json'),'--sample-std='+str(sigma),'--seed='+str(9541000+i)]
 try:
  with (folder/'player.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=15)
 finally:shutil.rmtree(overlay)
 data=json.loads((folder/'trace.json').read_text());assert not data['summary']['error'];assert data['summary']['training_exploration']==sigma
 x=dict(case=str(case),trace=str(folder/'trace.json'),steps=len(data['rows']),seconds=time.monotonic()-started,sigma=sigma,noise_seed=9541000+i,first_fall=data['summary']['first_fall']);atomic_json(folder/'completed.json',x);return x
items=[(i,c,.02) for i,c in enumerate(cases)]+[(16+i,c,0.) for i,c in enumerate(cases[:2])]
started=time.monotonic()
canary=run(items[-2])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=[canary,*list(pool.map(run,[*items[:16],items[-1]]))]
elapsed=time.monotonic()-started
atomic_json(out/'completed.json',dict(completed=True,rows=rows,elapsed_s=elapsed,policy_steps=sum(x['steps'] for x in rows),physics_steps=4*sum(x['steps'] for x in rows),policy_steps_per_wall_second=sum(x['steps'] for x in rows)/elapsed,workers=4,training_updates=0,accepted=False,scope='Exploratory target-engine data, not gameplay acceptance; all episodes replay complete history from reset, no state transplantation',instrumentation_sha256=hashlib.sha256(s.encode()).hexdigest()))
print(elapsed,sum(x['steps'] for x in rows)/elapsed,flush=True)
