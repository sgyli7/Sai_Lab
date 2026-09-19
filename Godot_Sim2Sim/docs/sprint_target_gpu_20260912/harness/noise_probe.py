from pathlib import Path
import concurrent.futures,json,shutil,subprocess,time,hashlib
import numpy as np
from sim2sim.research.native_sprint import prepare_sampler
from sim2sim.standalone.sprint import write_cases
from sim2sim.godot_proc import _headless_overlay
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_target_gpu_20260912').resolve();prior=Path('results/sprint_stop_state_20260912').resolve();out=r/'noise_probe';out.mkdir(exist_ok=False)
project=prepare_sampler(prior/'native_joint_fd/suite/runtime',out/'runtime',prior/'runtime_patches/target_sampler_retry/runtime/standalone/driver.gd')
p=project/'standalone/driver.gd';s=p.read_text();assert s.count('sample_std not in [0.0,0.02]')==1;s=s.replace('sample_std not in [0.0,0.02]','sample_std not in [0.0,0.005,0.02]');p.write_text(s)
deployment=json.loads((project/'runtime_assets/deployment.json').read_text());cases=write_cases(out/'cases',[927001,927007,927008],.3,selected=['alternate'],control=deployment['control_config'])
for case in cases:
 x=json.loads(case.read_text());x['training_exploration']=True;atomic_json(case,x)
items=[(case,sigma,95600000+n) for case in cases for sigma,count in [(0.,2),(.005,4),(.02,4)] for n in range(count)]
def run(item):
 i,(case,sigma,noise_seed)=item;folder=out/f'episode_{i:03d}';folder.mkdir();overlay=_headless_overlay(project)
 try:
  with (folder/'player.log').open('w') as log:subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(case),'--trace='+str(folder/'trace.json'),'--sample-std='+str(sigma),'--seed='+str(noise_seed)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=25)
 finally:shutil.rmtree(overlay)
 data=json.loads((folder/'trace.json').read_text());assert not data['summary']['error'];assert data['summary']['training_exploration']==sigma
 rows=data['rows'];times=np.array([x['t'] for x in rows]);q=np.array([x['body']['base_quat'] for x in rows]);yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)));ix=np.flatnonzero((times>=7.-1e-9)&(times<9.-1e-9));command=float(np.mean([x['requested_command'][2] for x in rows if 7.-1e-9<=x['t']<9.-1e-9]));rate=float((yaw[ix[-1]]-yaw[ix[0]])/(times[ix[-1]]-times[ix[0]])*np.sign(command))
 return dict(seed=json.loads(case.read_text())['seed'],sigma=sigma,noise_seed=noise_seed,signed_yaw_rate=rate,below_turn_threshold=rate<.6*abs(command),first_fall=data['summary']['first_fall'],trace=str(folder/'trace.json'),case=str(case))
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,enumerate(items)))
summary=[]
for seed in [927001,927007,927008]:
 for sigma in [0.,.005,.02]:
  selected=[x for x in rows if x['seed']==seed and x['sigma']==sigma];summary.append(dict(seed=seed,sigma=sigma,count=len(selected),below_threshold=sum(x['below_turn_threshold'] for x in selected),mean=float(np.mean([x['signed_yaw_rate'] for x in selected])),minimum=min(x['signed_yaw_rate'] for x in selected),maximum=max(x['signed_yaw_rate'] for x in selected)))
atomic_json(out/'completed.json',dict(completed=True,scope='Frozen initial actor, noise-state-distribution diagnostic only; no policy updates or acceptance',actor_sha256=deployment['policies']['sprint']['sha256'],rows=rows,summary=summary,elapsed_s=time.monotonic()-start))
print(summary,flush=True)
