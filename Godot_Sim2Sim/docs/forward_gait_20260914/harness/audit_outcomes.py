from pathlib import Path
import collections,hashlib,json
import numpy as np
from sim2sim.research.models import NativeAnchor
R=Path(__file__).resolve().parent
I=Path('/home/ethan/Projects/MicroDuck-SpeedControls')
ordinary=NativeAnchor(I/'src/sim2sim/assets/microduck_sprint_v1/Walk_Godot.onnx')
results={}
for label in ['default030','a402','off_952410','off_952411','support_952410','support_952411']:
 model=I/'src/sim2sim/assets/microduck_sprint_v1/Sprint_Godot.onnx' if label in ['default030','a402'] else R/label/'final.onnx'
 sprint=NativeAnchor(model);s=json.loads((R/('native_'+label)/'suite/summary.json').read_text());samples={'walking':[],'sprint':[]};failure_types=collections.Counter();failed_cases=collections.Counter();falls=[];first=[]
 for episode in s['episodes']:
  m=episode['task_metrics']
  if not m['success']:failed_cases[episode['case']]+=1
  for gate in ['straight','turns','stops']:
   if any(not x['passed'] for x in m[gate]):failure_types[gate]+=1
  if m['fell']:falls.append(dict(case=episode['case'],seed=episode['seed'],first_fall_s=m['first_fall_s']))
  rows=json.loads(Path(episode['trace']).read_text())['rows'];first.append((episode['case'],episode['seed'],rows[0]['obs']))
  for key in samples:
   selected=[r for r in rows if r['skill']==key]
   for i in np.unique(np.linspace(0,len(selected)-1,min(16,len(selected)),dtype=int)) if selected else []:samples[key].append(selected[i])
 parity={}
 for skill,anchor in [('walking',ordinary),('sprint',sprint)]:
  x=np.array([r['obs'] for r in samples[skill]],np.float32);actual=np.array([r['action'] for r in samples[skill]],np.float32)
  error=float(np.abs(anchor(x)-actual).max());parity[skill]=dict(rows=len(x),max_abs=error,passed=error<1e-5)
  assert error<1e-5
 gpu=R/('gpu_'+label)/'completed.json';initial=None
 if gpu.exists():
  g=json.loads(gpu.read_text());indices={(x['case'],x['seed']):j for j,x in enumerate(g['results'])}
  with np.load(gpu.parent/'trajectories.npz') as data:obs=data['obs'][0]
  initial=max(float(np.abs(np.array(row,np.float32)-obs[indices[(case,seed)]]).max()) for case,seed,row in first)
 results[label]=dict(real_action_parity=parity,initial_gpu_native_obs_max_abs=initial,failed_cases=dict(failed_cases),failure_types=dict(failure_types),falls=falls,actor_sha256=sprint.sha256,ordinary_sha256=ordinary.sha256)
result=dict(passed=True,results=results,scope='Real native observation/action samples: at most 16 equally spaced rows per skill per episode, after casting both actions to float32. Gate failure counts overlap and include post-fall failures; repeated-prefix falls are not independent failure modes. Initial actor observation comparison does not establish full solver/contact-state equivalence.')
(R/'outcome_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{x:v[x] for x in ['real_action_parity','initial_gpu_native_obs_max_abs','failure_types']} for k,v in results.items()},indent=2))
