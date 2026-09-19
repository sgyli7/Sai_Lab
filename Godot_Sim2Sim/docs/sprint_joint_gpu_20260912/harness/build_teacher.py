from pathlib import Path
import hashlib,json
import numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');out=r/'teacher';out.mkdir(exist_ok=False)
sha='27ebbf83d63e5e59f125ceb158414cf7d2574990676d73c75b9c9870bcefed6e';old=Path('results/sprint_joint_20260912/teacher/manifest.json');record=json.loads(old.read_text());assert record['anchor_sha256']==sha
source=old.parent/record['data'];assert hashlib.sha256(source.read_bytes()).hexdigest()==record['sha256']
with np.load(source,allow_pickle=False) as data:
 obs=data['obs'];frames=[obs[np.abs(obs[:,48])<.275].copy()]
sources=[dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),selection='Existing successful S05 teacher trajectories; only ordinary-speed/idle rows retained')]
summary_path=Path('results/sprint_joint_20260912/j01d_ordinary_regression/new/summary.json');summary=json.loads(summary_path.read_text());assert summary['errors']==0 and summary['models']['walking']==sha
for e in summary['episodes']:
 if not e['case'].startswith('walking_') or not e['task_metrics']['success']:continue
 path=Path(e['trace']);rows=json.loads(path.read_text())['rows'];x=np.array([row['obs'] for row in rows[::4] if row['skill']=='walking'],np.float32)
 if len(x):frames.append(x[np.abs(x[:,48])<.275])
 sources.append(dict(case=e['case'],seed=e['seed'],trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
x=np.concatenate(frames).astype(np.float32);assert x.shape[1]==61 and np.isfinite(x).all() and np.max(np.abs(x[:,48]))<.275
path=out/'observations.npz';np.savez_compressed(path,obs=x)
counts=dict(idle=int((np.linalg.norm(x[:,48:51],axis=1)<.01).sum()),forward=int((x[:,48]>.01).sum()),backward=int((x[:,48]<-.01).sum()),lateral=int((np.abs(x[:,49])>.01).sum()),turn=int((np.abs(x[:,50])>.25).sum()))
manifest=dict(version='walking_teacher_replay_v1',data=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),anchor_sha256=sha,frames=len(x),phase_counts=counts,sources=sources,selection='Only successful frozen S05 reference occupancy at ordinary commands; candidate on-policy recovery states are not teacher constrained')
atomic_json(out/'manifest.json',manifest);atomic_json(out/'completed.json',dict(completed=True,frames=len(x),phase_counts=counts));print(len(x),counts,flush=True)
