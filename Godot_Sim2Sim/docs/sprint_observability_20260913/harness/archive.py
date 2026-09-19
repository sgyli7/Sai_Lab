from pathlib import Path
import gzip
import hashlib
import json
import shutil
from sim2sim.research.queue import atomic_json

ROOT=Path.cwd()
R=ROOT/'results/sprint_observability_20260913'
D=ROOT/'docs/sprint_observability_20260913'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def copy(p,d):
    d.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,d)


def main():
    for experiment in ('predictor','predictor_long'):
        for name in ('protocol.json','learning.json','analysis.json','preregistered_followup.json'):
            p=R/experiment/name
            if p.exists():copy(p,D/'evidence'/experiment/name)
    dataset=R/'predictor/dataset.json'
    with gzip.GzipFile(filename=str(D/'evidence/dataset.json.gz'),mode='wb',mtime=0) as f:f.write(dataset.read_bytes())
    d=json.loads(dataset.read_text())
    atomic_json(D/'evidence/dataset_summary.json',dict(splits=d['splits'],elapsed_s=d['elapsed_s'],files=d['files'],
        original_manifest_sha256=sha(dataset),episodes=len(d['episodes']),
        scope='Complete source episodes/models/checksums in dataset.json.gz'))
    for p in (R/'predictor/source').rglob('*.py'):
        copy(p,D/'harness/predictor_v1'/p.relative_to(R/'predictor/source'))
    for p in ('scripts/sprint_observability.py','src/sim2sim/research/observability.py','tests/test_observability.py'):
        copy(ROOT/p,D/'harness/predictor_v2'/p)
    for p in R.glob('*.py'):copy(p,D/'harness'/p.name)
    for p in ('reproduce/completed.json','targeted_analysis.json','gpu_resource_sample.json','test_protocol.json','session.json'):
        copy(R/p,D/'evidence'/p)
    for p in (R/'closeout').glob('*.json'):copy(p,D/'evidence/closeout'/p.name)
    for p in (R/'action_tape').glob('*.json'):copy(p,D/'evidence/action_tape'/p.name)
    for p in R.glob('*.png'):copy(p,D/p.name)
    for p in (R/'supervisors').glob('*.log'):copy(p,D/'evidence/supervisors'/p.name)
    # Large raw arrays/weights stay local. Reused source traces are indexed by
    # the original input manifest, not duplicated into this session.
    artifacts=[]
    for p in sorted(R.rglob('*')):
        if p.is_file() and (p.suffix in ('.npy','.npz','.pt','.onnx') or p.name=='trace.json'):
            artifacts.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(files=artifacts,count=len(artifacts),logical_bytes=sum(p['bytes'] for p in artifacts),
        scope='New local raw data, predictors and replays only. Shared previous-session inputs indexed separately; symlinked follow-up data not duplicated.')
    atomic_json(R/'closeout/local_artifacts.json',manifest)
    with gzip.GzipFile(filename=str(D/'evidence/local_artifacts.json.gz'),mode='wb',mtime=0) as f:
        f.write(json.dumps(manifest,indent=2).encode()+b'\n')
    print(json.dumps(dict(files=len(artifacts),logical_bytes=manifest['logical_bytes'])),flush=True)


if __name__=='__main__':main()
