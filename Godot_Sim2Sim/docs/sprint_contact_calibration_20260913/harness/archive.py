"""Archive this bounded calibration without duplicating old native traces."""
from pathlib import Path
import gzip, hashlib, json, shutil, subprocess
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_contact_calibration_20260913')
d=Path('docs/sprint_contact_calibration_20260913')
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['contact_probe.py','parity_probe.py','parity_probe_strong.py','manifold_probe.py','compliance_probe.py','compliance_direct_probe.py','component_analysis.py','run_checks.py','audit_resources.py','archive.py']:
 copy(r/name,d/'harness'/name)
# Preserve each exact version, including both failed diagnostic attempts.
for name in ['contact_probe_attempt_01','contact_probe_attempt_02']:
 copy(r/name/'code.py',d/'harness'/name/'contact_probe.py')
for name in ['compliance_probe','compliance_direct_probe']:
 for p in (r/name/'source').rglob('*.py'):
  copy(p,d/'harness'/name/p.relative_to(r/name/'source'))
 protocol=json.loads((r/name/'protocol.json').read_text())
 for path,sha in protocol['code_sha256'].items():
  rel=Path(path).name if Path(path).is_absolute() else path
  assert digest(r/name/'source'/rel)==sha,(name,path)
base=subprocess.check_output(['git','show','60ea7d7f8e46b43bcc0384ab49bdf59ad99f8e06:scripts/sprint_gpu_world.py'])
b=d/'harness/contact_base/scripts/sprint_gpu_world.py';b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(base)
for p in r.rglob('*.json'):
 copy(p,d/'evidence'/p.relative_to(r))
for p in (r/'supervisors').glob('*.log'):
 dst=d/'evidence/supervisors'/(p.name+'.gz');dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
# Original arrays remain local. Manifest explicitly excludes the ledger/logs
# because they are separately archived byte-for-byte at seal.
artifacts=[]
for p in sorted(r.rglob('*.npz')):
 artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
atomic_json(d/'evidence/local_artifacts.json',dict(files=artifacts,count=len(artifacts),total_bytes=sum(x['bytes'] for x in artifacts),scope='New diagnostic arrays only; old native traces referenced by manifold_probe.json, never duplicated.'))
protected=json.loads(Path('results/sprint_observability_20260913/closeout/defaults_and_physics.json').read_text())
for item in protected['files']:
 item['actual']=digest(Path(item['path']));item['unchanged']=item['actual']==item['expected']
protected['passed']=all(x['unchanged'] for x in protected['files']);assert protected['passed']
atomic_json(r/'closeout/defaults_and_physics.json',protected)
copy(r/'closeout/defaults_and_physics.json',d/'evidence/closeout/defaults_and_physics.json')
source=json.loads((r/'manifold_probe.json').read_text())['sources']
for item in source:assert digest(Path(item['trace']))==item['sha256']
for name,script in [('contact_probe','contact_probe.py'),('contact_probe_attempt_01','contact_probe_attempt_01/code.py'),('contact_probe_attempt_02','contact_probe_attempt_02/code.py')]:
 p=json.loads((r/name/'protocol.json').read_text());assert digest(r/script)==p['code_sha256']
 assert digest(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json'))==p['source_sha256']
atomic_json(r/'closeout/source_validation.json',dict(passed=True,native_trace_hashes=len(source),exact_experiment_source_hashes=True,protected_files=len(protected['files']),local_arrays=len(artifacts),local_array_bytes=sum(x['bytes'] for x in artifacts)))
copy(r/'closeout/source_validation.json',d/'evidence/closeout/source_validation.json')
print(json.dumps(dict(archived=str(d),arrays=len(artifacts),bytes=sum(x['bytes'] for x in artifacts),protected_files=len(protected['files']))))
