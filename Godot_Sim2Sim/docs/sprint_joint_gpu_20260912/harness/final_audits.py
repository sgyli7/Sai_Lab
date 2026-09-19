from pathlib import Path
import hashlib,json,subprocess
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912')
for arm in ['native_trained_tracking_split','native_complete_courses']:
 subprocess.run(['.venv/bin/python',str(r/'handoff_audit.py'),arm,'--shadow'],check=True,timeout=90)
old=json.loads(Path('results/sprint_gpu_match_20260912/closeout/defaults_and_physics.json').read_text());records=[]
for row in old['files']:
 sha=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest();records.append(dict(path=row['path'],expected=row['expected'],actual=sha,unchanged=sha==row['expected']))
assert all(row['unchanged'] for row in records)
atomic_json(r/'closeout/defaults_and_physics.json',dict(passed=True,files=records))
subprocess.run(['.venv/bin/python',str(r/'cleanup_prepared.py')],check=True,timeout=120)
atomic_json(r/'closeout/audits_completed.json',dict(completed=True))
