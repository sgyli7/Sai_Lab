from pathlib import Path
import subprocess
r=Path('results/sprint_gpu_match_20260912')
for name,timeout in [('handoff_audit.py',90),('run_checks.py',300),('cleanup_prepared.py',120)]:
 subprocess.run(['.venv/bin/python',str(r/name)],check=True,timeout=timeout)
