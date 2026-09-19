"""Archive the exact tested app, with explicit rejection status outside its manifest."""
import hashlib,io,json,tarfile
from pathlib import Path
p=Path('results/jolt_learning_20260911');r=json.loads((p/'final_validation_complete.json').read_text())
assert json.loads((p/'frozen_runtime_benchmark/metrics.json').read_text())['completed']
package=Path(r['packages']['candidate']['directory'])
for line in (package/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split('  ',1)
 with (package/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,name
out=Path('dist/MicroDuck-ARM64-20260912-learning-trial.tar.gz')
assert not out.exists()
temporary=out.with_suffix('.gz.partial');top='MicroDuck-ARM64-20260912-learning-trial'
status=b'UNPROMOTED RESEARCH CANDIDATE - NOT A PASSED RELEASE\nFinal active braking: old 183/210, candidate 183/210; five old successes lost.\nOriginal models and packages were retained. Runtime checks do not establish task quality.\nRun app/MicroDuck.arm64. App files exactly match the tested package manifest.\n'
with tarfile.open(temporary,'w:gz',compresslevel=3) as archive:
 archive.add(package,arcname=top+'/app')
 info=tarfile.TarInfo(top+'/ACCEPTANCE.txt');info.size=len(status);archive.addfile(info,io.BytesIO(status))
 archive.add(p/'final_analysis.json',arcname=top+'/FINAL_RESULTS.json')
temporary.rename(out)
with out.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
record=dict(path=str(out.resolve()),sha256=sha,bytes=out.stat().st_size,status='unpromoted; final paired quality gate failed',models=r['packages']['candidate']['models'])
(p/'trial_archive.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
