from pathlib import Path
import hashlib,json,shutil,tarfile,time
from sim2sim.standalone.suite import runtime_inputs
R=Path(__file__).resolve().parent
base=R/'base';baseline=json.loads((R/'base_inputs.json').read_text());assert runtime_inputs(base)==baseline
archive=R/'baseline_runtime.tar.gz';assert not archive.exists()
paths=[p for p in sorted(base.rglob('*')) if p.is_file() and '.godot' not in p.relative_to(base).parts]
full={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
 for p in paths:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==full.pop(member.name)
assert not full
preserved=[];remove=[]
for label in ['neck_0','neck_5','neck_10']:
 p=R/label;record=json.loads(p.with_suffix('.json').read_text());assert runtime_inputs(p)==record['candidate_inputs']
 target=R/'frozen_drivers'/label/'standalone/driver.gd';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p/'standalone/driver.gd',target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==record['driver_sha256']
 preserved.append(str(target));remove.append(p)
for marker in sorted(R.glob('**/runtime_snapshot.json')):
 p=marker.parent/'runtime';record=json.loads(marker.read_text());assert runtime_inputs(p)==record['input_sha256']
 other={k:v for k,v in record['input_sha256'].items() if k!='standalone/driver.gd'}
 assert other=={k:v for k,v in baseline.items() if k!='standalone/driver.gd'}
 assert record['input_sha256']['standalone/driver.gd'] in [baseline['standalone/driver.gd']]+[json.loads((R/(label+'.json')).read_text())['driver_sha256'] for label in ['neck_0','neck_5','neck_10']]
 remove.append(p)
remove.append(base)
for p in remove:
 assert p.resolve().is_relative_to(R) and not p.is_symlink();shutil.rmtree(p)
result=dict(passed=True,unix=time.time(),removed=[str(x) for x in remove],count=len(remove),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,drivers=preserved,scope='Full base excluding .godot cache plus per-angle driver; exact input fingerprints preserved. Raw traces/metrics/protocols retained. Desktop runtime and other tasks untouched.')
(R/'closeout').mkdir(exist_ok=True);(R/'closeout/runtime_archive.json').write_text(json.dumps(result,indent=2)+'\n');print('ARCHIVED',result['count'],result['archive_bytes'])
