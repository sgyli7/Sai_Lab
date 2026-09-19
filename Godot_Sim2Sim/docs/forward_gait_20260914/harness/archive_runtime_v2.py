from pathlib import Path
import hashlib,json,shutil,tarfile,time
from sim2sim.standalone.suite import runtime_inputs
R=Path(__file__).resolve().parent
base=R/'runtime';preflight=json.loads((R/'runtime_hashes.json').read_text());original=runtime_inputs(base)
assert all(original.get(k)==v for k,v in preflight.items())
# Preflight hashed scripts/scenes/JSON only; suite also hashes binaries/assets.

archive=R/'baseline_runtime.tar.gz';assert not archive.exists()
files=[p for p in sorted(base.rglob('*')) if p.is_file() and '.godot' not in p.relative_to(base).parts]
full={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  assert member.isfile()
  assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==full.pop(member.name)
assert not full
records=[]
for d in sorted(R.glob('native_*')):
 if not d.is_dir():continue
 prepared=d/'prepared';runtime=d/'suite/runtime'
 if not runtime.exists():continue
 record=json.loads((d/'suite/runtime_snapshot.json').read_text())['input_sha256']
 assert runtime_inputs(runtime)==record and runtime_inputs(prepared)==record
 assert not set(original)-set(record)
 delta={k:v for k,v in record.items() if original.get(k)!=v}
 overlay=d/'runtime_overlay';overlay.mkdir(exist_ok=False)
 for key in delta:
  dest=overlay/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(runtime/key,dest)
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==delta[key]
 records.append(dict(label=d.name,overlay=str(overlay.relative_to(R)),changed_inputs=delta,removed=[str(prepared),str(runtime)]))
 for target in [prepared,runtime]:
  assert target.resolve().is_relative_to(R) and not target.is_symlink();shutil.rmtree(target)
shutil.rmtree(base)
result=dict(passed=True,unix=time.time(),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,source_inputs=original,overlays=records,removed_runtime_directories=1+2*len(records),reconstruction='Extract baseline archive into NEW owned directory, then copy the selected runtime_overlay over it. Verify suite/runtime_snapshot.json input_sha256 using runtime_inputs. .godot cache excluded; reimport if required. Raw traces/checkpoints/models retained.')
(R/'closeout/runtime_archive.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ['passed','archive_bytes','removed_runtime_directories']},flush=True)
