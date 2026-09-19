from pathlib import Path
import hashlib,json,shutil
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912').resolve()
assert json.loads((r/'complete_courses_completed.json').read_text())['completed']
def fingerprint(root):
 result={}
 for p in sorted(root.rglob('*')):
  if p.is_symlink():result[str(p.relative_to(root))]=('link',str(p.readlink()))
  elif p.is_file():result[str(p.relative_to(root))]=('file',hashlib.sha256(p.read_bytes()).hexdigest())
 return result
records=[]
for name in ['baseline_s05','baseline_cat_split','frozen_cat_joint','native_split','native_shared','native_tracking','native_tracking_split','feedback_disabled_canary','native_feedback','native_trained_tracking_split','native_complete_courses']:
 prepared=r/name/'prepared';canonical=r/name/'suite/runtime'
 assert prepared.is_relative_to(r) and canonical.is_relative_to(r)
 assert (r/name/'completed.json').exists() and prepared.is_dir() and canonical.is_dir()
 same=fingerprint(prepared)==fingerprint(canonical)
 record=dict(prepared=str(prepared),canonical=str(canonical),identical=same,bytes=sum(p.stat().st_size for p in prepared.rglob('*') if p.is_file() and not p.is_symlink()))
 if not same:raise RuntimeError('Preserving nonidentical '+str(prepared))
 shutil.rmtree(prepared);record['deleted']=True;record['restore']=['cp','--reflink=auto','-a',str(canonical),str(prepared)]
 records.append(record);atomic_json(r/'closeout/prepared_cleanup.json',records)
print('Removed',len(records),'byte-identical staging copies; bytes',sum(x['bytes'] for x in records),flush=True)
