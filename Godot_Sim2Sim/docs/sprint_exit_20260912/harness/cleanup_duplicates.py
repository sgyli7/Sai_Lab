"""Remove only byte-identical, completed trial staging copies; retain canonical runtime."""
from pathlib import Path
import json,shutil,subprocess,time
from sim2sim.research.queue import atomic_json
root=Path('results/sprint_exit_20260912').resolve();records=[]
for prepared in sorted(root.glob('*/prepared')):
 trial=prepared.parent;canonical=trial/'suite/runtime'
 if not (trial/'completed.json').is_file() or not canonical.is_dir():continue
 if prepared.is_symlink() or canonical.is_symlink():raise RuntimeError('Unexpected symlink')
 assert prepared.resolve().is_relative_to(root) and canonical.resolve().is_relative_to(root)
 result=subprocess.run(['diff','-qr','--no-dereference',str(prepared),str(canonical)],capture_output=True,text=True,timeout=45)
 item=dict(prepared=str(prepared),canonical=str(canonical),identical=result.returncode==0,removed=False,differences=result.stdout[:1000],error=result.stderr[:1000])
 if item['identical']:
  item.update(bytes=sum(p.lstat().st_size for p in prepared.rglob('*') if p.is_file() and not p.is_symlink()),removed_unix=time.time(),reason='Identical staging copy; frozen runtime and raw trajectories retained',rehydrate='cp -a '+str(canonical)+' '+str(prepared))
  shutil.rmtree(prepared);item['removed']=True
  atomic_json(trial/'prepared_copy_removed.json',item)
 records.append(item);atomic_json(root/'prepared_duplicate_cleanup.json',dict(records=records,removed=sum(r['removed'] for r in records),bytes=sum(r.get('bytes',0) for r in records)))
print(dict(removed=sum(r['removed'] for r in records),bytes=sum(r.get('bytes',0) for r in records)),flush=True)
