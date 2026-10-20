"""Verify the self-contained release file manifest using only the standard library."""
from pathlib import Path
import argparse,hashlib,json


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text());errors=[]
    for row in manifest['files']:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root)or not p.is_file():errors.append('Missing/nonlocal:'+row['path']);continue
        if p.stat().st_size!=row['bytes']or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:errors.append('Changed:'+row['path'])
    return dict(release_id=manifest['release_id'],verified_files=len(manifest['files']),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),passed=not errors,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package',nargs='?',type=Path,default=Path(__file__).parent);a=p.parse_args();r=verify(a.package);print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed']else 1)
