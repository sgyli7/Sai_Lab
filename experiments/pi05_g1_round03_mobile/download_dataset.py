"""Download only fixed public mobile LeRobot assets; CPU, size+SHA verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'.scratch/pi05_g1/round03_mobile/data'
REPO='nvidia/Arena-G1-Loco-Manipulation-Task'
REV='97f7b5a4135e4e6a206d394c9b6ec0253f1558a7'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/datasets/{REPO}/revision/{REV}?blobs=true'
    metadata=json.load(urllib.request.urlopen(url,timeout=60))
    if metadata['sha']!=REV:raise ValueError('Dataset revision changed')
    files=[f for f in metadata['siblings'] if f['rfilename'].startswith('lerobot/') or f['rfilename']=='README.md']
    manifest={'schema':'pi05_g1_mobile_pinned_download_manifest_v1','repo_id':REPO,'revision':REV,'api_url':url,'files':files}
    path=DEST/'remote_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Existing manifest differs')
    if not path.exists():path.write_text(json.dumps(manifest,indent=2)+'\n')
    def fetch(item):
        name=item['rfilename'];target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(4):
            try:
                if not target.exists():
                    temp=target.with_suffix(target.suffix+'.incomplete')
                    download=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}'
                    with urllib.request.urlopen(download,timeout=90) as source,temp.open('wb') as sink:
                        while block:=source.read(1024*1024):sink.write(block)
                    temp.rename(target)
                if target.stat().st_size!=item['size']:raise ValueError('Size mismatch: '+name)
                digest=sha(target)
                if item.get('lfs'):
                    if digest!=item['lfs']['sha256']:raise ValueError('LFS SHA mismatch: '+name)
                else:
                    blob=hashlib.sha1(b'blob '+str(item['size']).encode()+b'\0'+target.read_bytes()).hexdigest()
                    if blob!=item['blobId']:raise ValueError('Git blob mismatch: '+name)
                return {'path':name,'bytes':item['size'],'sha256':digest,'verified_lfs_or_git_blob':True}
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
    receipts=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,row in enumerate(pool.map(fetch,files),1):
            receipts.append(row)
            if i%20==0:print(json.dumps({'verified_files':i,'total_files':len(files)}),flush=True)
    receipt={'schema':'pi05_g1_mobile_verified_download_receipt_v1','status':'complete','repo_id':REPO,'revision':REV,'files':receipts,'verified_total_bytes':sum(f['bytes'] for f in receipts),'hdf5_downloaded':False}
    target=DEST/'download_receipt.json'
    if target.exists() and json.loads(target.read_text())!=receipt:raise ValueError('Prior download receipt differs')
    if not target.exists():target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'complete','files':len(receipts),'bytes':receipt['verified_total_bytes']}),flush=True)

if __name__=='__main__':main()
