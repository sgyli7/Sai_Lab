"""Shared frozen episode split; LingBot mean/std fitted only on training rows."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.pi05_g1.training import EpisodeDataset
from .contract import manifest as contract_manifest

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT/'.scratch/pi05_g1/round03_mobile'
SPLIT=SHARED/'prepared/split_manifest.json'
SPLIT_SHA='b5333c4e97c497518d50d637ab3f9234f1ded9c9ca3afb6a13e9e3de519b30f3'
OWN=ROOT/'.scratch/lingbot_vla2_g1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Identity:
    def normalize(self,values,kind):return np.asarray(values,dtype=np.float32)


class MeanStd:
    def __init__(self,path):
        self.doc=json.loads(Path(path).read_text())
        for kind in ['state','actions']:
            stat=self.doc[kind];mean=np.asarray(stat['mean'],np.float32)
            scale=np.asarray(stat.get('scale',stat.get('std')),np.float32)
            if mean.shape!=(32,) or scale.shape!=(32,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<0):raise ValueError('Finite32 mean/std required')
            stat['scale']=np.where(scale>1e-6,scale,1.).tolist()
    def normalize(self,values,kind):
        stat=self.doc[kind];return ((np.asarray(values,dtype=np.float32)-np.asarray(stat['mean'],np.float32))/np.asarray(stat['scale'],np.float32)).astype(np.float32)
    def denormalize(self,values,kind):
        stat=self.doc[kind];return (np.asarray(values,dtype=np.float32)*np.asarray(stat['scale'],np.float32)+np.asarray(stat['mean'],np.float32)).astype(np.float32)


def prepare():
    if sha(SPLIT)!=SPLIT_SHA:raise ValueError('Shared split changed')
    split=json.loads(SPLIT.read_text())
    if [len(split[k]) for k in ['train','validation','test']]!=[80,10,10]:raise ValueError('Whole-episode split mismatch')
    dataset=EpisodeDataset(SHARED/'data/lerobot',SPLIT,'train',Identity())
    result={'schema':'lingbot_vla2_g1_train_meanstd_v1','split_manifest_sha256':SPLIT_SHA,'train_episode_ids':split['train'],'fit_splits':['train'],'test_opened':False,'normalization':'per-dimension train-frame mean/std; constant scale=1; no quantile clipping','contract':contract_manifest()}
    for target,field in [('state','states'),('actions','actions')]:
        values=np.concatenate([ep[field] for ep in dataset.episodes]).astype(np.float64)
        mean=values.mean(axis=0);std=values.std(axis=0)
        result[target]={'mean':mean.tolist(),'scale':np.where(std>1e-6,std,1.).tolist(),'raw_std':std.tolist(),'frames':len(values),'min':values.min(axis=0).tolist(),'max':values.max(axis=0).tolist()}
    OWN.mkdir(exist_ok=True);path=OWN/'train_meanstd.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Refusing to overwrite changed normalization')
    path.write_text(json.dumps(result,indent=2)+'\n')
    (OWN/'contract.json').write_text(json.dumps(contract_manifest(),indent=2)+'\n')
    print(json.dumps({'event':'data_prepared','train_episodes':len(dataset.episodes),'train_frames':result['state']['frames'],'split_sha256':SPLIT_SHA,'test_opened':False}),flush=True)


if __name__=='__main__':prepare()
