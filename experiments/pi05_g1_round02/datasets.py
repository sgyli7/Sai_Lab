"""CPU adapters for original complete LeRobot episodes and fresh simulation teacher."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.pi05_g1.training import EpisodeDataset, decode_video_frame


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class TeacherEpisodes:
    def __init__(self, manifest, split, normalizer):
        path=Path(manifest).resolve();data=json.loads(path.read_text())
        if data.get('schema')!='pi05_g1_teacher_episodes_v1':raise ValueError('Unsupported teacher manifest schema')
        if data.get('fresh_source_enabled') is not True:raise ValueError('Teacher converter did not qualify a nonempty train+validation subset')
        train,val=list(map(str,data['train'])),list(map(str,data['validation']))
        if set(train)&set(val):raise ValueError('Whole teacher episodes overlap train and validation')
        if len(train)!=len(set(train)) or len(val)!=len(set(val)):raise ValueError('Duplicate teacher split IDs')
        records={str(record['episode_id']):record for record in data['episodes']}
        if len(records)!=len(data['episodes']):raise ValueError('Duplicate teacher records')
        ids=train if split=='train' else val
        if not ids:raise ValueError(f'No qualified fresh {split} episode')
        self.episodes=[];self.indices=[]
        for identifier in ids:
            record=records[identifier]
            resolve=lambda value:Path(value) if Path(value).is_absolute() else path.parent/value
            npz,video=resolve(record['npz']),resolve(record['video'])
            if sha256(npz)!=record['npz_sha256'] or sha256(video)!=record['video_sha256']:raise ValueError(f'Teacher payload changed: {identifier}')
            with np.load(npz,allow_pickle=False) as payload:
                states=np.asarray(payload['state'],np.float32);actions=np.asarray(payload['actions'],np.float32);times=np.asarray(payload['timestamps'],np.float64)
            n=int(record['length'])
            if states.shape!=(n,32) or actions.shape!=(n,32) or times.shape!=(n,) or not n:raise ValueError('Teacher array shape mismatch')
            if not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(times).all():raise ValueError('Nonfinite teacher data')
            if float(record['fps'])!=50 or not np.allclose(times,np.arange(n)/50,atol=1e-5,rtol=0):raise ValueError('Teacher must retain exact synchronous50Hz timing')
            if not np.all(states[:,-1]==0):raise ValueError('Teacher structural state padding must be zero')
            frames=record.get('raw_rgb_frames')
            if frames is not None:
                if len(frames)!=n:raise ValueError('Teacher raw RGB frames must match supervised length')
                frames=[{**entry,'path':resolve(entry['path'])} for entry in frames]
                shape=None
                for index,entry in enumerate(frames):
                    if entry['control_tick']!=index or abs(float(entry['timestamp_seconds'])-times[index])>1e-5:raise ValueError('Teacher raw RGB frame timing mismatch')
                    rgb=self.read_raw_rgb(entry)
                    if shape is not None and rgb.shape!=shape:raise ValueError('Teacher raw RGB dimensions changed within episode')
                    shape=rgb.shape
            self.episodes.append({'id':identifier,'states':states,'actions':normalizer.normalize(actions,'actions'),'timestamps':times,'video':video,'raw_rgb_frames':frames,'source_success_candidate_tick':record.get('source_success_candidate_tick')})
            self.indices.extend((len(self.episodes)-1,index) for index in range(n))
        self.normalizer=normalizer
        self.time_sampling={'kind':'uniform'};self.prefix_indices=None

    def configure_time_sampling(self,config):
        if config is None:config={'kind':'uniform'}
        if not isinstance(config,dict):raise ValueError('Explicit time_sampling object required')
        if config.get('kind')=='uniform':
            if set(config)!={'kind'}:raise ValueError('Unknown uniform time_sampling field')
            self.time_sampling={'kind':'uniform'};self.prefix_indices=None;return
        if config.get('kind')!='prefix_mixture' or set(config)!={'kind','prefix_probability','post_candidate_ticks','episode_choice'} or config['episode_choice']!='uniform':raise ValueError('Prefix time_sampling requires episode_choice=uniform')
        probability,post=config['prefix_probability'],config['post_candidate_ticks']
        if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0<=probability<=1:raise ValueError('Invalid prefix_probability')
        if isinstance(post,bool) or not isinstance(post,int) or post<0:raise ValueError('Invalid post_candidate_ticks')
        bounds=[]
        for episode in self.episodes:
            candidate=episode['source_success_candidate_tick']
            if isinstance(candidate,bool) or not isinstance(candidate,int) or not 0<=candidate<=300:raise ValueError('Prefix sampling requires integer source_success_candidate_tick in [0,300] for every episode')
            bounds.append(min(len(episode['actions'])-1,candidate+post))
        self.prefix_indices=[index for index,(episode,frame) in enumerate(self.indices) if frame<=bounds[episode]]
        self.episode_offsets=np.cumsum([0]+[len(episode['actions']) for episode in self.episodes[:-1]]).tolist()
        self.prefix_bounds=bounds
        self.time_sampling={**config,'prefix_end_tick_inclusive':dict(zip((episode['id'] for episode in self.episodes),bounds)),'prefix_frame_count':len(self.prefix_indices),'sampling_unit':'uniform whole episode first; then mixture branch and uniform eligible frame; validation remains complete'}

    def draw_index(self,rng):
        if self.prefix_indices is not None:
            episode=rng.randrange(len(self.episodes))
            prefix=rng.random()<self.time_sampling['prefix_probability']
            length=self.prefix_bounds[episode]+1 if prefix else len(self.episodes[episode]['actions'])
            return self.episode_offsets[episode]+rng.randrange(length),'prefix' if prefix else 'full'
        return rng.randrange(len(self.indices)),'full'

    @staticmethod
    def read_raw_rgb(entry):
        from PIL import Image
        if sha256(entry['path'])!=entry['byte_sha256']:raise ValueError('Teacher raw RGB byte hash mismatch')
        with Image.open(entry['path']) as image:
            if image.format!='PNG' or image.mode!='RGB':raise ValueError('Teacher raw RGB requires lossless RGB PNG')
            rgb=np.asarray(image).copy()
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:raise ValueError('Teacher raw RGB shape mismatch')
        if 'rgb_shape' in entry and list(rgb.shape)!=entry['rgb_shape']:raise ValueError('Teacher raw RGB recorded shape mismatch')
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=entry['rgb_sha256']:raise ValueError('Teacher decoded RGB hash mismatch')
        return rgb

    def sample(self,index):
        episode_index,frame=self.indices[index];episode=self.episodes[episode_index]
        stop=min(frame+50,len(episode['actions']));count=stop-frame
        actions=np.repeat(episode['actions'][stop-1:stop],50,axis=0)
        actions[:count]=episode['actions'][frame:stop];mask=np.arange(50)<count
        rgb=self.read_raw_rgb(episode['raw_rgb_frames'][frame]) if episode['raw_rgb_frames'] is not None else decode_video_frame(episode['video'],episode['timestamps'][frame])
        return rgb,episode['states'][frame],actions,mask
    validation_indices=EpisodeDataset.validation_indices


def build_sources(config_path, normalizer):
    path=Path(config_path).resolve();config=json.loads(path.read_text())
    records=config['sources']
    if not records or len({item['name'] for item in records})!=len(records):raise ValueError('Named unique data sources required')
    sources=[]
    for item in records:
        for key in ('train_sampling_weight','validation_weight'):
            if not isinstance(item.get(key),(int,float)) or not math.isfinite(item[key]) or item[key]<0:raise ValueError(f'Explicit finite nonnegative {key} required')
        resolve=lambda value:Path(value).resolve() if Path(value).is_absolute() else (path.parent/value).resolve()
        manifest=resolve(item['manifest'])
        if item['kind']=='lerobot':
            if item.get('time_sampling',{'kind':'uniform'})!={'kind':'uniform'}:raise ValueError('Original source must retain uniform time sampling')
            root=resolve(item['dataset_root'])
            train=EpisodeDataset(root,manifest,'train',normalizer);validation=EpisodeDataset(root,manifest,'validation',normalizer)
            if len(validation.episodes)!=21:raise ValueError('The original21 validation episodes must be retained')
        elif item['kind']=='npz_video':
            train=TeacherEpisodes(manifest,'train',normalizer);validation=TeacherEpisodes(manifest,'validation',normalizer)
            train.configure_time_sampling(item.get('time_sampling'))
        else:raise ValueError('Supported source kinds: lerobot,npz_video')
        sources.append({'name':item['name'],'kind':item['kind'],'train':train,'validation':validation,'train_sampling_weight':float(item['train_sampling_weight']),'validation_weight':float(item['validation_weight']),'time_sampling':train.time_sampling if isinstance(train,TeacherEpisodes) else {'kind':'uniform'},'manifest':str(manifest),'manifest_sha256':sha256(manifest),'train_episodes':len(train.episodes),'validation_episodes':len(validation.episodes),'train_frames':len(train.indices),'validation_chunks':len(validation.validation_indices())})
    if not any(item['kind']=='npz_video' for item in sources) or not any(item['kind']=='lerobot' for item in sources):raise ValueError('Round02 requires qualified fresh teacher and original sources')
    if sum(item['train_sampling_weight'] for item in sources)<=0 or sum(item['validation_weight'] for item in sources)<=0:raise ValueError('Positive aggregate training and validation weights required')
    return sources
