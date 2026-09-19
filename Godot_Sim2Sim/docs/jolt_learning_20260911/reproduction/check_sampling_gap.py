"""Diagnose mean deployment versus frozen training exploration, never promote noise."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS
from sim2sim.research.world import World
from sim2sim.research.evaluate import record
from sim2sim.standalone.cases import standard_cases
from sim2sim.standalone.score import brake_metrics
p=Path('results/jolt_learning_20260911');cases=['roller_brake_1s','roller_brake_3s','roller_brake_5s'];models=['r1_full_s71_1m','r2_sagittal_s71_1m']
recipe=dict(hypothesis='Does contact-sensitive control rely on training exploration that is absent from mean-action deployment?',models=models,cases=cases,initial_seeds=[917000,917001],noise_seeds=[None,91,92],note='Same mean policy; stochastic arm uses checkpoint std and actual action mask. Diagnostic only, not a candidate or deployment noise proposal.')
(p/'sampling_gap_recipe.json').write_text(json.dumps(recipe,indent=2))
def run(job):
 name,case_name,seed,noise_seed=job;case=standard_cases()[case_name];actor=NativeAnchor(p/'runs'/name/'final.onnx')
 ck=torch.load(p/'runs'/name/'latest.pt',weights_only=False);std=ck['policy']['log_std'].exp().numpy();mask=np.ones(14,np.float32)
 if 'sagittal' in name:mask[:]=0;mask[[2,3,11,12]]=1
 w=World(replace(TASKS['roller'],seconds=case['seconds']),roller_contract=True,state_input=actor.state_input,task_input=actor.task_input)
 rng=np.random.default_rng(seed+(noise_seed or 0)*10000000+cases.index(case_name)*1000000);rows=[]
 try:
  obs=w.reset(seed,'keyboard_'+case_name)
  for i in range(round(case['seconds']/.02)):
   action=actor(obs[None])[0]
   if noise_seed is not None:action=(action+rng.normal(size=14)*(std*np.clip(-obs[48]/.05,0,1)*mask+1e-5)).astype(np.float32)
   w.executed_command=w.command();rows.append(record(w,action));obs=w.step(action)
  metrics=brake_metrics(rows,case['brake_times'][0],case['seconds'])
  result=dict(model=name,case=case_name,seed=seed,noise_seed=noise_seed,metrics=metrics)
  d=p/'sampling_gap_traces';d.mkdir(exist_ok=True);path=d/f'{name}_{case_name}_{seed}_{noise_seed}.npz'
  np.savez_compressed(path,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]});result['trace']=str(path)
  print(json.dumps(result),flush=True);return result
 finally:w.close()
jobs=[(model,case,seed,noise) for model in models for case in cases for seed in [917000,917001] for noise in [None,91,92]]
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
(p/'sampling_gap_complete.json').write_text(json.dumps(results,indent=2))
