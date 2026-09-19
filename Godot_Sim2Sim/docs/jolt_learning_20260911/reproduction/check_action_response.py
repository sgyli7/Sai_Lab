"""Paired real-Jolt local response probe; no reset after the physical prefix."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS
from sim2sim.research.models import NativeAnchor
p=Path('results/jolt_learning_20260911');source=Path('results/research_20260911/delivery/candidate_models/Roller_Godot.onnx')
recipe=dict(hypothesis='Out-of-range brake targets may leave small residuals little joint authority; test actual responses without changing physics.',case='roller_brake_1s',seeds=[917000,917001],phases_s=[2.,3.,5.],duration_s=.2,offsets_rad=[-.15,0.,.15],joints=[0,2,3,4],note='Only the chosen left joint target is perturbed. This measures local coupled response, not a trained candidate or independent performance benchmark.')
(p/'action_response_recipe.json').write_text(json.dumps(recipe,indent=2))
def run(job):
 seed,start,joint,offset=job;actor=NativeAnchor(source)
 w=World(TASKS['roller'],roller_contract=True,state_input=actor.state_input);trace=[]
 try:
  obs=w.reset(seed,'keyboard_roller_brake_1s')
  for i in range(round((start+.2)/.02)):
   a=actor(obs[None])[0]
   if i>=round(start/.02):a[joint]+=offset
   obs=w.step(a)
   if i>=round(start/.02):trace.append(dict(t=w.t,q=w.state.q.tolist(),tau=w.state.extra['raw']['tau'],vel=w.features['vel'].tolist(),z=w.features['z'],tilt=w.features['tilt'],ctrl=(w.home+a).tolist()))
  return dict(seed=seed,start=start,joint=joint,offset=offset,trace=trace)
 finally:w.close()
jobs=[(seed,start,joint,offset) for seed in recipe['seeds'] for start in recipe['phases_s'] for joint in recipe['joints'] for offset in [-.15,.15]]
jobs += [(seed,start,0,0.) for seed in recipe['seeds'] for start in recipe['phases_s']]
with ThreadPoolExecutor(max_workers=4) as pool:
 rows=[]
 for r in pool.map(run,jobs):
  rows.append(r);print(json.dumps({k:r[k] for k in ['seed','start','joint','offset']}),flush=True)
(p/'action_response_raw.json').write_text(json.dumps(rows))
controls={(r['seed'],r['start']):r for r in rows if r['offset']==0}
report=[]
for r in rows:
 if r['offset']==0:continue
 b=controls[r['seed'],r['start']];i=r['joint'];q=np.array([x['q'] for x in r['trace']]);qb=np.array([x['q'] for x in b['trace']]);tau=np.array([x['tau'] for x in r['trace']]);taub=np.array([x['tau'] for x in b['trace']])
 report.append(dict(seed=r['seed'],start=r['start'],joint=i,offset=r['offset'],joint_final_delta=float(q[-1,i]-qb[-1,i]),max_abs_joint_delta=float(abs(q[:,i]-qb[:,i]).max()),mean_abs_tau_delta=float(abs(tau[:,i]-taub[:,i]).mean()),final_speed_delta=float(np.linalg.norm(r['trace'][-1]['vel'][:2])-np.linalg.norm(b['trace'][-1]['vel'][:2]))))
(p/'action_response_summary.json').write_text(json.dumps(report,indent=2));print('COMPLETE',len(report))
