"""Re-evaluate previously trained parameters under a causally validated descent rule."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sai_loaded_mujoco import run

def job(args):
    c,p,o=args
    return run(parameters=p,out=o,**c)

def main():
    root=Path('results/sai-cargo-suspension-20260915');out=root/'descent-selection';out.mkdir(exist_ok=False)
    previous=json.loads((root/'final-heldout/acceptance.json').read_text());cases=[x['case'] for x in previous['pairs']]
    trials=[json.loads(l) for l in (root/'training/trials.jsonl').read_text().splitlines()];ids=[13,36,45]
    (out/'protocol.json').write_text(json.dumps(dict(reason='Two independent down40 slip failures require a new descent controller. Final-heldout is consumed development data. Fixed candidate IDs from original training; no new parameter perturbation.',ids=ids,cases=cases,next_final_seeds=[3001,3203,3407]),indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for index in ids:
            p=trials[index]['parameters'];reports=list(pool.map(job,[(c,p,out/f'trial-{index}'/f'{i:02d}-{c["kind"]}') for i,c in enumerate(cases)]));bad=[];ratios=[]
            for i,(c,r,b) in enumerate(zip(cases,reports,previous['pairs'])):
                m=r['metrics'];bm=b['baseline'];stairs=c['kind'].startswith(('up','down','mixed'))
                if not m['completed'] or m['cargo_lost'] or m['min_upright']<=.8:bad.append(f'{i}:completion')
                if m['speed']<max(.08 if stairs else .4,min(.16 if stairs else .5,bm['speed'])*.9):bad.append(f'{i}:speed')
                if abs(m['height_error_mean'])>=.025:bad.append(f'{i}:height')
                if c.get('clamped') and m['bilateral_clamp_fraction']<=.9:bad.append(f'{i}:clamp')
                ratios.append(sum(w*m[k]/max(1e-8,bm[k]) for k,w in [('cargo_accel_rms',.35),('cargo_accel_p95',.25),('cargo_jerk_rms',.15),('deck_accel_rms',.15),('cargo_accel_peak',.1)]))
            row=dict(index=index,parameters=p,violations=bad,loss=sum(ratios)/len(ratios)+.35*max(ratios),reports=reports);rows.append(row)
            (out/'screened.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
    valid=[r for r in rows if not r['violations']]
    if valid:
        best=min(valid,key=lambda r:r['loss']);profile=json.loads((root/'candidate-runtime.json').read_text());profile.update(parameters=best['parameters'],selected_trial=best['index'],status='candidate_contact_following_descent',descent_control='contact_following',development_loss=best['loss'])
        (out/'candidate.json').write_text(json.dumps(profile,indent=2)+'\n')
    print('COMPLETE',len(valid),flush=True)
if __name__=='__main__':main()
