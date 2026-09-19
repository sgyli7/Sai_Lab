"""Expanded development screening of the already trained population; no holdout claims."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from sai_loaded_mujoco import run
from train_sai_loaded_suspension import assess
from validate_sai_loaded_suspension import cases
ROOT=Path('results/sai-cargo-suspension-20260915')

def job(args):
    case,p,out=args
    return run(parameters=p,out=out,**case)

def main():
    out=ROOT/'screening';out.mkdir(exist_ok=True)
    h=[json.loads(l) for l in (ROOT/'training/trials.jsonl').read_text().splitlines()];seen=set();population=[]
    for r in sorted(h,key=lambda r:r['loss']):
        p=tuple(r['parameters'])
        if p not in seen and not r['violations']:population.append(r)
        seen.add(p)
    matrix=cases();failure=matrix[10]
    protocol=dict(reason='First validation revealed unrestrained payload slip on oblique 40 mm descent. That matrix is now development data, not final validation.',population=[r['index'] for r in population],oracle_correction='Speed retention compares at most the requested .16 m/s; baseline became a sideways runaway and fell beyond the terrain. Its .175 m/s body speed is not successful forward progress. Original failed verdict retained.',first_case=failure,full_screen='best three feasible survivors by original training objective',final_seeds=[2003,2203,2401],no_parameter_changes=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    baseline=[r['baseline'] for r in json.loads((ROOT/'heldout/acceptance.json').read_text())['pairs']]
    survivors=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[(failure,r['parameters'],out/f'probe-{r["index"]:03d}') for r in population]
        for r,res in zip(population,[json.loads((args[2]/'report.json').read_text()) for args in jobs]):
            m=res['metrics'];ok=m['completed'] and not m['cargo_lost'] and m['min_upright']>.8 and m['speed']>=max(.08,min(.16,baseline[10]['speed'])*.9)
            print(json.dumps(dict(index=r['index'],probe_pass=ok,metrics=m)),flush=True)
            if ok:survivors.append(r)
        rows=[]
        for r in survivors[:3]:
            jobs=[(case,r['parameters'],out/f'full-{r["index"]:03d}'/f'{i:02d}-{case["kind"]}') for i,case in enumerate(matrix)]
            reports=list(pool.map(job,jobs));bad=[];ratios=[]
            for i,(case,res,b) in enumerate(zip(matrix,reports,baseline)):
                m=res['metrics'];stairs=case['kind'].startswith(('up','down','mixed'))
                if not m['completed'] or m['cargo_lost'] or m['min_upright']<=.8:bad.append(f'{i}:completion')
                if m['speed']<max(.08 if stairs else .4,min(.16 if stairs else .5,b['speed'])*.9):bad.append(f'{i}:speed')
                if abs(m['height_error_mean'])>=.025:bad.append(f'{i}:height')
                if case.get('clamped') and m['bilateral_clamp_fraction']<=.9:bad.append(f'{i}:clamp')
                ratios.append(sum(w*m[k]/max(1e-8,b[k]) for k,w in [('cargo_accel_rms',.35),('cargo_accel_p95',.25),('cargo_jerk_rms',.15),('deck_accel_rms',.15),('cargo_accel_peak',.1)]))
            row=dict(index=r['index'],parameters=r['parameters'],violations=bad,loss=sum(ratios)/len(ratios)+.35*max(ratios),reports=reports)
            rows.append(row);(out/'screened.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
    valid=[r for r in rows if not r['violations']]
    if valid:
        best=min(valid,key=lambda r:r['loss']);profile=json.loads((ROOT/'training/candidate.json').read_text());profile.update(parameters=best['parameters'],selected_trial=best['index'],status='candidate_after_expanded_development',development_loss=best['loss'],development_protocol=protocol)
        (out/'candidate.json').write_text(json.dumps(profile,indent=2)+'\n')
    print('SCREENING_COMPLETE',len(valid),flush=True)
if __name__=='__main__':main()
