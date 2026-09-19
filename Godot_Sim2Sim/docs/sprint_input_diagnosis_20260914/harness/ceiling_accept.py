"""A single preselected command-ceiling counterfactual, not a parameter sweep."""
from pathlib import Path
import json
import hashlib
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run

R=Path('results/sprint_input_diagnosis_20260914').resolve()
out=R/'ceiling_gate';out.mkdir(exist_ok=False)
deployment=json.loads((R/'runtime/runtime_assets/deployment.json').read_text())
control=deployment['control_config']
protocol=dict(hypothesis='If the forward command ceiling alone prevents useful faster locomotion, 0.45 m/s should increase speed while preserving all existing no-fall/straight/turn/stop gates.',
              changed_variable='sprint_vmax_x only, 0.30 -> 0.45',seed_start=929100,seed_count=4,models={k:v['sha256'] for k,v in deployment['policies'].items()},
              decision='Reject the ceiling-only fix on any new failed native case; no use of reserved final seeds.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
result={}
for name,speed in [('baseline',.3),('candidate',.45)]:
    cases=write_cases(out/name/'cases',range(929100,929104),speed,paired=False,control=control)
    summary=run(cases,out/name/'suite',workers=2,project=R/'runtime')
    episodes=summary['episodes']
    result[name]=dict(count=len(episodes),passed=sum(e.get('task_metrics',{}).get('success',False) for e in episodes),
        falls=sum(e.get('task_metrics',{}).get('fell',False) for e in episodes),errors=summary['errors'],
        failed=[dict(case=e['case'],seed=e['seed'],metrics=e.get('task_metrics'),error=e.get('error')) for e in episodes if not e.get('task_metrics',{}).get('success',False)],
        long_speed=[e['task_metrics']['sustained_mean_vx'] for e in episodes if e['case']=='sprint_long'])
    (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
result['accepted']=result['candidate']['passed']==32 and result['candidate']['falls']==0 and result['candidate']['errors']==0
(out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
