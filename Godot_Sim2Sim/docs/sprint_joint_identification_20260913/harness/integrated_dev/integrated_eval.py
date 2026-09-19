"""Frozen a402 actor, one input contract repair; never select model parameters."""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess
import numpy as np
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.standalone.suite import run,run_case,runtime_inputs
from sim2sim.standalone.replay import shadow
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve()
B=Path('results/sprint_stop_state_20260912/native_joint_fd').resolve()
FIELDS=['obs','action','command','ctrl','last_action']
def identical(old,new):
 a=json.loads(Path(old).read_text())['rows'];b=json.loads(Path(new).read_text())['rows']
 assert len(a)==len(b)
 maxima={k:float(np.max(np.abs(np.asarray([r[k] for r in a])-np.asarray([r[k] for r in b])))) for k in FIELDS}
 assert max(maxima.values())==0.,maxima
 return maxima

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','final'],required=True);a=p.parse_args()
 assert Path(os.environ['SIM2SIM_ACTIVE_BUDGET_DIR']).resolve()==R
 out=R/('integrated_'+a.stage);out.mkdir(exist_ok=False)
 old=json.loads((B/'suite/summary.json').read_text());control=json.loads((B/'control.json').read_text())
 control['walk']['twist_limits']['sprint_yaw_reversal_s']=.2
 project=out/'prepared'
 if a.stage=='dev':
  subprocess.run(['cp','--reflink=auto','-a',str(B/'suite/runtime'),str(project)],check=True,timeout=60)
  for name in ['driver.gd','play_brain.gd']:
   dest=project/'standalone'/name
   assert not dest.is_symlink()
   shutil.copy2(Path('godot/standalone')/name,dest)
  deployment=json.loads((project/'runtime_assets/deployment.json').read_text());deployment['control_config']=control
  atomic_json(project/'runtime_assets/deployment.json',deployment)
 else:
  dev=json.loads((R/'integrated_dev/completed.json').read_text());assert dev['accepted']
  frozen=json.loads((R/'candidate_freeze.json').read_text())
  assert runtime_inputs(R/'integrated_dev/suite/runtime')==frozen['runtime_inputs']
  subprocess.run(['cp','--reflink=auto','-a',str(R/'integrated_dev/suite/runtime'),str(project)],check=True,timeout=60)
 atomic_json(out/'control.json',control)
 seeds=range(927000,927016) if a.stage=='dev' else range(928000,928050)
 cases=write_cases(out/'cases',seeds,.3,paired=True,control=control)
 extras=[]
 for e in old['episodes']:
  if 927000 <= e['seed'] <= 927015:continue
  c=json.loads(Path(e['case_path']).read_text());c.setdefault('control_config',{}).setdefault('walk',{}).setdefault('twist_limits',{})['sprint_yaw_reversal_s']=.2
  path=out/'regressions'/Path(e['case_path']).name;path.parent.mkdir(exist_ok=True);atomic_json(path,c);extras.append(path)
 assert len(extras)==7,[(e['case'],e['seed']) for e in old['episodes']]
 atomic_json(out/'protocol.json',dict(stage=a.stage,seeds=list(seeds),control=control,main_cases=len(cases),regressions=len(extras),runtime_inputs=runtime_inputs(project),gate='All physical cases pass with no falls; paired mean sprint speed >=115%; ordinary mean speed >=95% frozen dev baseline. Unchanged models. Development ordinary trajectories bit-identical. Integrated canary bit-identical to successful prototype; Python shadow error <1e-5.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 if a.stage=='dev':
  canary=next(p for p in cases if json.loads(p.read_text())['case']=='sprint_alternate' and json.loads(p.read_text())['seed']==927001)
  models=old['models']
  result=run_case(canary,out/'canary',models,project=project)
  assert result['completed'],result
  equivalence=identical(R/'yaw_probe/927001_0.2/trace.json',result['trace'])
  reference=shadow(result['trace'],project);assert reference['passed'],reference
  atomic_json(out/'canary.json',dict(prototype_max_abs=equivalence,shadow=reference,task=result['task_metrics']))
 summary=run(cases+extras,out/'suite',workers=4,project=project)
 assert summary['errors']==0,summary['errors']
 mainpaths={str(p.resolve()) for p in cases};main={**summary,'episodes':[e for e in summary['episodes'] if e['case_path'] in mainpaths]}
 pairs=paired_acceptance(main,cases);atomic_json(out/'paired_acceptance.json',pairs)
 candidate=[e for e in main['episodes'] if not json.loads(Path(e['case_path']).read_text())['ordinary_control']]
 ordinary=[e for e in main['episodes'] if json.loads(Path(e['case_path']).read_text())['ordinary_control']]
 regressions=[e for e in summary['episodes'] if e['case_path'] not in mainpaths]
 baseline=json.loads((B/'completed.json').read_text());old_speed={r['case']:r['ordinary_vx'] for r in baseline['comparisons']}
 retention=[dict(case=r['case'],ratio=r['ordinary_vx']/old_speed[r['case']],passed=r['ordinary_vx']>=.95*old_speed[r['case']]) for r in pairs['comparisons']]
 matched=[]
 if a.stage=='dev':
  index={(e['case'],e['seed']):e for e in old['episodes']}
  for e in ordinary:matched.append(dict(case=e['case'],seed=e['seed'],max_abs=identical(index[e['case'],e['seed']]['trace'],e['trace'])))
  atomic_json(out/'ordinary_equivalence.json',dict(passed=True,results=matched))
 result=dict(completed=True,stage=a.stage,candidate_count=len(candidate),candidate_pass=sum(e['task_metrics']['success'] for e in candidate),ordinary_count=len(ordinary),ordinary_pass=sum(e['task_metrics']['success'] for e in ordinary),regression_count=len(regressions),regression_pass=sum(e['task_metrics']['success'] for e in regressions),falls=sum(e['task_metrics']['fell'] for e in summary['episodes']),failed=[dict(case=e['case'],seed=e['seed'],task=e['task_metrics']) for e in summary['episodes'] if not e['task_metrics']['success']],comparisons=[{k:v for k,v in r.items() if k!='pairs'} for r in pairs['comparisons']],ordinary_retention=retention,accepted=bool(pairs['accepted'] and all(e['task_metrics']['success'] and not e['task_metrics']['fell'] for e in summary['episodes']) and all(r['passed'] for r in retention)),models=summary['models'])
 atomic_json(out/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
