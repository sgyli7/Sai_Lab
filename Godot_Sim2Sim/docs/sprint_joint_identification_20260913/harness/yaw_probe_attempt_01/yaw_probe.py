"""One declared control-transition comparison, using the frozen a402 policy."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,os,shutil,subprocess,sys
from unittest.mock import patch
import numpy as np
from sim2sim.godot_proc import _headless_overlay
from sim2sim.standalone.score import score
from sim2sim.standalone import replay
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve();PRIOR=Path('results/sprint_stop_state_20260912/native_joint_fd').resolve()
sys.path.insert(0,str(R));from yaw_reversal import YawReversalControl

def run(entry,duration):
 folder=R/'yaw_probe'/f"{entry['seed']}_{duration:.1f}";folder.mkdir(exist_ok=False)
 case=copy.deepcopy(json.loads(Path(entry['case_path']).read_text()));case['control_config']['walk']['sprint_yaw_reversal_s']=duration
 atomic_json(folder/'case.json',case)
 overlay=_headless_overlay(PRIOR/'suite/runtime')
 try:
  (overlay/'standalone').unlink();shutil.copytree(PRIOR/'suite/runtime/standalone',overlay/'standalone')
  p=overlay/'standalone/motion_control.gd';p.rename(overlay/'standalone/motion_control_base.gd');shutil.copy2(R/'yaw_reversal.gd',p)
  with (folder/'player.log').open('w') as log:
   subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(folder/'case.json'),'--trace='+str(folder/'trace.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=35)
 finally:shutil.rmtree(overlay)
 old=json.loads(Path(entry['trace']).read_text())['rows'];new=json.loads((folder/'trace.json').read_text())['rows']
 prefix=300 if duration>0 else len(old)
 maxima={k:float(np.max(np.abs(np.asarray([x[k] for x in old[:prefix]])-np.asarray([x[k] for x in new[:prefix]])))) for k in ['obs','action','command','ctrl','last_action']}
 assert max(maxima.values())==0.,maxima
 result=score(folder/'trace.json',folder/'case.json');atomic_json(folder/'score.json',result)
 with patch.object(replay,'MotionControl',YawReversalControl):shadow=replay.shadow(folder/'trace.json',PRIOR/'suite/runtime')
 atomic_json(folder/'shadow.json',shadow)
 return dict(seed=entry['seed'],duration=duration,directory=str(folder),prefix_max_abs=maxima,task=result['task_metrics'],shadow=shadow)

def main():
 if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R:raise RuntimeError('Supervision required')
 out=R/'yaw_probe';out.mkdir(exist_ok=False)
 entries=sorted([e for e in json.loads((PRIOR/'suite/summary.json').read_text())['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
 atomic_json(out/'protocol.json',dict(seeds=[e['seed'] for e in entries],duration_seconds=.2,scope='Control change only; a402 policy, ordinary controls and game physics frozen. No reduction of cruise speed; active reversal interpolates over ten decisions, straight/stop/ordinary immediately use old command.',gate='Both direction phases and stop/straight/fall gates in original full 18s case, all 16 dev seeds; disabled canary is bit-identical, active prefix before first reversal is bit-identical. Python/native command and action parity use existing strict limits. No parameter sweep.',code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'yaw_reversal.py',R/'yaw_reversal.gd']}))
 control=run(next(e for e in entries if e['seed']==927001),0.)
 atomic_json(out/'disabled_canary.json',control)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda e:run(e,.2),entries))
 atomic_json(out/'completed.json',dict(completed=True,results=results,disabled=control,all_cases_passed=all(x['task']['success'] for x in results)))
 print(json.dumps(dict(passed=sum(x['task']['success'] for x in results),total=len(results),results=[dict(seed=x['seed'],success=x['task']['success'],turns=x['task']['turns'],fell=x['task']['fell']) for x in results])),flush=True)

if __name__=='__main__':main()
