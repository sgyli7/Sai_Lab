from pathlib import Path
import json,subprocess
import numpy as np
R=Path(__file__).resolve().parent
control=json.loads((R/'base/runtime_assets/deployment.json').read_text())['control_config']
case=dict(mode='walk',seconds=8,seed=929100,control_config=control,segments=[dict(at=0,held=[]),dict(at=1,held=['sprint','fwd']),dict(at=3,held=[],taps=['reset']),dict(at=4,held=['sprint','fwd']),dict(at=5,held=['fwd']),dict(at=6,held=[])])
path=R/'reset_case.json';path.write_text(json.dumps(case,indent=2)+'\n');trace=R/'reset_trace.json'
with (R/'reset_player.log').open('x') as log:subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(R/'neck_10'),'res://standalone/main.tscn','--',f'--replay={path}',f'--trace={trace}'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
p=json.loads(trace.read_text());assert p['summary']['resets']==1 and not p['summary']['error']
rows=p['rows'];resets=[(i,r) for i,r in enumerate(rows) if abs(r['episode_t'])<1e-9];assert len(resets)==2
for i,row in resets:
 assert row['experiment_neck_bias_rad']==0 and not any(row['last_action']) and not any(row['obs'][34:48])
after=[r for r in rows if 5.5<=r['t']<=7.9];assert after and all(r['experiment_neck_bias_rad']==0 for r in after)
(R/'reset_completed.json').write_text(json.dumps(dict(passed=True,resets=1,reset_rows=[i for i,r in resets],exit_bias_zero_after_half_second=True,scope='Diagnostic only; not a scored task, one deliberate reset. No automatic resets used in acceptance suites.'),indent=2)+'\n');print('RESET CHECK PASSED')
