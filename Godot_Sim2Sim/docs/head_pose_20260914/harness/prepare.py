from pathlib import Path
import hashlib,json,subprocess,sys
from sim2sim.standalone.suite import runtime_inputs
R=Path(__file__).resolve().parent;I=Path('/home/ethan/Projects/MicroDuck-SpeedControls');O=Path('/home/ethan/Projects/MicroDuck/sim2sim')
source=O/'results/workshop-hub/runtime';before=runtime_inputs(source)
subprocess.run(['cp','--reflink=auto','-a',str(source),str(R/'base')],check=True,timeout=60)
assert runtime_inputs(R/'base')==before and runtime_inputs(source)==before
(R/'base_inputs.json').write_text(json.dumps(before,indent=2)+'\n')
sys.path.insert(0,str(I/'scripts'))
from native_neck_bias_experiment import prepare
for angle in [0,-5,-10]:prepare(R/'base',R/('neck_'+str(abs(angle))),angle)
(R/'prepared.json').write_text(json.dumps(dict(completed=True,source=str(source),variants=[0,-5,-10]),indent=2)+'\n')
print('Three isolated variants prepared, only driver differs',flush=True)
