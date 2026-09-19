from pathlib import Path
import json,sys
from unittest.mock import patch
import torch
from sim2sim.research.joint_compliance import add_joint_compliance_direct
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913')
sys.path.insert(0,str(R.resolve()));from split_proxy import SplitWorld
sys.path.insert(0,str(Path('results/sprint_contact_calibration_20260913').resolve()));from contact_probe import initialize
summary=json.loads(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json').read_text())
entries=sorted([e for e in summary['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
rows=[json.loads(Path(e['trace']).read_text())['rows'] for e in entries]
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
def connections(spec,godot_spec,stiffness,damping):
 result=add_joint_compliance_direct(spec,godot_spec,stiffness,damping)
 for eq in spec.equalities:eq.solref=[0.,-200.]
 return result
with patch('sim2sim.research.joint_compliance.add_joint_compliance_direct',connections):
 w=SplitWorld(16,settings,contact='two_tick',feet='jolt',graphs=True,velocity_observer='joint_fd',joint_compliance_direct=(1.,200.))
try:
 w.prepare_projection(.5775);initialize(w,rows);w.canary()
 atomic_json(R/'projection_canary_diagnosis.json',w.canary_diagnostic)
finally:w.close()
