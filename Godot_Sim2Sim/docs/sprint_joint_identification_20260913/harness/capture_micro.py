"""Native 200 Hz pose/solver velocity consistency probe; no forces changed."""
from pathlib import Path
import copy,hashlib,json,os,shutil,subprocess
import numpy as np
from sim2sim.godot_proc import _headless_overlay
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve()
PRIOR=Path('results/sprint_stop_state_20260912/native_joint_fd').resolve()
HOOK='''
func _joint_micro_snapshot() -> void:
\t# [DEBUG-joint-micro] Read after integration, before control/next force.
\tvar poses: Array = []
\tfor name in _bodies:
\t\tvar b: RigidBody3D = _bodies[name]
\t\tvar p := _g2m(b.global_transform.origin)
\t\tvar q := _basis_to_m_quat(b.global_transform.basis)
\t\tvar v := _g2m(b.linear_velocity)
\t\tvar w := _g2m(b.angular_velocity)
\t\tposes.append({"name":name,"pos":[p.x,p.y,p.z],"quat":[q.w,q.x,q.y,q.z],"linvel":[v.x,v.y,v.z],"angvel":[w.x,w.y,w.z]})
\tif not session.has("joint_micro_rows"): session["joint_micro_rows"] = []
\tsession.joint_micro_rows.append({"t":_t,"poses":poses})
'''

def main():
 if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R:raise RuntimeError('Supervision required')
 out=R/'micro_capture';out.mkdir(exist_ok=False)
 atomic_json(out/'protocol.json',dict(seeds=[927000,927001,927008,927009],seconds=2,physics_hz=200,control_hz=50,gate='Every20ms obs/action/command/last_action/ctrl must be identical to frozen native prefix, 401 snapshots at exact 5ms increments',hypothesis='Split position stabilization yields a measurable COM displacement minus post-step solver velocity*dt; quantify this without claiming spring elasticity or policy improvement.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 summary=json.loads((PRIOR/'suite/summary.json').read_text());results=[]
 for seed in [927000,927001,927008,927009]:
  entry=next(e for e in summary['episodes'] if e['case']=='sprint_alternate' and e['seed']==seed)
  folder=out/str(seed);folder.mkdir()
  case=copy.deepcopy(json.loads(Path(entry['case_path']).read_text()));case['seconds']=2;case['scoring']['end']=2;case['segments']=[s for s in case['segments'] if s['at']<2];case['sprint_intervals']=[[1,2]]
  atomic_json(folder/'case.json',case);overlay=_headless_overlay(PRIOR/'suite/runtime')
  try:
   (overlay/'standalone').unlink();shutil.copytree(PRIOR/'suite/runtime/standalone',overlay/'standalone')
   p=overlay/'standalone/driver.gd';s=p.read_text()
   needle='\t_refresh_after_physics(delta)\n\tif _remaining <= 0:';assert s.count(needle)==1
   s=s.replace(needle,'\t_refresh_after_physics(delta)\n\t_joint_micro_snapshot()\n\tif _remaining <= 0:')
   needle='JSON.stringify({"summary":result,"rows":session.rows})';assert s.count(needle)==1
   s=s.replace(needle,'JSON.stringify({"summary":result,"rows":session.rows,"joint_micro":session.joint_micro_rows})')
   p.write_text(s+HOOK)
   with (folder/'player.log').open('w') as log:
    subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(folder/'case.json'),'--trace='+str(folder/'trace.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=25)
  finally:shutil.rmtree(overlay)
  new=json.loads((folder/'trace.json').read_text());old=json.loads(Path(entry['trace']).read_text())['rows'][:100]
  maxima={k:float(np.max(np.abs(np.array([x[k] for x in old])-np.array([x[k] for x in new['rows']])))) for k in ['obs','action','command','last_action','ctrl']}
  assert max(maxima.values())==0.,maxima
  assert len(new['joint_micro'])==401
  np.testing.assert_allclose([x['t'] for x in new['joint_micro']],np.arange(401)*.005,atol=1e-12,rtol=0)
  results.append(dict(seed=seed,directory=str(folder),prefix_max_abs=maxima,samples=401));print(seed,flush=True)
 atomic_json(out/'completed.json',dict(completed=True,results=results))

if __name__=='__main__':main()
