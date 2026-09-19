"""Diagnostic-only native body-pose snapshot; frozen controller and physics."""
from pathlib import Path
import concurrent.futures,copy,hashlib,json,os,shutil,subprocess
import numpy as np
from sim2sim.godot_proc import _headless_overlay
from sim2sim.standalone.score import score
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve()
PRIOR=Path('results/sprint_stop_state_20260912/native_joint_fd').resolve()
SNAPSHOT='''func _send_dict(value: Dictionary) -> void:
\t# [DEBUG-joint-identification] Read-only, same-sample COM poses.
\tif bool(session.replay.get("diagnostic_joint_poses",false)):
\t\tvar poses: Array = []
\t\tfor name in _bodies:
\t\t\tvar b: RigidBody3D = _bodies[name]
\t\t\tvar p := _g2m(b.global_transform.origin)
\t\t\tvar q := _basis_to_m_quat(b.global_transform.basis)
\t\t\tposes.append({"name":name,"pos":[p.x,p.y,p.z],"quat":[q.w,q.x,q.y,q.z]})
\t\tvalue["diagnostic_joint_poses"] = poses
\tlocal_reply = value'''

def run(entry, suffix='', enabled=True):
    folder=R/'capture'/(str(entry['seed'])+suffix);folder.mkdir(exist_ok=False)
    case=copy.deepcopy(json.loads(Path(entry['case_path']).read_text()))
    case['seconds']=9.;case['scoring']['end']=9.
    case['segments']=[s for s in case['segments'] if s['at']<9.];case['sprint_intervals']=[[1.,9.]]
    case['diagnostic_joint_poses']=enabled
    atomic_json(folder/'case.json',case)
    overlay=_headless_overlay(PRIOR/'suite/runtime')
    try:
        # The overlay initially symlinks directories; detach standalone before
        # writing so the frozen runtime can never be modified through a link.
        (overlay/'standalone').unlink()
        shutil.copytree(PRIOR/'suite/runtime/standalone',overlay/'standalone')
        p=overlay/'standalone/driver.gd';text=p.read_text()
        needle='func _send_dict(value: Dictionary) -> void:\n\tlocal_reply = value'
        assert text.count(needle)==1
        p.write_text(text.replace(needle,SNAPSHOT))
        with (folder/'player.log').open('w') as log:
            subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(folder/'case.json'),'--trace='+str(folder/'trace.json')],stdout=log,stderr=subprocess.STDOUT,timeout=35,check=True)
    finally:
        shutil.rmtree(overlay)
    data=json.loads((folder/'trace.json').read_text());rows=data['rows']
    old=json.loads(Path(entry['trace']).read_text())['rows'][:450]
    assert len(rows)==len(old)==450
    maxima={k:float(np.max(np.abs(np.asarray([x[k] for x in old])-np.asarray([x[k] for x in rows])))) for k in ['obs','action','last_action','command','ctrl']}
    assert max(maxima.values())==0.,maxima
    if enabled:
        assert all(len(x['raw']['diagnostic_joint_poses'])==16 for x in rows)
    result=score(folder/'trace.json',folder/'case.json')
    atomic_json(folder/'score.json',result)
    return dict(seed=entry['seed'],enabled=enabled,directory=str(folder),source_trace=entry['trace'],source_sha256=hashlib.sha256(Path(entry['trace']).read_bytes()).hexdigest(),prefix_max_abs=maxima,task=result['task_metrics'])

def main():
    if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R:raise RuntimeError('Supervision required')
    out=R/'capture';out.mkdir(exist_ok=False)
    summary=json.loads((PRIOR/'suite/summary.json').read_text())
    entries=sorted([e for e in summary['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    atomic_json(out/'protocol.json',dict(seeds=[e['seed'] for e in entries],seconds=9.,diagnostic_hz=50,physics_hz=200,game_physics_changes=False,gate='All 450 original rows must be bit-identical for obs/action/last_action/command/ctrl; enabled+disabled canaries preserve original 927001 failure',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hypotheses=['Translation at joint anchors dominates end-foot representation residual','Constrained-axis rotation also matters','Frame conversion error would disagree with independently reported hinge angles']))
    bad=next(e for e in entries if e['seed']==927001)
    canary=run(bad,'_disabled',False);assert not canary['task']['success'] and not canary['task']['fell']
    checked=run(bad);assert not checked['task']['success'] and not checked['task']['fell']
    atomic_json(out/'canary.json',dict(disabled=canary,enabled=checked))
    results=[checked]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run,[e for e in entries if e!=bad]):
            results.append(result);print(json.dumps(dict(seed=result['seed'],parity=True)),flush=True)
    atomic_json(out/'completed.json',dict(completed=True,results=sorted(results,key=lambda e:e['seed']),canary=canary,scope='Diagnostic replay on existing development seeds; no new final evaluation or policy candidate'))

if __name__=='__main__':main()
