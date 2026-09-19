"""Fixed 2x2 contact response probe; existing native tapes, GPU physics only."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch
import warp as wp
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld

R=Path('results/sprint_contact_calibration_20260913')


def microstep(w,action):
    # Exact GpuWorld.step motor and observer equations, one tick instead of
    # four; the parity check below guards this diagnostic subdivision.
    caller=torch.cuda.current_stream();w.torch_stream.wait_stream(caller)
    with torch.cuda.stream(w.torch_stream),wp.ScopedStream(w.stream):
        target=w.home+action
        v=w.observer.qd
        w.ctrl.copy_((.55*(target-w.qpos[:,w.qi])).clamp(-w.limit,w.limit)-.053*v-.0048*torch.tanh(v/.05))
        wp.capture_launch(w.step_graph)
        w.observer.update(w.qpos[:,w.qi],w.qpos[:,w.qa+3:w.qa+7])
        w.last.copy_(action)
    caller.wait_stream(w.torch_stream)


def initialize(w,rows):
    w.reset(torch.arange(len(rows),device='cuda'))
    w.qpos[:,w.qa:w.qa+3]=torch.tensor([r[0]['body']['base_pos'] for r in rows],device='cuda')
    w.qpos[:,w.qa+3:w.qa+7]=torch.tensor([r[0]['body']['base_quat'] for r in rows],device='cuda')
    w.qpos[:,w.qi]=torch.tensor([r[0]['raw']['q'] for r in rows],device='cuda')
    w.qvel.zero_();w.forward()
    w.observer.reset(torch.arange(len(rows),device='cuda'),w.qpos[:,w.qi],w.qpos[:,w.qa+3:w.qa+7])


def main():
    if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R.resolve():
        raise RuntimeError('Active-budget supervision required')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    output=R/'contact_probe';output.mkdir(exist_ok=False)
    source=Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json')
    selected=sorted([e for e in json.loads(source.read_text())['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    assert len(selected)==16
    rows=[json.loads(Path(e['trace']).read_text())['rows'] for e in selected]
    settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
    profiles=[('source','source',False),('two_tick','two_tick',False),('imp99','source',True),('two_tick_imp99','two_tick',True)]
    protocol=dict(profiles=profiles,velocity_observer='joint_fd',physics_step=.005,policy_step=.02,
        horizon_s=2,landing_window=[.04,.30],standing_window=[.3,1.],moving_window=[1.,1.3],
        calibration_train_seeds=[927000,927007],calibration_validation_seeds=[927008,927015],
        objective='Mean normalized squared COM velocity (.05 m/s), joint q (.02 rad), body position (.002 m) errors; each group equally weighted.',
        gate='At least 20% objective reduction in landing and movement, both seed splits; no component or standing objective >10% worse; pre-contact joint RMS no worse than baseline+1e-5 rad.',
        contact_buffer='Last GPU physics-step contact buffer; not assumed synchronized with Godot contact-report internals.',
        lag_probe='Descriptive +/-10 ms COM velocity alignment during first landing; NEVER used to shift acceptance time.',
        microstep_tolerances=dict(qpos=1e-8,qvel=5e-7,observer_qd=1e-6),
        microstep_tolerance_basis='Normal repeated 20ms operations themselves vary up to 5.96e-7 in filtered qd (parity_probe.json), including fresh worlds. Initial 1e-7 diagnostic cap was below this repeatability; amended before any contact-profile measurements. Policy export/acceptance gates unchanged.',
        scope='Fixed development-only physics calibration, no actor selection/training or game physics edits.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sources=[dict(seed=e['seed'],trace=e['trace'],sha256=hashlib.sha256(Path(e['trace']).read_bytes()).hexdigest()) for e in selected],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'protocol.json',protocol)
    native={k:np.array([[get(r[i]) for r in rows] for i in range(101)]) for k,get in {
        'position':lambda x:x['body']['base_pos'],'velocity':lambda x:x['body']['base_linvel'],
        'joint':lambda x:x['raw']['q'],'contacts':lambda x:[f['ground_contact'] for f in x['raw']['feet']]}.items()}
    action=torch.tensor([[r[i]['action'] for r in rows] for i in range(100)],device='cuda')
    results={}
    for name,contact,strong_impedance in profiles:
        w=GpuWorld(16,settings,contact=contact,feet='jolt',graphs=True,velocity_observer='joint_fd')
        try:
            if strong_impedance:
                w.model.geom_solimp[:,:2]=.99
                target=wp.to_torch(w.wm.geom_solimp)
                target.copy_(torch.as_tensor(w.model.geom_solimp,device='cuda',dtype=target.dtype).expand_as(target))
            initialize(w,rows)
            # One real decision and four diagnostic substeps must agree before
            # contact instrumentation can be trusted. Fresh reset each side.
            w.step(action[0]);expected_q=w.qpos.clone();expected_v=w.qvel.clone();expected_qd=w.observer.qd.clone()
            initialize(w,rows)
            for _ in range(4):microstep(w,action[0])
            parity_components=dict(qpos=float((w.qpos-expected_q).abs().max()),qvel=float((w.qvel-expected_v).abs().max()),observer_qd=float((w.observer.qd-expected_qd).abs().max()))
            parity=max(parity_components.values())
            if any(v>protocol['microstep_tolerances'][k] for k,v in parity_components.items()):
                raise RuntimeError('Microstep subdivision exceeds measured normal repeatability: '+str(parity_components))
            initialize(w,rows)
            states=torch.empty((401,16,13),device='cuda');joints=torch.empty((401,16,14),device='cuda')
            contact_flags=np.zeros((401,16,2),bool);contact_dist=np.full((401,16,2),np.nan)
            import mujoco
            floor=mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
            foot=[mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['left_foot_collision','right_foot_collision']]
            start=time.monotonic()
            with torch.inference_mode():
                for tick in range(401):
                    pos,quat,ang,vel=w.state();states[tick]=torch.cat((pos,quat,ang,vel),dim=-1);joints[tick]=w.qpos[:,w.qi]
                    count=int(wp.to_torch(w.wd.nacon).item())
                    geom=wp.to_torch(w.wd.contact.geom)[:count].cpu().numpy()
                    worldid=wp.to_torch(w.wd.contact.worldid)[:count].cpu().numpy()
                    dist=wp.to_torch(w.wd.contact.dist)[:count].cpu().numpy()
                    margin=wp.to_torch(w.wd.contact.includemargin)[:count].cpu().numpy()
                    for side,fid in enumerate(foot):
                        mask=(((geom[:,0]==floor)&(geom[:,1]==fid))|((geom[:,1]==floor)&(geom[:,0]==fid)))&(dist<margin)
                        for world in np.unique(worldid[mask]):
                            contact_flags[tick,world,side]=True
                            contact_dist[tick,world,side]=float(dist[mask&(worldid==world)].min())
                    if tick<400:microstep(w,action[tick//4])
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            state=states.cpu().numpy();joint=joints.cpu().numpy()
            if not np.isfinite(state).all():raise RuntimeError('Nonfinite contact response')
            np.savez_compressed(output/(name+'.npz'),states=state,joints=joint,contact_flags=contact_flags,contact_dist=contact_dist,**{'native_'+k:v for k,v in native.items()})
            error=dict(velocity=np.mean((state[::4,:,10:13]-native['velocity'])**2,axis=-1)/.05**2,
                       joint=np.mean((joint[::4]-native['joint'])**2,axis=-1)/.02**2,
                       position=np.mean((state[::4,:,:3]-native['position'])**2,axis=-1)/.002**2)
            metric={}
            for split,ids in [('train',slice(0,8)),('validation',slice(8,16))]:
                metric[split]={}
                for phase,(lo,hi) in [('landing',(.04,.3)),('standing',(.3,1.)),('moving',(1.,1.3))]:
                    t=np.arange(101)*.02;mask=(t>=lo-1e-8)&(t<hi-1e-8)
                    values={k:float(v[mask,ids].mean()) for k,v in error.items()}
                    metric[split][phase]=dict(components=values,objective=float(np.mean(list(values.values()))))
            lag={}
            for offset in [-2,-1,0,1,2]:
                ix=np.arange(2,16);difference=state[ix*4+offset,:,10:13]-native['velocity'][ix]
                lag[str(offset*.005)]=float(np.sqrt(np.mean(difference**2)))
            values=dict(microstep_parity_max_abs=parity,microstep_parity_components=parity_components,metrics=metric,lag_probe_velocity_rmse=lag,
                first_contact_gpu_s=[next((i*.005 for i in range(401) if contact_flags[i,j].any()),None) for j in range(16)],
                first_contact_native_s=[next((i*.02 for i in range(101) if native['contacts'][i,j].any()),None) for j in range(16)],
                precontact_joint_rms=float(np.sqrt(np.mean((joint[4]-native['joint'][1])**2))),elapsed_s=elapsed)
            results[name]=values;atomic_json(output/(name+'.json'),values)
            print(json.dumps(dict(profile=name,**values)),flush=True)
        finally:w.close()
    decisions={}
    baseline=results['source']
    for name,v in results.items():
        if name=='source':continue
        checks=[]
        for split in ['train','validation']:
            for phase in ['landing','moving','standing']:
                a=baseline['metrics'][split][phase];b=v['metrics'][split][phase]
                ratio=b['objective']/a['objective']
                maximum=max(b['components'][k]/a['components'][k] for k in a['components'])
                checks.append(dict(split=split,phase=phase,objective_ratio=ratio,max_component_ratio=maximum,
                    passed=bool(ratio<=(1.1 if phase=='standing' else .8) and maximum<=1.1)))
        precontact=v['precontact_joint_rms']<=baseline['precontact_joint_rms']+1e-5
        decisions[name]=dict(checks=checks,precontact_passed=precontact,passed=precontact and all(c['passed'] for c in checks))
    atomic_json(output/'completed.json',dict(results=results,decisions=decisions,eligible=[k for k,v in decisions.items() if v['passed']],completed=True))


if __name__=='__main__':main()
