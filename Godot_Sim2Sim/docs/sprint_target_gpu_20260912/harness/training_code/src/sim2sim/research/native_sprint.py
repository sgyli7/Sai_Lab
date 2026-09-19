"""Full-history Jolt rollouts for bounded CUDA corrective learning.

Training traces remain explicitly ineligible for acceptance. Physics runs in
the exported native controller; Python orchestrates complete episodes only.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import torch

from sim2sim.godot_proc import _headless_overlay
from sim2sim.standalone.sprint import write_cases, templates
from sim2sim.standalone.replay import raw_state
from .queue import atomic_json
from .torch_walking import WalkingControl, rotate
from .sprint_constraints import SprintConstraints, yaw
from .locomotion_objectives import command_tracking


def prepare_sampler(canonical, project, instrumentation):
    project = Path(project).resolve()
    subprocess.run(['cp', '--reflink=auto', '-a', str(Path(canonical).resolve()), str(project)], check=True, timeout=60)
    # The already audited noise wrapper records actual executed actions/history.
    shutil.copy2(instrumentation, project / 'standalone/driver.gd')
    return project


def collect(project, actor, directory, seeds, noise_seed, *, sigma=.02, workers=4):
    project = Path(project).resolve()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    deployment_path = project / 'runtime_assets/deployment.json'
    deployment = json.loads(deployment_path.read_text())
    item = deployment['policies']['sprint']
    target = project / item['path'].removeprefix('res://')
    shutil.copy2(actor, target)
    item['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(deployment_path, deployment)
    cases = write_cases(directory / 'cases', seeds, .3, paired=False, control=deployment['control_config'])
    for case in cases:
        value = json.loads(case.read_text())
        value['training_exploration'] = True
        atomic_json(case, value)

    def run(index_case):
        i, case = index_case
        folder = directory / f'episode_{i:03d}'
        folder.mkdir()
        overlay = _headless_overlay(project)
        start = time.monotonic()
        try:
            with (folder / 'player.log').open('w') as log:
                subprocess.run(['godot', '--headless', '--fixed-fps', '200', '--path', str(overlay),
                                'res://standalone/main.tscn', '--', '--replay=' + str(case),
                                '--trace=' + str(folder / 'trace.json'), '--sample-std=' + str(sigma),
                                '--seed=' + str(noise_seed + i)], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=25)
        finally:
            shutil.rmtree(overlay)
        trace = json.loads((folder / 'trace.json').read_text())
        summary = trace['summary']
        if summary['error'] or summary['resets'] or summary['switches']:
            raise RuntimeError('Invalid native training episode: ' + str(summary))
        assert summary['training_exploration'] == sigma
        assert summary['models'] == {k: v['sha256'] for k, v in deployment['policies'].items()}
        assert len(trace['rows']) == round(json.loads(case.read_text())['seconds'] / .02)
        value = dict(case=str(case), trace=str(folder / 'trace.json'), steps=len(trace['rows']),
                     sigma=sigma, noise_seed=noise_seed + i, first_fall=summary['first_fall'],
                     seconds=time.monotonic() - start)
        atomic_json(folder / 'completed.json', value)
        return value

    start = time.monotonic()
    # Fail fast on parser/model errors before dispatching a full batch.
    first = run((0, cases[0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [first, *pool.map(run, enumerate(cases[1:], start=1))]
    result = dict(completed=True, rows=rows, elapsed_s=time.monotonic() - start,
                  policy_steps=sum(e['steps'] for e in rows), collection_device='cpu_native_jolt',
                  model_sha256=item['sha256'], workers=workers, accepted=False)
    atomic_json(directory / 'completed.json', result)
    return result, deployment


def dataset(rollouts, deployment, device='cuda'):
    """Action i is paired with physical state i+1, including the final sample.

    Pads finite episodes only for vectorization. Padding and samples after the
    first physical fall do not contribute to any update or bootstrap.
    """
    entries = [e for e in rollouts['rows'] if e['sigma'] > 0]
    if not entries or any(e['sigma'] != .02 for e in entries):
        raise ValueError('Expected stochastic sigma=.02 behavior episodes')
    names = list(templates())
    lengths = np.array([e['steps'] for e in entries])
    steps, count = int(lengths.max()), len(entries)
    shape = (steps, count)
    arrays = {k: np.zeros((*shape, n), np.float32) for k, n in
              [('obs',61), ('action',14), ('previous',14), ('requested',13), ('command',13)]}
    states = {k: np.zeros((steps + 1, count, n), np.float64) for k, n in
              [('position',3), ('quaternion',4), ('velocity',3), ('angular',3)]}
    states['quaternion'][:,:,0] = 1.
    sprint = np.zeros(shape, bool)
    program = []
    last_error = 0.
    final_time_error = 0.
    for n, e in enumerate(entries):
        trace = json.loads(Path(e['trace']).read_text())
        assert trace['summary']['training_exploration'] == e['sigma']
        assert trace['summary']['models'] == {k: v['sha256'] for k, v in deployment['policies'].items()}
        case = json.loads(Path(e['case']).read_text())
        program.append(names.index(case['case'].removeprefix('sprint_')))
        previous = np.zeros(14)
        for t, row in enumerate(trace['rows']):
            assert abs(row['t'] - t*.02) < 1e-7
            for key, source in [('obs','obs'), ('action','action'), ('previous','last_action'),
                                ('requested','requested_command'), ('command','command')]:
                arrays[key][t,n] = row[source]
            last_error = max(last_error, float(np.abs(previous - row['last_action']).max()))
            previous = np.asarray(row['action'])
            body = row['body']
            states['position'][t,n] = body['base_pos']
            states['quaternion'][t,n] = body['base_quat']
            states['velocity'][t,n] = body['base_linvel']
            states['angular'][t,n] = row['raw']['base_angvel_local']
            sprint[t,n] = row['skill'] == 'sprint'
        final = raw_state(trace['summary']['final_raw'], deployment['robots']['walk'])
        end = lengths[n]
        states['position'][end:,n] = final.base_pos
        states['quaternion'][end:,n] = final.base_quat_wxyz
        states['velocity'][end:,n] = final.base_linvel
        states['angular'][end:,n] = final.base_angvel_local
        final_time_error = max(final_time_error, abs(final.t - end*.02))
    assert last_error == 0. and final_time_error < 1e-7
    data = {k: torch.as_tensor(v, device=device) for k,v in arrays.items()}
    state = {k: torch.as_tensor(v, device=device) for k,v in states.items()}
    lengths = torch.as_tensor(lengths, device=device)
    program = torch.as_tensor(program, device=device)
    selection = torch.as_tensor(sprint, device=device)
    control = WalkingControl(count, deployment['control_config']['walk'], device)
    constraints = SprintConstraints(count, device)
    valid = torch.arange(steps, device=device)[:,None] < lengths
    cobs, rewards, costs, fallen = [], [], [], []
    command_error = torch.zeros((), device=device)
    with torch.no_grad():
        for t in range(steps):
            pos, q, vel = [state[k][t] for k in ['position','quaternion','velocity']]
            next_pos, next_q, next_vel, next_ang = [state[k][t+1] for k in ['position','quaternion','velocity','angular']]
            requested = data['requested'][t]
            rebuilt = control.command(requested, pos, q, vel, selection[t])
            command_error = torch.maximum(command_error, (rebuilt-data['command'][t])[valid[t]].abs().max() if valid[t].any() else command_error)
            angle = yaw(q)
            local = torch.stack((angle.cos()*vel[:,0]+angle.sin()*vel[:,1],
                                 -angle.sin()*vel[:,0]+angle.cos()*vel[:,1],vel[:,2]),-1)
            difference = control.yaw_target-angle
            heading = torch.atan2(difference.sin(),difference.cos())
            displacement = pos[:,:2]-control.origin
            path = -control.path_yaw.sin()*displacement[:,0]+control.path_yaw.cos()*displacement[:,1]
            extra = torch.cat((local,pos[:,2:3]/.125,heading[:,None],path[:,None]/.05,
                               selection[t,:,None].float(),(t/lengths)[:,None],
                               ((lengths-t)/1250)[:,None],(program/7)[:,None]),-1)
            cobs.append(torch.cat((data['obs'][t],extra.float()),-1))
            costs.append(constraints.observe(requested,pos,angle,next_pos,next_q,next_vel))
            next_yaw = yaw(next_q)
            tilt = torch.acos((1-2*(next_q[:,1].square()+next_q[:,2].square())).clamp(-1,1))
            next_local = torch.stack((next_yaw.cos()*next_vel[:,0]+next_yaw.sin()*next_vel[:,1],
                                     -next_yaw.sin()*next_vel[:,0]+next_yaw.cos()*next_vel[:,1],next_vel[:,2]),-1)
            progression = command_tracking(data['command'][t],next_local,rotate(next_q,next_ang)[:,2])
            straight = (requested[:,0]>.01)&(requested[:,2].abs()<.05)&(requested[:,1].abs()<.01)
            delta = next_pos[:,:2]-control.origin
            cross = -control.path_yaw.sin()*delta[:,0]+control.path_yaw.cos()*delta[:,1]
            difference = control.yaw_target-next_yaw
            heading = torch.atan2(difference.sin(),difference.cos())
            tracking = torch.where(straight,20*cross.square()+2*heading.square(),0.)
            balance = 8*torch.relu(tilt-.12).square()
            smooth = .05*(data['action'][t]-data['previous'][t]).square().sum(-1)
            fell = (next_pos[:,2]<.055)|(tilt>np.pi/3)
            fallen.append(fell)
            rewards.append((progression-tracking-balance-smooth-10*fell.float()).float())
    error = float(command_error)
    assert error < 1e-5, ('Native control reconstruction',error)
    fall = torch.stack(fallen)&valid
    # Keep the transition into a fall, exclude every transition after it.
    valid &= (fall.long().cumsum(0)-fall.long()) == 0
    done = fall | ((torch.arange(steps,device=device)[:,None]+1)>=lengths)
    done |= ~valid
    data.update(cobs=torch.stack(cobs),reward=torch.stack(rewards),cost=torch.stack(costs).float(),
                done=done,valid=valid,mask=selection&valid)
    data['reward'][~valid]=0.
    data['cost'][~valid]=0.
    audit = dict(episodes=count,rows=int(valid.sum()),actor_rows=int(data['mask'].sum()),
                 discarded_after_fall=int((torch.arange(steps,device=device)[:,None]<lengths).sum()-valid.sum()),
                 falls=int(fall.any(0).sum()),command_max_abs=error,last_action_max_abs=last_error,
                 final_time_max_abs=final_time_error,post_action_reward=True,padding_excluded=True)
    return data,audit
