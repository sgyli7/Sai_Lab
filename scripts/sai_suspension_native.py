"""Independent native Jolt validation of frozen suspension parameters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
from sai_agent.paths import resource_root
from sai_agent.cli import prepare_godot
from sim2sim.sai_controller import MotionController
from sim2sim.sai_suspension import Suspension
from sim2sim.sai_timebase import DEFAULT_PHYSICS_HZ, SUPPORTED_PHYSICS_HZ, configure_project_timebase, patch_generated_runtime
from sai_suspension_experiment import Terrain,metrics,ROOT


def run(out,seed=47,kind='rough',parameters=None,mode='gate',riser=0.,descending=False,speed=.5,yaw=0.,crouch=0.,maneuver="drive",physics_hz=DEFAULT_PHYSICS_HZ):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    runtime=out/'runtime';prepare_godot(resource_root(),runtime)
    patch_generated_runtime(runtime);configure_project_timebase(runtime/'project.godot',physics_hz)
    shutil.copyfile(ROOT/'scripts/sai_suspension_probe.gd',runtime/'probe.gd')
    shutil.copyfile(ROOT/'godot/sai/terrain_scan.gd',runtime/'terrain_scan.gd')
    terrain=Terrain(seed,kind)
    (runtime/'experiment.json').write_text(json.dumps(dict(terrain=dict(x=terrain.x.tolist(),y=terrain.y.tolist(),z=terrain.z.tolist()),speed=speed,crouch=crouch,maneuver=maneuver)))
    scene=(runtime/'main.tscn').read_text().replace('res://main.gd','res://probe.gd')
    (runtime/'probe.tscn').write_text(scene)
    project=runtime/'project.godot';project.write_text(project.read_text()+'\n[threading]\nworker_pool/max_threads=2\n')
    godot=shutil.which('godot')
    c=MotionController(resource_root());c.suspension=Suspension(parameters) if parameters is not None else None
    if mode=='original':
        c._step_in_wheel_path=lambda state,honor_course=True:bool(np.ptp(state['terrain_path_heights'])>.004)
    with socket.socket() as listener,ThreadPoolExecutor(max_workers=1) as pool:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(.25)
        stop=threading.Event();service=pool.submit(c.serve,listener,stop)
        command=[godot,'--headless','--fixed-fps',str(physics_hz),'--path',str(runtime),'res://probe.tscn','--',
                 f'--port={listener.getsockname()[1]}','--no-visuals','--case=W',f'--duration={45 if riser else 7}',
                 f'--output={out / "raw.json"}',f'--stairs={riser}',f'--initial-yaw={yaw}']
        if descending:command.append('--descending')
        try:
            with (out/'engine.log').open('w') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=150)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
            if code:raise RuntimeError(f'Jolt process exited {code}: {out / "engine.log"}')
        finally:stop.set()
        service.result(timeout=5)
    if 'SCRIPT ERROR:' in (out/'engine.log').read_text():raise RuntimeError('Godot script error')
    raw=json.loads((out/'raw.json').read_text());rows=[]
    for s in raw['samples']:
        rotation=np.asarray(s['base_rotation_columns']).T
        ground=np.array(s['wheel_ground_heights']);wheel=np.array(s['wheel_positions'])
        rows.append(dict(time=s['time'],position=s['base_position'],velocity=(rotation.T@np.array(s['base_linear_world'])).tolist(),
            angular=(rotation.T@np.array(s['base_angular_world'])).tolist(),upright=s['upright'],gap=(wheel[:,2]-ground-.048).tolist(),
            supported=s['wheels_supported'],stage=s['controller_stage'],command=s['command'],effective_speed=s['policy_observation'][9]))
    report=dict(seed=seed,kind=kind,parameters=parameters,mode=mode,riser=riser,descending=descending,
                physics='Godot/Jolt',physics_hz=physics_hz,controller_hz=50,
                metrics=metrics(rows),engine=raw['engine'])
    if riser:
        last=raw['samples'][-1];settled=raw['samples'][-50:]
        report['stairs_passed']=bool(np.min(np.array(last['wheel_positions'])[:,0])>.45+3*.18+.05 and raw['cleared_at']>=0
            and last['time']-raw['cleared_at']>=2.999 and min(s['upright'] for s in raw['samples'])>.6
            and max(abs(s['base_position'][1]) for s in raw['samples'])<.3
            and np.mean([s['wheels_supported']==4 for s in settled])>.7)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--kind',default='rough');p.add_argument('--profile',type=Path);p.add_argument('--mode',default='gate')
    p.add_argument('--riser',type=float,default=0.);p.add_argument('--descending',action='store_true')
    p.add_argument('--physics-hz',type=int,choices=SUPPORTED_PHYSICS_HZ,default=DEFAULT_PHYSICS_HZ)
    a=p.parse_args();parameters=json.loads(a.profile.read_text())['parameters'] if a.profile else None
    run(a.out,a.seed,a.kind,parameters,a.mode,a.riser,a.descending,physics_hz=a.physics_hz)

if __name__=='__main__':main()
