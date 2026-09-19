"""Native Jolt check for MuJoCo-trained impedance and an unconstrained cargo body."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import numpy as np
from sai_agent.paths import resource_root
from sai_agent.cli import prepare_godot
from sim2sim.sai_compliance import CompliantController
from sai_loaded_mujoco import LoadedTerrain
from sim2sim.sai_timebase import (
    DEFAULT_PHYSICS_HZ,
    SUPPORTED_PHYSICS_HZ,
    configure_project_timebase,
    patch_generated_runtime,
    validate_physics_hz,
)

ROOT=Path(__file__).resolve().parents[1]


def run(out,kind='rough',seed=47,mass=.1,parameters=None,clamped=False,drive_speed=.5,turn_rate=0.,physics_hz=DEFAULT_PHYSICS_HZ,
        cargo_belt_stiffness_npm=20000.,cargo_belt_damping_ns_m=4.):
    physics_hz=validate_physics_hz(physics_hz)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);runtime=out/'runtime';prepare_godot(resource_root(),runtime)
    patch_generated_runtime(runtime);configure_project_timebase(runtime/'project.godot',physics_hz)
    robot_path=runtime/'robot.gd';robot_text=robot_path.read_text()
    robot_text=robot_text.replace('20000.0*(ratio*s.q[24]-s.q[i])',f'{float(cargo_belt_stiffness_npm)}*(ratio*s.q[24]-s.q[i])')
    robot_text=robot_text.replace('4.0*(ratio*s.v[24]-s.v[i])',f'{float(cargo_belt_damping_ns_m)}*(ratio*s.v[24]-s.v[i])')
    robot_path.write_text(robot_text)
    (runtime/'sai').mkdir(exist_ok=True)
    for file in ['impedance.gd','compliant_robot.gd','terrain_scan.gd']:shutil.copyfile(ROOT/'godot/sai'/file,runtime/'sai'/file)
    shutil.copyfile(ROOT/'scripts/sai_loaded_probe.gd',runtime/'probe.gd')
    (runtime/'probe.tscn').write_text((runtime/'main.tscn').read_text().replace('res://main.gd','res://probe.gd'))
    project=runtime/'project.godot';project.write_text(project.read_text()+'\n[threading]\nworker_pool/max_threads=2\n')
    terrain=LoadedTerrain(seed,kind,mass);initial=float(terrain.query(np.array([[0.,0.]]))[0]);stairs=kind.startswith(('up','down','mixed'))
    config=dict(terrain=dict(x=terrain.x.tolist(),y=terrain.y.tolist(),z=terrain.z.tolist()),mass=mass,initial_ground=initial,
                drive_speed=drive_speed,turn_rate=turn_rate,stair_start=1.1 if kind.startswith('mixed') else .45,clamped=clamped)
    (runtime/'experiment.json').write_text(json.dumps(config)+'\n')
    riser=(.04 if '40' in kind else .02) if stairs else 0.;c=CompliantController(resource_root(),parameters)
    with socket.socket() as listener,ThreadPoolExecutor(max_workers=1) as pool:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(.25);stop=threading.Event();service=pool.submit(c.serve,listener,stop)
        cmd=['godot','--headless','--fixed-fps',str(physics_hz),'--path',str(runtime),'res://probe.tscn','--',f'--port={listener.getsockname()[1]}',
             '--no-visuals','--case=loaded',f'--duration={35 if stairs else 8}',f'--stairs={riser}',f'--output={out / "raw.json"}']
        if kind.startswith('down'):cmd.append('--descending')
        try:
            wall_start=time.monotonic()
            with (out/'engine.log').open('w') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=180)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
            wall_seconds=time.monotonic()-wall_start
            if code:raise RuntimeError(f'Jolt exit {code}; inspect {out / "engine.log"}')
        finally:stop.set()
        service.result(timeout=5)
    if 'SCRIPT ERROR:' in (out/'engine.log').read_text():raise RuntimeError('Godot script errors')
    raw=json.loads((out/'raw.json').read_text());cargo=json.loads((out/'cargo-physics.json').read_text());a=np.array(cargo['samples'])
    cleared=raw['cleared_at'] if raw['cleared_at']>=0 else None
    moving=a[a[:,0]<(cleared or 7.)];ss=[r for r in raw['samples'] if r['time']>=1.5 and r['command'][0]>0]
    sm=np.linalg.norm(moving[:,4:7],axis=1);j=np.linalg.norm(moving[:,7:10],axis=1);acc=np.linalg.norm(moving[:,1:4],axis=1)
    m=dict(cargo_accel_rms=float(np.sqrt(np.mean(sm**2))),cargo_accel_p95=float(np.quantile(sm,.95)),cargo_accel_peak=float(acc.max()),
        cargo_jerk_rms=float(np.sqrt(np.mean(j**2))),deck_accel_rms=float(np.sqrt(np.mean(np.linalg.norm(moving[:,10:13],axis=1)**2))),
        contact_impulse_peak_Ns=cargo['contact_impulse_peak_Ns'],max_cargo_slip_m=cargo['max_slip'],cargo_lost=cargo['cargo_lost'],min_upright=float(a[:,13].min()),
        height_error_mean=float(moving[:,17].mean()),height_error_p95=float(np.quantile(abs(moving[:,17]),.95)),cargo_supported_fraction=float(moving[:,18].mean()),bilateral_clamp_fraction=float(moving[:,19].mean()),
        speed=float(np.mean([r['policy_observation'][3] for r in ss])),distance=raw['samples'][-1]['base_position'][0],cleared_at=cleared,
        completed=bool(cleared is not None if stairs else raw['samples'][-1]['time']>=7.999),duration=raw['samples'][-1]['time'])
    report=dict(drive_speed=drive_speed,turn_rate=turn_rate,kind=kind,seed=seed,mass=mass,clamped=clamped,parameters=parameters,metrics=m,engine=raw['engine'],physics_hz=physics_hz,controller_hz=50,
        cargo_belt_stiffness_npm=cargo_belt_stiffness_npm,cargo_belt_damping_ns_m=cargo_belt_damping_ns_m,
        control_decimation=physics_hz//50,wall_seconds=wall_seconds,sim_to_wall_ratio=float(m['duration']/wall_seconds))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'physics.npz',samples=a)
    print(json.dumps(report),flush=True);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',default='rough');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--mass',type=float,default=.1);p.add_argument('--profile',type=Path);p.add_argument('--clamped',action='store_true')
    p.add_argument('--physics-hz',type=int,choices=SUPPORTED_PHYSICS_HZ,default=DEFAULT_PHYSICS_HZ);a=p.parse_args()
    parameters=json.loads(a.profile.read_text())['parameters'] if a.profile else None
    run(a.out,a.kind,a.seed,a.mass,parameters,a.clamped,physics_hz=a.physics_hz)

if __name__=='__main__':main()
