"""Check the current workshop with actual X11 keyboard events in owned windows."""
from pathlib import Path
import ctypes as c
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import time

ROOT=Path('/home/ethan/Projects/MicroDuck/sim2sim')
R=ROOT/'results/sprint_input_diagnosis_20260914'
spec=importlib.util.spec_from_file_location('keyboard',ROOT/'scripts/sprint_keyboard_check.py')
keyboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(keyboard)

def main():
    out=R/'host_baseline';out.mkdir(exist_ok=False);project=R/'runtime'
    reports={}
    keys=keyboard.XKeys()
    old_focus=c.c_ulong();revert=c.c_int()
    keys.x.XGetInputFocus(keys.display,c.byref(old_focus),c.byref(revert))
    try:
        for name,held in [('forward',['w']),('sprint',['Shift_L','w']),('backward',['s'])]:
            folder=out/name;folder.mkdir()
            options=dict(robot='microduck',task='drive',drive_speed=.5,record=False,output=str(folder),plan={'seconds':12})
            (project/'hub/options.json').write_text(json.dumps(options))
            with (folder/'window.log').open('w') as log:
                cmd=['godot','--path',str(project),'res://hub/main.tscn','--resolution','1280x800','--max-fps','60']
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'SIM2SIM_TRACE_INPUT':'1','SIM2SIM_VISUAL_STYLE':'legacy','MD_MODE':'hub','MD_WORKSHOP_COLLISIONS':'1'})
                try:
                    end=time.monotonic()+20
                    while time.monotonic()<end and child.poll() is None:
                        keys.window=keys.owned_window(child.pid)
                        if keys.window and 'HUB_ROBOT_READY' in (folder/'window.log').read_text():break
                        time.sleep(.05)
                    assert keys.window and child.poll() is None,'Owned workshop not ready'
                    keys.focus();time.sleep(.5)
                    for k in held:keys.key(k,True)
                    time.sleep(3.5)
                    for k in reversed(held):keys.key(k,False)
                    time.sleep(.15);keys.tap('Escape');child.wait(timeout=8)
                    assert child.returncode==0,child.returncode
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            data=json.loads((folder/'native-0.json').read_text())
            move=[v for v in data['rows'] if ('back' if name=='backward' else 'fwd') in v['held']]
            assert move,'No physical movement keys reached workshop'
            rows=[v for v in move if v['episode_t']>=move[0]['episode_t']+1]
            assert len(rows)>60, (name,len(rows))
            a,b=rows[0],rows[-1];dt=b['episode_t']-a['episode_t']
            reports[name]=dict(skills=sorted(set(v['skill'] for v in rows)),held=sorted(set(k for v in rows for k in v['held'])),
                request_x=statistics.mean(v['requested_command'][0] for v in rows),
                velocity_x=(b['body']['base_pos'][0]-a['body']['base_pos'][0])/dt,first_fall=data['first_fall'])
        f,s,b=[reports[k] for k in ['forward','sprint','backward']]
        checks=dict(sprint_model_selected=s['skills']==['sprint'],sprint_faster=s['velocity_x']>f['velocity_x']*1.1,
                    backward_not_faster=abs(b['velocity_x'])<=f['velocity_x']*1.05)
        v=dict(cases=reports,checks=checks,passed=all(checks.values()),input='Synthetic OS X11 events, actual desktop workshop scene and current model/control files')
        (out/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2),flush=True)
        if not v['passed']:raise SystemExit(1)
    finally:
        if old_focus.value not in [0,1]:keys.x.XSetInputFocus(keys.display,old_focus.value,1,0)
        keys.close()

if __name__=='__main__':main()
