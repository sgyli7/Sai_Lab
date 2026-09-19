"""Hold W continuously and press/release left Shift in the actual workshop UI."""
from pathlib import Path
import ctypes as c
import importlib.util
import json
import os
import signal
import subprocess
import time
import numpy as np

ROOT=Path('/home/ethan/Projects/MicroDuck/sim2sim');R=ROOT/'results/sprint_input_diagnosis_20260914'
spec=importlib.util.spec_from_file_location('keyboard',ROOT/'scripts/sprint_keyboard_check.py')
keyboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(keyboard)

def main():
    out=R/'host_transition';out.mkdir(exist_ok=False);project=R/'ui_runtime'
    options=dict(robot='microduck',task='drive',drive_speed=.5,record=True,output=str(out),plan={'seconds':12,'capture_interval':.5})
    (project/'hub/options.json').write_text(json.dumps(options))
    keys=keyboard.XKeys();old=c.c_ulong();revert=c.c_int();keys.x.XGetInputFocus(keys.display,c.byref(old),c.byref(revert));child=None
    try:
        with (out/'window.log').open('w') as log:
            child=subprocess.Popen(['godot','--path',str(project),'res://hub/main.tscn','--resolution','1280x800','--max-fps','60'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'SIM2SIM_TRACE_INPUT':'1','SIM2SIM_VISUAL_STYLE':'legacy','MD_MODE':'hub','MD_WORKSHOP_COLLISIONS':'1'})
            end=time.monotonic()+20
            while time.monotonic()<end and child.poll() is None:
                keys.window=keys.owned_window(child.pid)
                if keys.window and 'HUB_ROBOT_READY' in (out/'window.log').read_text():break
                time.sleep(.05)
            assert keys.window and child.poll() is None,'Owned workshop not ready'
            keys.focus();time.sleep(.5);keys.key('w',True);time.sleep(2)
            keys.key('Shift_L',True);time.sleep(2)
            keys.key('Shift_L',False);time.sleep(2)
            keys.key('w',False);time.sleep(.8);keys.tap('Escape');child.wait(timeout=8)
            assert child.returncode==0
        data=json.loads((out/'native-0.json').read_text());groups=[]
        for row in data['rows']:
            if 'fwd' not in row['held']:continue
            sprint=row['skill']=='sprint'
            if not groups or groups[-1]['sprint']!=sprint:groups.append(dict(sprint=sprint,rows=[]))
            groups[-1]['rows'].append(row)
        assert len(groups)==3 and [v['sprint'] for v in groups]==[False,True,False]
        measures=[]
        for group in groups:
            rows=group['rows'];rows=[v for v in rows if v['episode_t']>=rows[0]['episode_t']+1.]
            assert len(rows)>25
            a,b=rows[0],rows[-1];v=(b['body']['base_pos'][0]-a['body']['base_pos'][0])/(b['episode_t']-a['episode_t'])
            measures.append(dict(sprint=group['sprint'],displacement_speed=float(v),displayed_speed_mean=float(np.mean([v['measured_speed_mps'] for v in rows])),start=a['episode_t'],end=b['episode_t']))
        checks=dict(same_w_hold_shift_switches=True,sprint_faster_than_both=all(measures[1]['displacement_speed']>measures[i]['displacement_speed']*1.1 for i in [0,2]),no_fall=data['first_fall'] is None,
                    speed_display_matches=all(abs(v['displayed_speed_mean']-v['displacement_speed'])<.035 for v in measures))
        v=dict(passed=all(checks.values()),checks=checks,phases=measures,scope='Actual X11 W held continuously; only left Shift pressed/released. Current desktop workshop plus read-only speed HUD. Model and control unchanged.')
        (out/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2));assert v['passed']
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if old.value not in [0,1]:keys.x.XSetInputFocus(keys.display,old.value,1,0)
        keys.close()

if __name__=='__main__':main()
