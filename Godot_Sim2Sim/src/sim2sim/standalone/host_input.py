"""Bounded X11 keyboard acceptance using only the newly launched player's window."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import re
import subprocess
import time

from sim2sim.paths import sim2sim_root


def desktop_locked():
    try:
        result=subprocess.run(['gdbus','call','--session','--dest','org.gnome.ScreenSaver',
            '--object-path','/org/gnome/ScreenSaver','--method','org.gnome.ScreenSaver.GetActive'],
            capture_output=True,text=True,timeout=3)
    except (FileNotFoundError,subprocess.TimeoutExpired):return None
    if result.returncode:return None
    return result.stdout.strip()=='(true,)'


def run(output, seconds=90, fps=60, probe=False, transport='xtest', executable=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    locked=desktop_locked()
    if transport=='xtest' and locked is not False:
        (output/'input_events.json').write_text(json.dumps(dict(status='blocked',reason='desktop_locked_or_unknown',events=[])))
        raise RuntimeError('Global keyboard validation requires a confirmed unlocked desktop')
    x=c.CDLL('libX11.so.6');xt=c.CDLL('libXtst.so.6')
    x.XOpenDisplay.restype=c.c_void_p
    x.XSetInputFocus.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_ulong]
    x.XGetInputFocus.argtypes=[c.c_void_p,c.POINTER(c.c_ulong),c.POINTER(c.c_int)]
    x.XRaiseWindow.argtypes=[c.c_void_p,c.c_ulong];x.XFlush.argtypes=[c.c_void_p]
    x.XStringToKeysym.argtypes=[c.c_char_p];x.XStringToKeysym.restype=c.c_ulong
    x.XKeysymToKeycode.argtypes=[c.c_void_p,c.c_ulong];x.XKeysymToKeycode.restype=c.c_ubyte
    x.XQueryKeymap.argtypes=[c.c_void_p,c.c_void_p]
    xt.XTestFakeKeyEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
    xt.XTestFakeMotionEvent.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_ulong]
    xt.XTestFakeButtonEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
    class KeyEvent(c.Structure):
        _fields_=[('type',c.c_int),('serial',c.c_ulong),('send_event',c.c_int),
            ('display',c.c_void_p),('window',c.c_ulong),('root',c.c_ulong),('subwindow',c.c_ulong),
            ('time',c.c_ulong),('x',c.c_int),('y',c.c_int),('x_root',c.c_int),('y_root',c.c_int),
            ('state',c.c_uint),('keycode',c.c_uint),('same_screen',c.c_int)]
    x.XSendEvent.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_long,c.c_void_p]
    x.XDefaultRootWindow.argtypes=[c.c_void_p];x.XDefaultRootWindow.restype=c.c_ulong
    display=x.XOpenDisplay(None)
    if not display:raise RuntimeError('An X11 host display is required')
    environment=os.environ.copy();environment['SIM2SIM_TRACE_INPUT']='1'
    environment['SIM2SIM_SHOT']=str(output/'startup.png')
    log=(output/'player.log').open('w')
    command=[str(Path(executable).resolve())] if executable else ['godot','--path',str(sim2sim_root()/'godot'),'res://standalone/main.tscn']
    process=subprocess.Popen([*command,'--',f'--seconds={seconds}',f'--render-fps={fps}',f'--trace={output/"trace.json"}'],
        env=environment,stdout=log,stderr=subprocess.STDOUT)
    held=set();events=[];event_monitor=None;event_log=None;validation=None
    def focus():
        current=c.c_ulong();revert=c.c_int()
        x.XGetInputFocus(display,c.byref(current),c.byref(revert));return current.value
    def key(name,on):
        if transport=='xtest' and on and desktop_locked() is not False:
            raise RuntimeError('Desktop lock state changed; keyboard injection stopped')
        x.XSetInputFocus(display,window,1,0);x.XFlush(display)
        if focus()!=window:raise RuntimeError(f'Player lost focus ({focus()} != {window}); input injection stopped')
        code=x.XKeysymToKeycode(display,x.XStringToKeysym(name.encode()))
        if transport=='xtest':
            ok=xt.XTestFakeKeyEvent(display,code,int(on),0)
        else:
            event=KeyEvent(type=2 if on else 3,display=display,window=window,
                root=x.XDefaultRootWindow(display),keycode=code,same_screen=1)
            ok=x.XSendEvent(display,window,0,1 if on else 2,c.byref(event))
        x.XFlush(display)
        if on:held.add(name)
        else:held.discard(name)
        time.sleep(.04)
        bitmap=(c.c_ubyte*32)();x.XQueryKeymap(display,bitmap)
        events.append(dict(unix=time.time(),key=name,code=code,pressed=on,
                           transport=transport,send_result=ok,server_pressed=bool(bitmap[code//8]&(1<<(code%8))),focus=focus()))
    def tap(name):key(name,True);key(name,False)
    try:
        deadline=time.monotonic()+15;window=None
        while time.monotonic()<deadline and process.poll() is None:
            tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
            for line in tree.splitlines():
                if 'Microduck Sim2Sim' not in line:continue
                match=re.search(r'0x[0-9a-fA-F]+',line)
                if not match:continue
                value=subprocess.run(['xprop','-id',match[0],'_NET_WM_PID'],capture_output=True,text=True)
                if re.search(r'=\s*'+str(process.pid)+r'\s*$',value.stdout):window=int(match[0],16);break
            if window and 'STANDALONE_READY' in (output/'player.log').read_text():break
            time.sleep(.1)
        if not window:raise RuntimeError('Player window not found')
        event_log=(output/'xev.log').open('w')
        event_monitor=subprocess.Popen(['stdbuf','-oL','xev','-id',hex(window),'-event','keyboard'],stdout=event_log,stderr=subprocess.STDOUT)
        time.sleep(1)
        x.XRaiseWindow(display,window);x.XSetInputFocus(display,window,1,0);x.XFlush(display)
        geometry=subprocess.check_output(['xwininfo','-id',hex(window)],text=True)
        xpos=int(re.search(r'Absolute upper-left X:\s*(-?\d+)',geometry)[1])
        ypos=int(re.search(r'Absolute upper-left Y:\s*(-?\d+)',geometry)[1])
        # Avoid relying on compositor window raising: X11 focus must be verified
        # before each key, even when another desktop task has a visible window.
        time.sleep(.2)
        key('w',True);time.sleep(2);key('w',False);time.sleep(1)
        if not probe:
            tap('7');time.sleep(1);tap('F8');time.sleep(2);tap('F8');time.sleep(.5)
            tap('g');time.sleep(4.3);tap('k');time.sleep(5.3)
            tap('l');time.sleep(5.3);tap('r');time.sleep(5.3)
            tap('y');time.sleep(4.);tap('y');time.sleep(4.)
            tap('6');time.sleep(1.);key('w',True);time.sleep(3);key('w',False)
            key('s',True);time.sleep(3.);key('s',False);time.sleep(1)
            tap('y');time.sleep(5.3);tap('0');time.sleep(.5);tap('6');time.sleep(.7)
        tap('Escape')
        process.wait(timeout=10)
        trace=json.loads((output/'trace.json').read_text())
        skills=sorted({row['skill'] for row in trace['rows']})
        expected={'standing','walking','sitstand','ground_pick','kick_left','kick_right','roulade','roller','roller_crouch'}
        pauses=[event for event in trace['summary']['events'] if event['kind']=='pause']
        pause_ok=len(pauses)==2 and pauses[0]['step']==pauses[1]['step'] and pauses[0]['episode_t']==pauses[1]['episode_t']
        passed=process.returncode==0 and not trace['summary']['error']
        if not probe:passed=passed and set(skills)==expected and pause_ok and trace['summary']['resets']==1 and trace['summary']['switches']==2
        validation=dict(passed=passed,skills=skills,pause_stopped_physics=pause_ok,
            scope='global keyboard path' if transport=='xtest' else 'window-directed events; not global keyboard acceptance')
        result=dict(pid=process.pid,window=window,returncode=process.returncode,events=events,
                    desktop_locked_on_start=locked,transport=transport,validation=validation)
        (output/'input_events.json').write_text(json.dumps(result,indent=2))
        if not passed:raise RuntimeError('Keyboard sequence did not cover the required controls; inspect input_events.json')
        return result
    finally:
        (output/'input_events.json').write_text(json.dumps(dict(pid=process.pid,
            window=locals().get('window'),returncode=process.poll(),events=events,
            desktop_locked_on_start=locked,transport=transport,validation=validation),indent=2))
        # Release only keys we injected, even if the test aborted.
        for name in held:
            code=x.XKeysymToKeycode(display,x.XStringToKeysym(name.encode()))
            if transport=='xtest':xt.XTestFakeKeyEvent(display,code,0,0)
            elif process.poll() is None:
                event=KeyEvent(type=3,display=display,window=window,
                    root=x.XDefaultRootWindow(display),keycode=code,same_screen=1)
                x.XSendEvent(display,window,0,2,c.byref(event))
        x.XFlush(display)
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()
        if event_monitor is not None:
            event_monitor.terminate();event_monitor.wait(timeout=3);event_log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output')
    parser.add_argument('--fps',type=int,default=60)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--transport',choices=['xtest','send-event'],default='xtest')
    parser.add_argument('--executable',type=Path)
    a=parser.parse_args();result=run(a.output,fps=a.fps,seconds=8 if a.probe else 90,probe=a.probe,transport=a.transport,executable=a.executable)
    print(json.dumps({k:v for k,v in result.items() if k!='events'}))
