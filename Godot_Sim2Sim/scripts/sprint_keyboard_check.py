"""Bounded X11 host-window keyboard integration check (synthetic OS events)."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import shutil
import time


class XKeys:
    def __init__(self):
        self.x=c.CDLL('libX11.so.6');self.xt=c.CDLL('libXtst.so.6')
        signatures={
            'XOpenDisplay':([c.c_char_p],c.c_void_p),
            'XDefaultRootWindow':([c.c_void_p],c.c_ulong),
            'XInternAtom':([c.c_void_p,c.c_char_p,c.c_int],c.c_ulong),
            'XGetWindowProperty':([c.c_void_p,c.c_ulong,c.c_ulong,c.c_long,c.c_long,c.c_int,
                c.c_ulong,c.POINTER(c.c_ulong),c.POINTER(c.c_int),c.POINTER(c.c_ulong),
                c.POINTER(c.c_ulong),c.POINTER(c.POINTER(c.c_ubyte))],c.c_int),
            'XFree':([c.c_void_p],c.c_int),
            'XGetInputFocus':([c.c_void_p,c.POINTER(c.c_ulong),c.POINTER(c.c_int)],c.c_int),
            'XSetInputFocus':([c.c_void_p,c.c_ulong,c.c_int,c.c_ulong],c.c_int),
            'XRaiseWindow':([c.c_void_p,c.c_ulong],c.c_int),
            'XStringToKeysym':([c.c_char_p],c.c_ulong),
            'XKeysymToKeycode':([c.c_void_p,c.c_ulong],c.c_ubyte),
            'XFlush':([c.c_void_p],c.c_int),
            'XCloseDisplay':([c.c_void_p],c.c_int),
        }
        for name,(args,result) in signatures.items():
            function=getattr(self.x,name);function.argtypes=args;function.restype=result
        self.xt.XTestFakeKeyEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
        self.xt.XTestFakeKeyEvent.restype=c.c_int
        self.display=self.x.XOpenDisplay(None)
        if not self.display:raise RuntimeError('An X11 host display is required')
        self.root=self.x.XDefaultRootWindow(self.display);self.held=set();self.window=None

    def property(self,window,name):
        atom=self.x.XInternAtom(self.display,name.encode(),True)
        if not atom:return []
        actual=c.c_ulong();fmt=c.c_int();count=c.c_ulong();remaining=c.c_ulong();data=c.POINTER(c.c_ubyte)()
        status=self.x.XGetWindowProperty(self.display,window,atom,0,4096,False,0,
            c.byref(actual),c.byref(fmt),c.byref(count),c.byref(remaining),c.byref(data))
        try:
            if status or fmt.value!=32:return []
            values=c.cast(data,c.POINTER(c.c_ulong))
            return [values[i] for i in range(count.value)]
        finally:
            if data:self.x.XFree(data)

    def owned_window(self,pid):
        for window in self.property(self.root,'_NET_CLIENT_LIST'):
            if self.property(window,'_NET_WM_PID')==[pid]:return window
        return None

    def focus(self):
        self.x.XRaiseWindow(self.display,self.window)
        self.x.XSetInputFocus(self.display,self.window,1,0);self.x.XFlush(self.display)

    def key(self,name,down):
        current=c.c_ulong();revert=c.c_int()
        self.x.XGetInputFocus(self.display,c.byref(current),c.byref(revert))
        if current.value!=self.window:raise RuntimeError('Owned test window lost focus; stop injecting keys')
        code=self.x.XKeysymToKeycode(self.display,self.x.XStringToKeysym(name.encode()))
        if not code:raise ValueError('No host keycode: '+name)
        if not self.xt.XTestFakeKeyEvent(self.display,code,down,0):raise RuntimeError('XTest rejected the host key event')
        self.x.XFlush(self.display)
        if down:self.held.add(name)
        else:self.held.discard(name)

    def tap(self,name):
        self.key(name,True);time.sleep(.06);self.key(name,False)

    def close(self):
        # Release exactly the keys synthesized by this test, even on failure.
        for name in self.held:
            code=self.x.XKeysymToKeycode(self.display,self.x.XStringToKeysym(name.encode()))
            self.xt.XTestFakeKeyEvent(self.display,code,False,0)
        self.x.XFlush(self.display);self.x.XCloseDisplay(self.display)


def main():
    parser=argparse.ArgumentParser()
    runtime=parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument('--project',type=Path)
    runtime.add_argument('--executable',type=Path,help='Check the actual exported player')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    budget=os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR')
    if not budget or not args.out.resolve().is_relative_to(Path(budget).resolve()):
        raise RuntimeError('Use the bounded session supervisor and an owned output directory')
    args.out.mkdir(parents=True,exist_ok=False);trace=(args.out/'trace.json').resolve()
    if shutil.which('gdbus'):
        locked=subprocess.run(['gdbus','call','--session','--dest','org.gnome.ScreenSaver',
            '--object-path','/org/gnome/ScreenSaver','--method','org.gnome.ScreenSaver.GetActive'],
            capture_output=True,text=True,timeout=3)
        if locked.returncode==0 and '(true,)' in locked.stdout:
            report=dict(check='x11_synthetic_host_keyboard',status='blocked_desktop_locked',passed=False)
            (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit('Desktop is locked; no window or input events were created')
    keys=XKeys();process=None
    try:
        with (args.out/'window.log').open('w') as log:
            command=([str(args.executable.resolve())] if args.executable else
                     ['godot','--path',str(args.project.resolve())])
            command+=['--rendering-method','gl_compatibility','--resolution','1280x800']
            if not args.executable:command+=['res://standalone/main.tscn']
            command+=['--','--render-fps=60',f'--trace={trace}','--seconds=40']
            process=subprocess.Popen(command,
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'SIM2SIM_TRACE_INPUT':'1'})
            end=time.monotonic()+25
            while time.monotonic()<end and process.poll() is None:
                keys.window=keys.owned_window(process.pid)
                if keys.window and 'STANDALONE_READY' in (args.out/'window.log').read_text():break
                time.sleep(.1)
            if not keys.window or process.poll() is not None:raise RuntimeError('Owned Godot window did not become ready')
            keys.focus();time.sleep(.5)
            keys.key('Shift_R',True);keys.key('w',True);time.sleep(.8)
            keys.key('w',False);keys.key('Shift_R',False);keys.tap('0');time.sleep(.3)
            keys.key('Shift_L',True);keys.key('w',True);time.sleep(.8)
            keys.key('a',True);time.sleep(.5);keys.key('a',False)
            keys.key('d',True);time.sleep(.5);keys.key('d',False)
            keys.key('Shift_L',False);time.sleep(.5);keys.key('w',False)
            keys.tap('F8');time.sleep(.4);keys.tap('F8');time.sleep(.3)
            keys.tap('0');time.sleep(.3)
            keys.key('w',True);time.sleep(.5);keys.key('Shift_L',True);time.sleep(.8)
            keys.key('Shift_L',False);keys.key('w',False);keys.tap('0');time.sleep(.3)
            keys.tap('6');time.sleep(.7)
            keys.key('Shift_L',True);keys.key('w',True);time.sleep(.6)
            keys.key('w',False);keys.key('Shift_L',False);keys.tap('6');time.sleep(.7)
            keys.tap('Escape');process.wait(timeout=15)
            if process.returncode:raise RuntimeError('Host application failed')
        data=json.loads(trace.read_text());rows=data['rows']
        first_reset=next((i for i in range(1,len(rows)) if rows[i]['episode_t']<rows[i-1]['episode_t']),len(rows))
        right_phase=rows[:first_reset]
        pause_events=[event for event in data['summary'].get('events',[]) if event['kind']=='pause']
        checks=dict(
            right_shift_ignored=any('fwd' in r['held'] for r in right_phase) and
                all('sprint' not in r['held'] and r['skill']!='sprint' for r in right_phase),
            left_shift_sprint=any(r['skill']=='sprint' for r in rows),
            ordinary_forward=any(r['mode']=='walk' and r['skill']=='walking' and 'fwd' in r['held'] and 'sprint' not in r['held'] for r in rows),
            left_turn=any(r['skill']=='sprint' and r['requested_command'][2]>.1 for r in rows),
            right_turn=any(r['skill']=='sprint' and r['requested_command'][2]<-.1 for r in rows),
            roller_modifier_ignored=any(r['mode']=='roller' and 'sprint' in r['held'] and r['skill']=='roller' for r in rows),
            no_roller_sprint=not any(r['mode']=='roller' and r['skill']=='sprint' for r in rows),
            resets=data['summary']['resets']>=3,switches=data['summary']['switches']>=2,
            pause_resume=len(pause_events)==2 and pause_events[0]['paused'] and not pause_events[1]['paused'] and
                pause_events[0]['step']==pause_events[1]['step'],
            clean_exit=not data['summary'].get('error'))
        report=dict(check='x11_synthetic_host_keyboard',checks=checks,passed=all(checks.values()),
                    rows=len(rows),trace=str(trace),window_pid=process.pid,command=command,
                    runtime='exported_binary' if args.executable else 'prepared_project')
        (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
        if not report['passed']:raise RuntimeError('Host keyboard checks failed')
    finally:
        keys.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


if __name__=='__main__':main()
