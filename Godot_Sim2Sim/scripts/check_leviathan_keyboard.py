"""Drive the running Linux game window through real X11 key/focus events."""
import json
import re
from pathlib import Path
import subprocess
import time
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
start=time.monotonic()
while time.monotonic()-start<120:
    if a.log.exists() and 'HUB_ROBOT_READY' in a.log.read_text(errors='replace'):break
    time.sleep(.25)
else:raise TimeoutError('Game readiness')
pid=re.search(r'"pid":(\d+)',a.log.read_text()).group(1)
r=subprocess.check_output(['xdotool','search','--onlyvisible','--pid',pid]).decode().splitlines()
window=r[-1];events=[]
a.output.parent.mkdir(parents=True,exist_ok=True)
def command(*args):
    subprocess.run(['xdotool',*args],check=True,timeout=3)
    events.append(dict(wall=time.monotonic(),args=args))
    a.output.write_text(json.dumps(dict(window=window,events=events),indent=2))
command('windowactivate',window);time.sleep(.3);command('windowfocus',window)
command('key','F8');time.sleep(.5)
command('keydown','w');time.sleep(4)
command('keydown','a');time.sleep(2)
command('keyup','a')
command('windowminimize',window);time.sleep(1)
command('keyup','w')
command('windowmap',window);command('windowactivate',window);time.sleep(.5);command('windowfocus',window);time.sleep(.5)
command('key','F7');time.sleep(1)
command('key','F8');time.sleep(1)
command('keydown','s');time.sleep(4)
command('keydown','space');time.sleep(1)
command('keyup','s');command('keyup','space')
a.output.write_text(json.dumps(dict(window=window,events=events),indent=2))
print('OS_KEYBOARD_EVENTS_COMPLETE',window)
