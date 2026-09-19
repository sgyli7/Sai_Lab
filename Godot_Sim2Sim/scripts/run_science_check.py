"""Run one integration check under the existing preview/resource lease."""
from pathlib import Path
import os
import json
import signal
import subprocess
import sys
import threading
import time
from showcase_resource_guard import preflight, preview_lease, checkpoint_session, watch, PreviewDeferred

def main():
    root=Path(__file__).resolve().parents[1]
    arguments=sys.argv[1:]
    shared_eval='--share-resources' in arguments
    if shared_eval:
        arguments.remove('--share-resources')
        if len(os.sched_getaffinity(0))>2:
            raise SystemExit('--share-resources requires taskset to at most two CPUs')
    wait_resources="--wait-for-resources" in arguments
    if wait_resources:arguments.remove("--wait-for-resources")
    destination=Path(arguments[arguments.index('--output')+1]) if '--output' in arguments else root/'results/science-station'
    log=destination/'run.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with preview_lease():
        deadline=time.monotonic()+900
        while True:
            try:
                baseline=preflight(30,cpu_only='--headless' in arguments,shared_eval=shared_eval);break
            except PreviewDeferred:
                if not wait_resources or time.monotonic()>=deadline:raise
                time.sleep(15)
        checkpoint_session(baseline)
        baseline['render_fps']=30
        stop=threading.Event(); pauses=[]
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,'-m','sim2sim.workshop',*arguments],cwd=root,
                env=dict(os.environ,SIM2SIM_ROOT=str(root),PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            def pause(reason):
                pauses.append(reason)
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            monitor=threading.Thread(target=watch,args=(stop,baseline,pause),daemon=True);monitor.start()
            try:code=child.wait()
            finally:stop.set();monitor.join(timeout=6)
        print(f'Exit {code}; log: {log}')
        if pauses:raise RuntimeError('; '.join(pauses))
        errors=[line for line in log.read_text().splitlines() if line.startswith(('SCRIPT ERROR:', 'SHADER ERROR:'))]
        if errors:raise RuntimeError(f'{len(errors)} engine script/shader errors: {errors[0]}')
        result=destination/'hub.json'
        if result.exists():
            route=json.loads(result.read_text()).get('route',{})
            if route and not route.get('passed',False):raise RuntimeError(f'Route incomplete: {route}')
        return code
if __name__=='__main__':raise SystemExit(main())
