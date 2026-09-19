"""Measure an exported player's real process memory and native inference latency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from sim2sim.research.queue import atomic_json


def benchmark(executable,output,seconds=1800.,replay=None,timeout=300.,fps=None):
    executable=Path(executable).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    command=[str(executable)]
    if fps is None:command.extend(['--headless','--fixed-fps','200'])
    command.extend(['--',f'--seconds={seconds}'])
    if replay:command.append('--replay='+str(Path(replay).resolve()))
    if fps is not None:command.append(f'--render-fps={fps}')
    samples=[];started=time.monotonic();reason='completed';load_start=list(os.getloadavg())
    with (output/'player.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=timeout:
                    reason='timeout';process.terminate();break
                try:
                    status=Path(f'/proc/{process.pid}/status').read_text()
                    values={k:int(v.strip().split()[0]) for line in status.splitlines()
                            if ':' in line for k,v in [line.split(':',1)] if k in ['VmRSS','VmHWM','Threads']}
                    lifecycle=(output/'player.log').read_text()
                    running='STANDALONE_READY ' in lifecycle and 'STANDALONE_RESULT ' not in lifecycle
                    samples.append(dict(elapsed_s=elapsed,running=running,**values))
                except FileNotFoundError:pass
                time.sleep(.2)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        finally:
            if process.poll() is None:process.kill();process.wait()
    lines=(output/'player.log').read_text().splitlines()
    results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('STANDALONE_RESULT ')]
    rss=[s for s in samples if s.get('VmRSS',0)>0]
    steady=[sample for sample in rss if sample['running']]
    tail=steady[len(steady)//2:]
    slope=float(np.polyfit([s['elapsed_s'] for s in tail],[s['VmRSS'] for s in tail],1)[0]) if len(tail)>2 else None
    result=dict(command=command,returncode=process.returncode,reason=reason,elapsed_s=time.monotonic()-started,
        player=results[-1] if results else None,memory_samples=samples,
        load_average_start=load_start,load_average_end=list(os.getloadavg()),
        peak_rss_kib=max((s.get('VmHWM',0) for s in rss),default=0),
        final_rss_kib=rss[-1]['VmRSS'] if rss else None,tail_rss_slope_kib_per_s=slope,
        steady_rss_min_kib=min((s['VmRSS'] for s in steady),default=None),
        steady_rss_max_kib=max((s['VmRSS'] for s in steady),default=None),
        memory_window='STANDALONE_READY through before STANDALONE_RESULT; tail is the latter half')
    if results:
        player=results[-1];result['simulation_speed']=player['sim_seconds']/player['wall_seconds']
    result['completed']=process.returncode==0 and reason=='completed' and bool(results) and not results[-1]['error']
    atomic_json(output/'metrics.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('executable');p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=1800);p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--replay');p.add_argument('--fps',type=int)
    a=p.parse_args();result=benchmark(a.executable,a.out,a.seconds,a.replay,a.timeout,a.fps)
    print(json.dumps({k:v for k,v in result.items() if k not in ['memory_samples','player']}))
    if not result['completed']:raise SystemExit(1)
