"""Record the exported player itself, preserving its original movie and trace."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from sim2sim.research.queue import atomic_json


def record(executable, case, output, timeout=180., fps=60):
    binary=Path(executable).resolve();case=Path(case).resolve();output=Path(output).resolve()
    specification=json.loads(case.read_text())
    if fps not in (30,60):raise ValueError('Movie capture supports 30 or 60 frames per second')
    output.mkdir(parents=True,exist_ok=False)
    movie=output/'capture.avi';trace=output/'trace.json';started=time.monotonic()
    command=[str(binary),'--windowed','--resolution','1280x720','--write-movie',str(movie),
        '--fixed-fps',str(fps),'--disable-vsync','--','--replay='+str(case),'--trace='+str(trace)]
    evidence=dict(executable=str(binary),package_sha256=hashlib.sha256((binary.parent/'SHA256SUMS').read_bytes()).hexdigest(),
        case=str(case),case_sha256=hashlib.sha256(case.read_bytes()).hexdigest(),command=command,
        capture_fps=fps,timeout_seconds=timeout,expected_seconds=specification['seconds'],
        input_source='declared replay; not a human keyboard acceptance',completed=False)
    atomic_json(output/'recording.json',evidence)
    try:
        with (output/'player.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        evidence['returncode']=process.returncode
        if process.returncode:raise RuntimeError('Movie player exited with '+str(process.returncode))
        summary=json.loads(trace.read_text())['summary']
        if summary['error']:raise RuntimeError(summary['error'])
        if abs(float(summary['sim_seconds'])-float(specification['seconds']))>.021:
            raise RuntimeError('Movie stopped before the declared replay duration')
        encode=['ffmpeg','-nostdin','-i',str(movie),'-an','-c:v','libx264','-threads','2',
            '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'demo.mp4')]
        with (output/'encode.log').open('w') as log:
            subprocess.run(encode,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
        evidence.update(completed=True,player=summary,encode_command=encode,
            artifacts={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                       for name in ['capture.avi','demo.mp4','trace.json']})
        return evidence
    except Exception as error:
        evidence['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        evidence['elapsed_s']=time.monotonic()-started
        atomic_json(output/'recording.json',evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable');parser.add_argument('case');parser.add_argument('--out',required=True)
    parser.add_argument('--timeout',type=float,default=180.);parser.add_argument('--fps',type=int,default=60)
    args=parser.parse_args()
    result=record(args.executable,args.case,args.out,args.timeout,args.fps)
    print(json.dumps({key:result[key] for key in ['completed','elapsed_s','artifacts']}))
