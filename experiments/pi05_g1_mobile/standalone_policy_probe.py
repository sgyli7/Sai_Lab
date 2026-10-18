"""One real native-service call, no simulator imports or action execution.

PolicyProcess is copied verbatim from round02 simulator IPC, not imported from
that simulator module. Root alone launches this through GPU budget and FIFO.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,selectors,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import PROMPT,encode_state

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='pi05_g1_mobile_standalone_probe_v1'
NATIVE_PYTHON=Path('/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class PolicyProcess:
    """One isolated model process; synchronous calls pause simulator physics."""
    def __init__(self, command, stderr_path, timeout_seconds=180):
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('policy_command must be an explicit argv array')
        self.stderr = Path(stderr_path).open('x')
        # Isaac's interpreter exports its own Python3.12/Kit/CUDA library paths.
        # The policy uses its independent Python3.11 venv and cu13 wheels.
        child_env = dict(os.environ)
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONHOME', 'PYTHONSTARTUP'):
            child_env.pop(key, None)
        repo = Path(__file__).resolve().parents[2]
        child_env['PYTHONPATH'] = os.pathsep.join((str(repo), str(repo / '.scratch/pi05_g1/openpi/src'),
                                                 str(repo / '.scratch/pi05_g1/openpi/packages/openpi-client/src')))
        child_env['JAX_PLATFORMS'] = 'cpu'
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1, env=child_env)
        self.timeout = timeout_seconds

    def infer(self, rgb, state, case, tick, prompt, q43, names):
        import numpy as np
        request = {'schema': 'pi05_g1_observation_v1', 'episode_id': case['case_id'],
                   'control_tick': tick, 'rgb_shape': list(rgb.shape),
                   'rgb_uint8_b64': base64.b64encode(rgb.tobytes()).decode(),
                   'state': np.asarray(state).tolist(), 'prompt': prompt, 'seed': case['seed'] + tick,
                   'measured_joint_positions43': np.asarray(q43).tolist(), 'measured_joint_names43': names}
        started = time.monotonic()
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError('Model inference deadline exceeded while simulation paused')
        line = self.process.stdout.readline(1_048_577)
        if not line or len(line) > 1_048_576:
            raise ValueError('Model disconnected or exceeded reply bound')
        reply = json.loads(line)
        if reply.get('error'):
            raise RuntimeError(reply['error'])
        if reply.get('episode_id') != case['case_id'] or reply.get('control_tick') != tick:
            raise ValueError('Model reply changed observation identity')
        actions = np.asarray(reply['actions'], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 32 or not 1 <= len(actions) <= 50 or not np.isfinite(actions).all():
            raise ValueError('Model must return one to fifty finite 32-dimensional actions')
        return actions, {'inference_seconds': reply['inference_seconds'],
                         'simulation_paused_wall_seconds': time.monotonic() - started,
                         'observation_control_tick': tick, 'request_started_monotonic_ns': int(started*1e9),
                         'reply_received_monotonic_ns': time.monotonic_ns(), 'model_load_seconds': reply.get('load_seconds'),
                         'stage_seconds': reply.get('stage_seconds'), 'action_horizon': len(actions)}

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()

def run(output,observation_dir,policy='native',timeout_seconds=90,*,process_class=PolicyProcess):
    output=Path(output).resolve();observation_dir=Path(observation_dir).resolve()
    if output.exists():raise FileExistsError('Fresh output required; never overwrite probe evidence')
    if policy!='native':raise ValueError('Only native service supported')
    output.mkdir(parents=True);started=time.monotonic();process=None;error=None;actions=None;timing=None;close_error=None
    report={'schema':SCHEMA,'status':'failed','policy':'native','seed':201,'output':str(output),'simulator_imported':False,'actions_executed':False,'task_success_evaluated':False,'service_call_attempts':0}
    try:
        rgb_path=observation_dir/'head_00000.png';state_path=observation_dir/'observation_00000.json'
        observation=json.loads(state_path.read_text())
        allowed={'control_tick','state','measured_joint_positions43','measured_joint_names43','prompt','rgb_sha256'}
        if set(observation)!=allowed or observation['control_tick']!=0 or observation['prompt']!=PROMPT:
            raise ValueError('Exact real tick0 measured observation required, no truth inputs')
        with Image.open(rgb_path) as image:
            if image.mode!='RGB':raise ValueError('Original RGB image required')
            rgb=np.asarray(image).copy()
        if rgb.shape!=(480,640,3) or hashlib.sha256(rgb.tobytes()).hexdigest()!=observation['rgb_sha256']:
            raise ValueError('RGB identity/shape mismatch')
        state=np.asarray(observation['state'],np.float32);q=np.asarray(observation['measured_joint_positions43'],np.float32);names=observation['measured_joint_names43']
        if state.shape!=(32,) or q.shape!=(43,) or not np.isfinite(state).all() or not np.isfinite(q).all() or not np.array_equal(state,encode_state(q,names)):
            raise ValueError('State must equal named measured43 encoding')
        command=[str(NATIVE_PYTHON),'-m','experiments.pi05_g1_mobile.gr00t_service','--seed','201','--decoded-evidence',str(output/'native_decoded.jsonl')]
        report.update(service_command=command,input_sha256={str(rgb_path):sha(rgb_path),str(state_path):sha(state_path)},rgb_sha256=observation['rgb_sha256'],source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'experiments/pi05_g1_mobile/gr00t_service.py',ROOT/'experiments/pi05_g1_mobile/contract.py',ROOT/'experiments/pi05_g1_round02/simulator.py']})
        process=process_class(command,output/'policy.stderr.log',timeout_seconds)
        report['service_call_attempts']=1
        actions,timing=process.infer(rgb,state,{'case_id':output.name,'seed':201},0,PROMPT,q,names)
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(50,32) or not np.isfinite(actions).all():raise ValueError('Probe requires exactly finite50x32 actions')
        np.save(output/'actions.npy',actions,allow_pickle=False)
        report.update(actions_path=str(output/'actions.npy'),actions_sha256=sha(output/'actions.npy'),actions_shape=[50,32],timing=timing)
    except BaseException as caught:error=caught;report['error']=repr(caught)
    finally:
        if process is not None:
            try:process.close()
            except BaseException as caught:close_error=caught;report['service_close_error']=repr(caught)
            returncode=process.process.poll();report['service_exit_code']=returncode
            report['service_closed_cleanly']=close_error is None and returncode==0
        else:report['service_closed_cleanly']=False
        report['elapsed_seconds']=time.monotonic()-started
        if error is None and report['service_closed_cleanly']:report['status']='completed'
        elif error is None:report['error']='Service did not close cleanly'
        stderr=output/'policy.stderr.log'
        if stderr.exists():report['policy_stderr_sha256']=sha(stderr)
        with (output/'probe_result.json').open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--observation-dir',required=True)
    parser.add_argument('--policy',choices=['native'],default='native');parser.add_argument('--timeout-seconds',type=float,default=90)
    args=parser.parse_args()
    if not 0<args.timeout_seconds<=90:parser.error('Service timeout must be in(0,90], leaving close margin inside120s budget')
    result=run(args.output,args.observation_dir,args.policy,args.timeout_seconds)
    print(json.dumps({'schema':SCHEMA,'status':result['status'],'output':str(Path(args.output).resolve())}))
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
