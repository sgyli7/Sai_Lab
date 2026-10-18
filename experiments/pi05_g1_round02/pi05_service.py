"""Versioned pi05 JSONL service with explicit cold load timing."""
import argparse,base64,contextlib,json,sys,time
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--norm-stats',required=True);p.add_argument('--head-only',action='store_true');a=p.parse_args();start=time.monotonic()
 with contextlib.redirect_stdout(sys.stderr):
  from experiments.pi05_g1.model_runtime import ModelRuntime
  runtime=ModelRuntime(a.checkpoint,a.norm_stats,head_only=a.head_only)
 load=time.monotonic()-start
 for line in sys.stdin:
  req={}
  try:
   req=json.loads(line)
   if req['rgb_shape']!=[480,640,3]:raise ValueError('Original640x480 RGB required')
   rgb=np.frombuffer(base64.b64decode(req['rgb_uint8_b64'],validate=True),np.uint8).reshape(480,640,3)
   with contextlib.redirect_stdout(sys.stderr):r=runtime.infer(rgb,req['state'],req['prompt'],req['seed'])
   reply={'episode_id':req['episode_id'],'control_tick':req['control_tick'],'actions':r['actions'].tolist(),'inference_seconds':r['inference_seconds'],'load_seconds':load,'action_horizon':50}
  except Exception as e:reply={'episode_id':req.get('episode_id'),'control_tick':req.get('control_tick'),'error':repr(e)}
  print(json.dumps(reply,allow_nan=False),flush=True)
if __name__=='__main__':main()
