"""Freeze the candidate before independent terrain validation; never select on this file."""
import argparse,json
from pathlib import Path
from sai_suspension_experiment import rollout
from sai_suspension_native import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--native',action='store_true');a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);parameters=json.loads(a.profile.read_text())['parameters'];results=[]
 cases=[('rough',211),('rough',307),('rough',401),('ramp',503),('cross',607),('flat',709)]
 for kind,seed in cases:
  for mode in ['original','gate','suspension']:
   params=parameters if mode=='suspension' else None
   report=run(a.out/f'{kind}-{seed}-{mode}',seed,kind,params,mode) if a.native else rollout(seed,kind,params,mode,trace=True)
   if not a.native:(a.out/f'{kind}-{seed}-{mode}.json').write_text(json.dumps(report)+'\n')
   compact={k:v for k,v in report.items() if k!='rows'};results.append(compact)
   (a.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
   if not a.native:print(json.dumps(compact),flush=True)

if __name__=='__main__':main()
