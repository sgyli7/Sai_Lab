from pathlib import Path
import json,subprocess,time
R=Path(__file__).resolve().parent
base=json.loads((R/'training_base_args.json').read_text())
results=[]
for seed in [952410,952411]:
 for objective in ['off','support']:
  label=f'{objective}_{seed}'
  args=base.copy();args[args.index('--output')+1]=str(R/label);args[args.index('--seed')+1]=str(seed);args[args.index('--swing-objective')+1]=objective
  start=time.monotonic()
  with (R/(label+'.log')).open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=480)
  done=json.loads((R/label/'completed.json').read_text());assert done['status']=='completed' and done['samples']==8388608
  results.append(dict(label=label,wall_s=time.monotonic()-start,**done));(R/'training_progress.json').write_text(json.dumps(results,indent=2)+'\n')
  print(label,done['final_sha256'],done['elapsed_s'],flush=True)
(R/'training_completed.json').write_text(json.dumps(results,indent=2)+'\n')
