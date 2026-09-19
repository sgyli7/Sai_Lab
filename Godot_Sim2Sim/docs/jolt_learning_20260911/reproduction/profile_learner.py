"""Serial CPU/CUDA pipeline measurement, excluded from model selection."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
from sim2sim.research.budget import live_group_members
p=Path('results/jolt_learning_20260911').resolve()
if not (p/'r3_completed.json').exists() or live_group_members(818888):
    raise RuntimeError('Formal training and evaluation must finish before profiling')
fixed=['--skill','roller','--variant','residual','--source',str(p/'task_anchor.onnx'),
 '--template',str(p/'baseline/roller.onnx'),'--roller-contract','native',
 '--command-gate','negative_throttle','--roller-objective','stop_hold_v1',
 '--teacher-mode','replay_kl','--teacher-replay',str(p/'teacher_replay_complete.json'),
 '--iterations','6','--envs','4','--steps','1024','--minibatch','1024','--epochs','2',
 '--critic-warmup','0','--minutes','6','--conditions','keyboard_roller_brake_3s',
 '--eval-conditions','brake','--eval-seeds','1','--eval-seed-start','917100',
 '--eval-seconds','99999','--std','.02','--freeze-std','10000','--seed','71']
rows=[]
for device in ['cpu','cuda']:
    name='pipeline_profile_'+device
    command=[sys.executable,'-m','sim2sim.research.train','--name',name,'--learner-device',device,*fixed]
    start=time.monotonic()
    with (p/(name+'.log')).open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=420)
    run=p/'runs'/name
    done=json.loads((run/'completed.json').read_text())
    assert done['samples']==6*4096 and done['final_parity']['passed'], done
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    stable=metrics[2:]
    row=dict(device=device,command=command,wall_seconds=time.monotonic()-start,
      config=json.loads((run/'config.json').read_text()),completed=done,
      warmup_updates_excluded=2,measured_updates=len(stable),
      median_collection_s=statistics.median(x['collection_seconds'] for x in stable),
      median_learning_s=statistics.median(x['learner_seconds'] for x in stable),
      median_collector_sync_s=statistics.median(x['collector_sync_seconds'] for x in stable),
      median_update_s=statistics.median(4096/x['fps'] for x in stable))
    rows.append(row)
    (p/'pipeline_profile_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ['config','command']}),flush=True)
result=dict(scope='Four Jolt environments, 4096 transitions/update, six PPO updates, serial CPU then CUDA; first two updates excluded. One timing pair, not a model-quality experiment.',cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),runs=rows,update_speedup=rows[0]['median_update_s']/rows[1]['median_update_s'],learning_speedup=rows[0]['median_learning_s']/rows[1]['median_learning_s'],wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'])
(p/'pipeline_profile_complete.json').write_text(json.dumps(result,indent=2)+'\n')
