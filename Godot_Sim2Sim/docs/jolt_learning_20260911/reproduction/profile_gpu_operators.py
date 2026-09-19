"""Operator attribution only; concurrent native validation prevents throughput claims."""
import json
from pathlib import Path
import torch
from sim2sim.research.models import Policy
from sim2sim.research.teacher_retention import TeacherRetention,phase_sample
p=Path('results/jolt_learning_20260911')
torch.set_num_threads(2)
policy=Policy(p/'task_anchor.onnx','residual',template=p/'baseline/roller.onnx',command_gate='negative_throttle').cuda()
r=TeacherRetention(policy,'replay_kl',p/'teacher_replay_complete.json').to('cuda')
for _ in range(5):
 policy.zero_grad();r.loss(policy,r.reference).backward()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
 for _ in range(5):
  policy.zero_grad()
  with torch.profiler.record_function('teacher_filter_and_sample'):
   x=phase_sample(r.reference,r.samples)
  with torch.profiler.record_function('actor_forward_backward'):
   policy.delta(x).square().mean().backward()
 torch.cuda.synchronize()
rows=[dict(operator=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,self_device_us=getattr(e,'self_device_time_total',0),cpu_us=e.cpu_time_total,device_us=getattr(e,'device_time_total',0)) for e in profile.key_averages()]
(p/'gpu_operator_profile.json').write_text(json.dumps(dict(scope=__doc__,steps=5,operators=sorted(rows,key=lambda e:e['self_cpu_us'],reverse=True)),indent=2)+'\n')
print(json.dumps(rows[:8]))
