from pathlib import Path
import json
import shutil
import time
from sim2sim.research.budget import ActiveBudget
from sim2sim.research.queue import atomic_json

R=Path('results/sprint_observability_20260913')
D=Path('docs/sprint_observability_20260913')
resources=json.loads((R/'closeout/resources.json').read_text())
tests=json.loads((R/'closeout/tests.json').read_text())
defaults=json.loads((R/'closeout/defaults_and_physics.json').read_text())
assert resources['passed'] and not resources['all_godot_processes']
assert resources['started']==resources['finished']==12
assert tests['successful'] and defaults['passed']
status=ActiveBudget(R).heartbeat('agent',stop=True,max_gap=1800)
assert not status['actors'] and status['unconfirmed_gaps']==0
for source,destination in [(R/'active_budget.json',D/'evidence/active_budget.json'),
                           (R/'closeout/resources.json',D/'evidence/closeout/resources.json'),
                           (Path(__file__),D/'harness/seal.py')]:
    shutil.copy2(source,destination)
for p in (R/'supervisors').glob('*.log'):
    shutil.copy2(p,D/'evidence/supervisors'/p.name)
artifact=json.loads((R/'closeout/local_artifacts.json').read_text())
result=dict(sealed_unix=time.time(),session=str(R),base_commit='32097ebded91d0772308701bce14d4f8fb66f521',
    status='research_completed_goal_incomplete',goal_complete=False,policy_promoted=False,new_package=False,
    current_sprint_baseline=dict(sha256='a402791e30e8ec9ffba5a288a2ebc377e43e7b6925b49986a3c523c686665770',
        sprint_pass=127,sprint_total=128,ordinary_pass=128,ordinary_total=128,regression_pass=7,falls=0),
    final_seeds_unused='928000–928049',keyboard_revalidated=False,
    reproduction=json.loads((R/'reproduce/completed.json').read_text()),
    diagnostic=dict(gpu_predictors=20,fixed_epochs=[40,640],learner_seeds=[997101,997102],
        corpus_splits=json.loads((R/'predictor/dataset.json').read_text())['splits'],
        complete_information_gate=json.loads((R/'predictor_long/analysis.json').read_text())['information_gate_passed'],
        conclusion='No policy architecture promoted. State improves prediction but does not pass the complete preregistered comparator gate; history/contact not supported by these diagnostics.'),
    action_tape=json.loads((R/'action_tape/completed.json').read_text()),
    resources=dict(started=resources['started'],finished=resources['finished'],passed=resources['passed'],
        job_results=resources['job_results'],all_godot_processes=resources['all_godot_processes']),
    tests=tests,failed_test_attempt_preserved='evidence/closeout/tests_attempt_01.json',defaults_unchanged=defaults['passed'],
    active_budget=status,active_minutes=status['used_seconds']/60,
    accounting_note='Includes explicit 300-second preledger preparation estimate; subsequent confirmed work/process union. Final checksum/Git bookkeeping after this seal excluded.',
    retained_artifacts=dict(count=artifact['count'],logical_bytes=artifact['logical_bytes']),
    next_step='Use fixed action tapes to decompose short-horizon proxy errors by landing/support/foot switching before choosing a policy or memory change.')
atomic_json(R/'closeout/RESULT.json',result)
atomic_json(D/'RESULT.json',result)
with (D/'RESULT.md').open('a') as f:
    f.write(f'\n\n封存工时 **{result["active_minutes"]:.3f} 分钟**，其中开账前 5 分钟为显式估计，之后按确认心跳和任务时间并集计时；最终校验／Git 记账未计入该截止值。12 个监督任务均有终态，11 个成功、1 个为已保留的测试环境失败；无残留 Godot 或学习进程，账本 actors 为空。回归 **295 项，293 通过、2 可选跳过**。73 个本地原始数据／研究权重文件合计约 **669.8 MB 逻辑大小**，未重复复制此前采样数据；默认模型及物理 15 项哈希保持不变。\n')
print(json.dumps(dict(minutes=result['active_minutes'],actors=status['actors'],jobs=12,tests=tests)))
