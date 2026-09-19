"""Finish the measured round; Git/checksum bookkeeping follows the seal."""
from pathlib import Path
import hashlib, json, subprocess, time
from sim2sim.paths import load_robot_json
from sim2sim.research.budget import ActiveBudget
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_contact_calibration_20260913');d=Path('docs/sprint_contact_calibration_20260913')
cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
inputs={Path('robots/microduck_ball_stand_fix.json'),Path(cfg['godot_spec']),Path('results/sprint_joint_20260912/delivery/control.json')}
inputs.update(p for p in Path(cfg['mjcf']).parent.rglob('*') if p.is_file() and p.suffix.lower() in {'.xml','.stl','.obj','.png','.jpg'})
atomic_json(r/'closeout/source_inputs.json',dict(scope='Post-experiment input inventory; entire source robot asset directory for dependency completeness, including unused assets. Native trace hashes are separately checked against preregistered values.',files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)]))
subprocess.run(['.venv/bin/python',str(r/'audit_resources.py')],check=True)
assert json.loads((r/'closeout/resources.json').read_text())['passed']
b=ActiveBudget(r);status=b.heartbeat('agent',max_gap=1800,stop=True)
assert not status['actors'] and not status['unconfirmed_gaps']
atomic_json(r/'closeout/time.json',dict(**status,used_minutes=status['used_seconds']/60,preledger_estimate_seconds=120,excluded_after_seal='Final evidence copying, checksums and Git bookkeeping'))
ledger=json.loads((r/'active_budget.json').read_text())
finished=[dict(actor=e['actor'],returncode=e['returncode'],reason=e['reason']) for e in ledger['events'] if e['kind']=='job_finished']
result=dict(sealed_unix=time.time(),session=str(r),base_commit='60ea7d7f8e46b43bcc0384ab49bdf59ad99f8e06',
 status='calibration_completed_goal_incomplete',goal_complete=False,policy_training_started=False,policy_promoted=False,new_package=False,
 current_sprint_baseline=dict(sha256='a402791e30e8ec9ffba5a288a2ebc377e43e7b6925b49986a3c523c686665770',sprint_pass=127,sprint_total=128,ordinary_pass=128,ordinary_total=128,regression_pass=7,falls=0,scope='Prior native results, not new results from this round'),
 final_seeds_unused='928000–928049',keyboard_revalidated=False,
 experiments={name:dict(eligible=json.loads((r/name/'completed.json').read_text())['eligible'],completed=True) for name in ['contact_probe','compliance_probe','compliance_direct_probe']},
 tests=json.loads((r/'closeout/tests.json').read_text()),
 time=json.loads((r/'closeout/time.json').read_text()),jobs=finished,
 source_validation=json.loads((r/'closeout/source_validation.json').read_text()),
 resource_audit_passed=True,
 conclusion='Passive translation coordinates reduce foot representation residual, but no tested proxy passes full native short-horizon response gates. Per-joint constrained-axis telemetry is needed before further parameter fitting.',
 previous_turn='progress',this_turn='progress')
atomic_json(r/'RESULT.json',result);atomic_json(d/'RESULT.json',result)
subprocess.run(['.venv/bin/python',str(r/'archive.py')],check=True)
(d/'harness/seal.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(used_minutes=result['time']['used_minutes'],jobs=len(finished),failed_attempts=sum(x['returncode']!=0 for x in finished),tests=result['tests'],goal_complete=False)))
