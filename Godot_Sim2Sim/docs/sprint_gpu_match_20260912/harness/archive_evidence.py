from pathlib import Path
import json,hashlib,shutil
r=Path('results/sprint_gpu_match_20260912');d=Path('docs/sprint_gpu_match_20260912');h=d/'harness';h.mkdir(exist_ok=True);records={}
for source in sorted(r.glob('*.py')):
 target=h/source.name;shutil.copy2(source,target);records[source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
for name in ['protocol.json','gpu_pair_protocol.json','gpu_pair_graph_protocol.json','cat_pair_protocol.json','native_evaluation_protocol.json','cat_native_protocol.json','matched_start_protocol.json','test_protocol.json']:
 target=h/name;shutil.copy2(r/name,target);records[name]=hashlib.sha256(target.read_bytes()).hexdigest()
(h/'SHA256.json').write_text(json.dumps(records,indent=2)+'\n')
items=['session.json','active_budget.json','candidate_freeze.json','candidate_numerical_verification.json','native_comparison.json','handoff_audit.json','cat_pair_audit.json','environment.json','asset_audit/semantic_differences.json','response/completed.json','graph_probe_owned_stream/completed.json','capacity_probe/completed.json','gpu_contract_attempt02/completed.json','anchor_parity/progress.json','anchor_parity_followup/completed.json','train_source/interruption.json','closeout/resources.json','closeout/defaults_and_physics.json','closeout/tests.json','closeout/prepared_cleanup.json']
for name in ['train_source_graph','train_jolt_graph','train_positive','train_cat']:
 items += [name+'/'+f for f in ['config.json','completed.json','metrics.jsonl','parity.json']]
for name in ['matched_start_s05_source','matched_start_s05_jolt','matched_start_source_candidate','matched_start_jolt_candidate','matched_start_cat_candidate']:
 items.append(name+'/completed.json')
for name in items:
 source=r/name;target=d/'evidence'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if name.startswith('matched_start_'):
  value=json.loads(source.read_text());value.pop('results');value.update(original_path=str(source),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),archive_projection='Aggregate; per-case metrics remain in original results file')
  target=target.with_name('aggregate.json');target.write_text(json.dumps(value,indent=2)+'\n')
 else:shutil.copy2(source,target)
shutil.copy2(r/'constraint_research.md',d/'constraint_research.md')
manifest={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='ARCHIVE_SHA256.json'}
(d/'ARCHIVE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Archived',len(records),'harness files and',len(items),'evidence files',flush=True)
