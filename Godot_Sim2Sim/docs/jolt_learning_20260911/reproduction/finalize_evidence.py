"""Publish audit artifacts only after every preregistered physical endpoint finishes."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,shutil
p=Path("results/jolt_learning_20260911");out=Path("docs/jolt_learning_20260911")
start=time.monotonic()
while not (p/"r3_completed.json").exists():
 if time.monotonic()-start>7500:raise RuntimeError("Final planned series did not complete")
 time.sleep(5)
for name,count in [("r0",6),("r1",6),("r2",3),("extension",6),("r3",6)]:
 rows=json.loads((p/(name+"_completed.json")).read_text())
 if len(rows)!=count or not all(r.get("completed") for r in rows):raise RuntimeError("Incomplete physical endpoints: "+name)
subprocess.run([sys.executable,str(p/"summarize_pilot.py")],check=True,timeout=30)
subprocess.run([sys.executable,str(p/"plot_trial_outcomes.py")],check=True,timeout=60)
analysis=json.loads((p/"trial_analysis.json").read_text())
if len(analysis["trials"])!=27:raise RuntimeError("Unexpected endpoint count")
model_records=[]
for trial in analysis["trials"]:
 run=p/"runs"/trial["name"]
 files={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ["config.json","final.onnx","latest.pt","completed.json"]}
 model_records.append(dict(name=trial["name"],files=files,new_samples=trial["new_samples"],cumulative_samples=trial["cumulative_samples"]))
manifest=dict(physical_endpoints=27,ppo_new_transitions=sum(r["new_samples"] for r in model_records),runs=model_records,eligible_improvements=[r["name"] for r in analysis["trials"] if r.get("eligible_improvement")],automatic_promotion=False)
(p/"completed_evidence_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
for src,dst in [("trial_analysis.json","TRIALS.json"),("trial_outcomes.png","trial_outcomes.png"),("completed_evidence_manifest.json","EVIDENCE_MANIFEST.json"),("r3_completed.json","R3_RUNS.json"),("proxy_return_audit.json","PROXY_RETURNS.json"),("sampling_gap_complete.json","SAMPLING_GAP.json"),("physics_preservation_audit.json","PHYSICS_PRESERVATION.json"),("reproduction_environment.json","ENVIRONMENT.json")]:
 shutil.copy2(p/src,out/dst)
print("EVIDENCE_READY "+json.dumps({k:v for k,v in manifest.items() if k!="runs"}),flush=True)
