"""Run a reviewable, editable experiment queue within the session deadline."""
from .budget import legacy_deadline
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .tasks import SESSION

def main():
    p=argparse.ArgumentParser();p.add_argument("queue",type=Path);p.add_argument("--wait-for")
    args=p.parse_args();deadline=legacy_deadline(SESSION)-180
    if args.wait_for:
        pending=SESSION/"runs"/args.wait_for
        while not (pending/"completed.json").exists() and not (pending/"error.txt").exists():
            if time.time()>deadline:return
            time.sleep(5)
    while time.time()<deadline:
        entries=json.loads(args.queue.read_text())
        candidates=[e for e in entries if e.get("enabled",True) and not (SESSION/"runs"/e["name"]).exists()]
        if not candidates:return
        e=candidates[0];name=e["name"]
        command=[sys.executable,"-m","sim2sim.research.train","--skill",e["skill"],"--name",name]
        for key,value in e.get("options",{}).items():command.extend(["--"+key.replace("_","-"),str(value)])
        environment=os.environ.copy();environment.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
        record={"name":name,"skill":e["skill"],"hypothesis":e["hypothesis"],"command":command,"start_unix":time.time()}
        with (SESSION/(name+".log")).open("w") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
            record["pid"]=process.pid
            (args.queue.with_suffix(".active.json")).write_text(json.dumps(record,indent=2))
            while process.poll() is None:
                if time.time()>deadline+120:
                    process.terminate();process.wait(timeout=15);break
                time.sleep(5)
        record.update(returncode=process.returncode,end_unix=time.time())
        with (SESSION/"experiments.jsonl").open("a") as log:log.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)

if __name__=="__main__":main()
