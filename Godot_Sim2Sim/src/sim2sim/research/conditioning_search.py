"""Bounded command-conditioning search with unchanged physical evaluation."""
from .budget import legacy_deadline
import json,time
import numpy as np
from .tasks import TASKS,BASELINE,SESSION
from .conditioning import adapt,parity
from .evaluate import run_suite


def main():
    root=SESSION/"conditioning_probe_v2";root.mkdir(exist_ok=True)
    deadline=min(time.time()+15*60,legacy_deadline(SESSION)-3600)
    records=[]
    for label,source in [("factory",TASKS["walking"].source),("previous",BASELINE/"Walk_Godot.onnx")]:
        best={}
        for axis,values,conds in [("x",[.75,1,1.5,2,2.5,3],["walk_015","walk_025","run_040","back_020"]),
                                   ("yaw",[.5,1,1.5,2,3],["turn_l","turn_r"])]:
            scores=[]
            for gain in values:
                if time.time()>deadline:return
                name=f"{label}_{axis}{gain:g}";out=root/name;path=root/(name+".onnx")
                matrix=np.eye(3,dtype=np.float32);matrix[0 if axis=="x" else 2,0 if axis=="x" else 2]=gain
                adapt(source,path,matrix)
                result=run_suite("walking",path,seeds=(300,),workers=4,out=out,selected_conditions=conds,entry="both")
                entry=dict(name=name,source=str(source),matrix=matrix.tolist(),success=result["success_rate"],score=result["score"],errors=result["errors"])
                records.append(entry);scores.append((result["score"],gain));print(json.dumps(entry),flush=True)
                (root/"search.json").write_text(json.dumps(records,indent=2))
            best[axis]=max(scores)[1]
        matrix=np.diag([best["x"],1.,best["yaw"]]).astype(np.float32)
        path=root/(label+"_combined.onnx");adapt(source,path,matrix)
        report=parity(source,path,matrix);path.with_suffix(".parity.json").write_text(json.dumps(report,indent=2))
        result=run_suite("walking",path,seeds=(110,111,112),workers=4,out=root/(label+"_combined"),entry="both")
        print(json.dumps(dict(name=label+"_combined",matrix=matrix.tolist(),success=result["success_rate"],score=result["score"],errors=result["errors"],parity=report)),flush=True)


if __name__=="__main__":main()
