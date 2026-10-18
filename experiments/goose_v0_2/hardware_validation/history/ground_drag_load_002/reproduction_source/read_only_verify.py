"""Recompute the recorded audit without running top-level experiment/render code or writing evidence."""
from pathlib import Path
import os, sys, ast, json, gzip, hashlib, time, resource, argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(180,180))
parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parent));parser.add_argument('--receipt');args=parser.parse_args();ROOT=Path(args.root).resolve();START=time.time()
import numpy as realnp
import mujoco as mj
assert mj.__version__=='3.10.0'
PKG=ROOT/'sealed04/physics_v0_2_20261018_04';counts={'JSON_results_compared':0,'NPZ_arrays_compared':0,'physics_integrations':0}
class ReadOnlyNumpy:
    def __getattr__(self,name):return getattr(realnp,name)
    def savez_compressed(self,path,**arrays):
        with realnp.load(path) as old:
            assert set(old.files)==set(arrays),str(path)
            for k,v in arrays.items():assert realnp.array_equal(old[k],v),(str(path),k)
            counts['NPZ_arrays_compared']+=len(arrays)
def compare_json(path,value):
    assert json.loads(Path(path).read_text())==value,str(path)
    counts['JSON_results_compared']+=1
ns={'ROOT':ROOT,'PKG':PKG,'C':json.loads((PKG/'physical_contract.json').read_text()),'DR':json.loads((PKG/'actuator_contract.json').read_text()),'PLAN':json.loads((ROOT/'kinematic_plan.json').read_text()),'COUNTS':{'kinematic_queries':0},'RESULTS':[],'PLOTS':[],'Path':Path,'np':ReadOnlyNumpy(),'mj':mj,'json':json,'gzip':gzip,'sha':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),'write':compare_json}
# Only these pure audit functions are compiled. Top-level initialization, deadline,
# serve/render loops and their output writes are deliberately not executed.
source=ROOT/'analyze_and_render.py';tree=ast.parse(source.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'mat','gaps','analyze'}];assert len(selected)==3
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
for n in range(1,6):ns['analyze'](n)
r=resource.getrusage(resource.RUSAGE_SELF)
out={'status':'PASS_READ_ONLY_RECOMPUTATION','engine':mj.__version__,'rows_checked':sum(x['rows_checked'] for x in ns['RESULTS']),'counts':counts,'physics_integrations':0,'NN':0,'GPU':0,'training':0,'evidence_files_written':0,'source_audit_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'resources':{'process_wall_s':time.time()-START,'CPU_affinity':sorted(os.sched_getaffinity(0)),'maxRSS_KiB':r.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'address_space_cap_bytes':resource.getrlimit(resource.RLIMIT_AS)[0]},'limits':'Checks recorded state/drive/contact-wrench consistency and prior derived outputs. Does not regenerate solved contact forces or prove hardware capability.'}
if args.receipt:Path(args.receipt).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
