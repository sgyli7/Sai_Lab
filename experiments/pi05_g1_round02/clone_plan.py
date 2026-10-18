"""Relocate only evaluation outputs; preserve frozen models, inputs and replay gates."""
import argparse
import copy
from pathlib import Path
from .make_benchmark_plan import read,write
from .prepare_inputs import sha


def clone(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Clone destination must not exist, including partial directories')
    plans={name:read(source/(name+'_plan.json')) for name in ('primary','optional_base')}
    for plan in plans.values():
        for path,expected in plan['frozen_files'].items():
            if sha(path)!=expected:raise ValueError('Original frozen input changed: '+path)
        for job in plan['jobs']:
            if sha(job['config_path'])!=job['config_sha256']:raise ValueError('Original config changed')
    output.mkdir(parents=True)
    # Preserve original selection bytes, with a separate relocation receipt rather than rewriting selection history.
    (output/'selection_receipt.json').write_bytes((source/'selection_receipt.json').read_bytes())
    for name,original in plans.items():
        plan=copy.deepcopy(original);plan['selection_receipt']=str(output/'selection_receipt.json')
        plan['frozen_files'][str(output/'selection_receipt.json')]=sha(output/'selection_receipt.json')
        plan['frozen_files'][str(source/(name+'_plan.json'))]=sha(source/(name+'_plan.json'))
        for job in plan['jobs']:
            config=read(job['config_path']);policy=config['policy_id'];cid=config['case']['case_id']
            config['output_dir']=str(output/policy/cid);config['video_output_dir']=str(output/'video_exports'/policy/cid)
            path=output/'configs'/Path(job['config_path']).name;write(path,config)
            job.update(config_path=str(path),config_sha256=sha(path))
            plan['frozen_files'][str(path)]=sha(path)
        write(output/(name+'_plan.json'),plan)
    write(output/'clone_receipt.json',{'schema':'pi05_g1_round02_plan_clone_v1','source':str(source),'output':str(output),
          'changed_fields':['config.output_dir','config.video_output_dir','job.config_path','job.config_sha256','plan.selection_receipt'],
          'immutable_inputs_and_replay_gates_preserved':True,'source_selection_sha256':sha(source/'selection_receipt.json')})
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();clone(a.source,a.output);print('CLONED',a.output)


if __name__=='__main__':main()
