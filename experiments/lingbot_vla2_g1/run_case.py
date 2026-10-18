"""CPU preflight then the same owned frozen simulator with LingBot evidence root."""
import argparse
import json
from pathlib import Path
from .data import sha
from .sim_launcher import run_owned_docker


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text());result=Path(config['output_dir'])/'result.json'
    if result.exists():raise FileExistsError('Preserve prior case')
    try:
        for path,digest in config['frozen_files'].items():
            if sha(path)!=digest:raise ValueError('Frozen case input changed: '+path)
        run_owned_docker(args.config)
    except BaseException as error:
        if not result.exists():
            result.parent.mkdir(parents=True,exist_ok=True)
            with result.open('x') as output:json.dump({'schema':'pi05_g1_mobile_case_result_v1','status':'failed','infrastructure_failure':True,'success':None,'error':repr(error),'case_id':config['case']['case_id'],'policy':config['policy_id']},output,indent=2)
        raise


if __name__=='__main__':main()
