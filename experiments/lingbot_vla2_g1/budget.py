"""Independent owned GPU accounting, inside the shared FIFO and host flock."""
import argparse
import json
from pathlib import Path
import sys
import time
from experiments.pi05_g1 import budget as owned

ROOT=Path(__file__).resolve().parents[2]/'.scratch/lingbot_vla2_g1'
LIMITS={'preflight':3600,'training':14400,'evaluation':10800}
TOTAL=28800


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--result-file',type=Path,required=True)
    parser.add_argument('--result-schema',required=True)
    args,rest=parser.parse_known_args()
    if not args.ledger.resolve().is_relative_to(ROOT) or not args.result_file.resolve().is_relative_to(ROOT):raise ValueError('LingBot owned paths required')
    if args.result_file.exists():raise FileExistsError('Preserve prior result; choose a new attempt')
    started=time.time_ns();old=sys.argv;code=0
    owned.LIMITS=dict(LIMITS);owned.TOTAL=TOTAL
    try:
        sys.argv=[old[0],'--ledger',str(args.ledger),*rest]
        try:owned.main()
        except SystemExit as error:code=error.code or 0
    finally:sys.argv=old
    contract={'expected_schema':args.result_schema,'path':str(args.result_file),'valid':False}
    try:
        payload=json.loads(args.result_file.read_text())
        contract['valid']=(args.result_file.stat().st_mtime_ns>=started and payload.get('schema')==args.result_schema and payload.get('status')=='completed')
        contract['producer_status']=payload.get('status')
    except (OSError,ValueError) as error:contract['error']=repr(error)
    if not contract['valid'] and code==0:code=2
    print('LINGBOT_WORKLOAD_COMPLETION',json.dumps({'exit_code':code,'contract':contract}),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
