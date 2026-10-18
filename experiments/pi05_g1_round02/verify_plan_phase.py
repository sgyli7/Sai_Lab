"""Read-only CPU payload verification at explicit evaluation phase boundaries."""
import argparse
import datetime
from pathlib import Path
from .make_benchmark_plan import read,write,artifact,sha


def verify(plan_path,phase,receipt_path):
    if phase not in ('start','end'):raise ValueError('Boundary must be start or end')
    receipt_path=Path(receipt_path)
    if receipt_path.exists():raise FileExistsError('Boundary evidence must never be overwritten')
    plan_path=Path(plan_path).resolve();plan=read(plan_path)
    receipt={'schema':'pi05_g1_round02_phase_verification_v1','boundary':phase,'plan':str(plan_path),
             'plan_sha256':sha(plan_path),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'verified_payloads':[],'frozen_files_verified':{},'errors':[],'passed':False}
    cache={}
    for path,expected in plan['frozen_files'].items():
        try:
            actual=sha(path);receipt['frozen_files_verified'][path]=actual
            if actual!=expected:receipt['errors'].append('Immutable input changed: '+path)
        except OSError as error:receipt['errors'].append('Immutable input unavailable: '+path+': '+str(error))
    for job in plan['jobs']:
        path=job['config_path']
        try:
            actual=sha(path)
            if actual!=job['config_sha256']:receipt['errors'].append('Case config changed: '+path)
            receipt['frozen_files_verified'][path]=actual
        except OSError as error:receipt['errors'].append('Case config unavailable: '+path+': '+str(error))
    policies=set(plan['policies'])
    payloads=[v for k,v in plan['model_artifacts'].items() if k in policies]
    if 'pi05_round02' in policies:payloads+=plan.get('ancestor_model_artifacts',[])
    if 'gr00t_n17' in policies:payloads+=plan['gr00t_artifacts']
    for entry in payloads:
        try:
            actual=artifact(entry['path'],cache);receipt['verified_payloads'].append(actual)
            if actual['sha256']!=entry['sha256'] or actual['bytes']!=entry['bytes']:receipt['errors'].append('Model payload changed: '+entry['path'])
        except OSError as error:receipt['errors'].append('Model payload unavailable: '+entry['path']+': '+str(error))
    receipt.update(passed=not receipt['errors'],ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(receipt_path,receipt)
    if not receipt['passed']:raise ValueError('Phase verification failed; inspect preserved receipt')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--phase',choices=('start','end'),required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();r=verify(a.plan,a.phase,a.receipt);print('PHASE_VERIFIED',a.phase,len(r['verified_payloads']))


if __name__=='__main__':main()
