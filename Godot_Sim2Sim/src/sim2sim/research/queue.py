"""Recoverable experiment queue governed by confirmed active work, never wall time.

Queue completion means the requested experiment and evidence were produced.
Model promotion is a separate, explicit comparison decision.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .budget import ActiveBudget, run_supervised, process_start_ticks, _boot_id


def atomic_json(path, data):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path)


class ExperimentQueue:
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        self.path=self.directory/'experiment_queue.json'
        self.budget=ActiveBudget(self.directory)

    @contextmanager
    def locked(self):
        with (self.directory/'experiment_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    def initialize(self,specification,append=False):
        with self.locked():
            if self.path.exists() and not append:raise FileExistsError(self.path)
            queue=json.loads(self.path.read_text()) if append else dict(schema_version=1,reserve_seconds=5400,
                                      created_unix=time.time(),experiments=[])
            experiments=specification['experiments']
            ids=[item['id'] for item in experiments+queue['experiments']]
            if len(ids)!=len(set(ids)):raise ValueError('Experiment ids must be unique')
            for item in experiments:
                if not all(item.get(key) for key in ['id','skill','hypothesis','source','control_version','command','evidence']):
                    raise ValueError('Each experiment needs a falsifiable hypothesis, source, control, command and evidence')
                seconds=float(item['timeout_seconds'])
                if not math.isfinite(seconds) or seconds<=0:raise ValueError('A finite positive timeout is required')
                minimum=float(item.get('minimum_seconds',seconds))
                if not math.isfinite(minimum) or minimum<=0 or minimum>seconds:
                    raise ValueError('Minimum completion budget must be positive and no greater than timeout')
                if not 1<=int(item.get('max_attempts',2))<=3:raise ValueError('Retries must be bounded to 1–3 attempts')
                item.update(status='queued',attempts=[],decision=None)
            queue['experiments'].extend(experiments)
            atomic_json(self.path,queue)

    def _tokens(self,item,number):
        attempt_id=f"{item['id']}_attempt_{number:02d}"
        return dict(session=str(self.directory),attempt_id=attempt_id,
                    run_dir=str(self.directory/'runs'/attempt_id),python=sys.executable)

    def _evidence(self,item,tokens):
        checked=[]
        for requirement in item['evidence']:
            path=Path(requirement['path'].format(**tokens))
            if not path.is_file():raise RuntimeError(f'Missing experiment evidence: {path}')
            data=json.loads(path.read_text())
            for dotted,expected in requirement.get('equals',{}).items():
                value=data
                for key in dotted.split('.'):value=value[key]
                if value!=expected:raise RuntimeError(f'{path}: {dotted}={value!r}, expected {expected!r}')
            checked.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return checked

    def execute(self,selected=None):
        # One scheduler owns a queue. Independent, frozen commands may themselves
        # parallelize workers; formal benchmarks use a separate exclusive job.
        with self.locked():
            queue=json.loads(self.path.read_text())
            for item in queue['experiments']:
                if selected is not None and item['id']!=selected:continue
                if item['status']=='running':
                    # The scheduler may have died while its detached job survived.
                    # PID start time plus boot id prevents PID reuse confusion.
                    events=json.loads(self.budget.path.read_text())['events']
                    prefix=item['attempts'][-1]['id']+':'
                    jobs=[event for event in events if event['kind']=='job_started' and event.get('actor','').startswith(prefix)]
                    if jobs:
                        job=jobs[-1]
                        if job.get('boot')==_boot_id() and job.get('process_start_ticks') is not None and process_start_ticks(job['pid'])==job['process_start_ticks']:
                            return 'prior_job_still_running'
                    # Preserve the unfinished attempt, even when its directory
                    # contains artifacts; no successful exit was confirmed here.
                    item['attempts'][-1].update(status='interrupted',reason='scheduler_disconnected')
                    item['status']='interrupted'
                if item['status'] in ['completed','abandoned']:continue
                attempts=int(item.get('max_attempts',2))
                while len(item['attempts'])<attempts:
                    minimum=float(item.get('minimum_seconds',item['timeout_seconds']))
                    if self.budget.status()['remaining_seconds']<queue['reserve_seconds']+minimum:
                        atomic_json(self.path,queue)
                        return 'insufficient_budget'
                    number=len(item['attempts'])+1;tokens=self._tokens(item,number)
                    command=[]
                    for token in item['command']:
                        value=str(token)
                        for key,replacement in tokens.items():value=value.replace('{'+key+'}',replacement)
                        command.append(value)
                    resume=None
                    if number>1 and item.get('resume_complete_checkpoint',False):
                        previous=self._tokens(item,number-1)
                        checkpoint=Path(previous['run_dir'])/'latest.pt'
                        if checkpoint.is_file():
                            resume=str(checkpoint);command.extend(['--resume',resume])
                    files={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for path in [item['source'],*item.get('control_files',[])]}
                    attempt=dict(id=tokens['attempt_id'],started_unix=time.time(),
                        active_seconds_at_start=self.budget.status()['used_seconds'],
                        status='running',command=command,source_and_control_sha256=files,
                        resume_checkpoint=resume)
                    item['attempts'].append(attempt);item['status']='running'
                    atomic_json(self.path,queue)
                    logs=self.directory/'queue_logs';logs.mkdir(exist_ok=True)
                    # The child inherits normal stdout; callers persist the queue
                    # runner log. Per-attempt records retain the exact argv.
                    try:
                        code=run_supervised(self.directory,command,timeout=float(item['timeout_seconds']),
                            reserve=queue['reserve_seconds'],label=tokens['attempt_id'])
                        attempt['returncode']=code
                        finished=[event for event in json.loads(self.budget.path.read_text())['events']
                                  if event['kind']=='job_finished' and event.get('actor','').startswith(tokens['attempt_id']+':')]
                        attempt['supervisor_reason']=finished[-1]['reason'] if finished else 'unknown'
                        if code!=0:raise RuntimeError(f'Experiment exited with {code}')
                        attempt['evidence']=self._evidence(item,tokens)
                        attempt['status']='completed';item['status']='completed'
                    except (Exception,KeyboardInterrupt) as error:
                        attempt.update(status='failed',reason=f'{type(error).__name__}: {error}')
                        item['status']='failed'
                    finally:
                        attempt['finished_unix']=time.time()
                        attempt['active_seconds_at_finish']=self.budget.status()['used_seconds']
                        atomic_json(self.path,queue)
                        atomic_json(logs/(tokens['attempt_id']+'.json'),attempt)
                    if item['status']=='completed':break
                    if attempt.get('supervisor_reason')=='interrupted':return 'interrupted'
                    if not item.get('retry_failed',False):break
            return 'finished'

    def decide(self,identifier,decision,reason):
        if decision not in ['keep_incumbent','candidate_for_freeze','abandon']:
            raise ValueError('Unknown comparison decision')
        if not reason.strip():raise ValueError('Record metrics and a reason for the decision')
        with self.locked():
            queue=json.loads(self.path.read_text())
            item=next(item for item in queue['experiments'] if item['id']==identifier)
            if decision!='abandon' and item['status']!='completed':
                raise ValueError('An unfinished experiment cannot select a candidate')
            item['decision']=dict(decision=decision,reason=reason,unix=time.time())
            if decision=='abandon':item['status']='abandoned'
            atomic_json(self.path,queue)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    sub=parser.add_subparsers(dest='action',required=True)
    init=sub.add_parser('init');init.add_argument('specification',type=Path)
    add=sub.add_parser('add');add.add_argument('specification',type=Path)
    run=sub.add_parser('run');run.add_argument('--id')
    decision=sub.add_parser('decide');decision.add_argument('id')
    decision.add_argument('decision');decision.add_argument('--reason',required=True)
    args=parser.parse_args();queue=ExperimentQueue(args.directory)
    if args.action=='init':queue.initialize(json.loads(args.specification.read_text()))
    elif args.action=='add':queue.initialize(json.loads(args.specification.read_text()),append=True)
    elif args.action=='run':print(queue.execute(args.id))
    else:queue.decide(args.id,args.decision,args.reason)


if __name__=='__main__':main()
