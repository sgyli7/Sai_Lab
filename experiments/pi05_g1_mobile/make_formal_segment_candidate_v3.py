"""New independently audited recursive-prefix seam; preserve all v2 drafts.

One explicit unapproved segment only. No GPU, queue, driver, model import or
approval. Existing v2 full frozen-input/raw-prefix verification is unchanged.
"""
import argparse,copy,json
from pathlib import Path
from . import make_formal_segment_candidate_v2 as v2
from .formal_audit_chain_v1 import verify_prior_chain


def upgrade(driver,segment_index,previous_pointer,chain):
    """Pure small-dict seam; GPU argv/physics and all existing guards preserved."""
    result=copy.deepcopy(driver);start=segment_index*3;end=start+3;stages=result['stages'];original_gpu=copy.deepcopy(stages[1]);previous=str(Path(previous_pointer).resolve()) if previous_pointer else None
    a=stages[0]['argv'];ci=a.index('-c');code=a[ci+1]
    code='from experiments.pi05_g1_mobile.formal_audit_chain_v1 import verify_prior_chain\n'+code
    marker='output=Path(out);'
    addition=f"prior_chain=verify_prior_chain({previous!r},start,plan_path) if start else {{'passed':True,'audited_completed_prefix':0}}\nassert prior_chain['passed'] is True and prior_chain['audited_completed_prefix']==start\n"
    v2.check(marker in code,'Prefix code seam changed');code=code.replace(marker,addition+marker,1)
    code=code.replace("'mode':mode,","'mode':mode,'prior_independent_chain_verified':True,'prior_independent_audited_jobs':start,",1);a[ci+1]=code
    stages[0]['result']['predicates'] += [{'path':['prior_independent_chain_verified'],'op':'eq','value':True},{'path':['prior_independent_audited_jobs'],'op':'eq','value':start}]
    a=stages[4]['argv'];ci=a.index('-c');code=a[ci+1]
    template=v2.read(v2.TEMPLATE)['driver_plan_template']['stages'][4]['argv'];tc=template[template.index('-c')+1];left=tc.index('if start:\n');right=tc.index('finished=end==60\n',left);block=tc[left:right]
    v2.check('previous=None\nfinished=end==60\n' in code,'Pointer v2 seam changed')
    code=code.replace('previous=None\nfinished=end==60\n','previous=None\n'+block+"if start:\n prior_chain=verify_prior_chain(previous_name,start,plan_name)\n assert prior_chain['passed'] is True and prior_chain['audited_completed_prefix']==start\nfinished=end==60\n",1)
    code='from experiments.pi05_g1_mobile.formal_audit_chain_v1 import verify_prior_chain\n'+code
    code=code.replace("'bindings_sha256':bindings", "'previous_pointer_path':str(Path(previous_name).resolve()) if previous is not None else None,'previous_pointer_sha256':sha(previous_name) if previous is not None else None,'audited_completed_prefix':end,'bindings_sha256':bindings",1)
    a[ci+1]=code;a[-2]=previous if start else 'NONE_FIRST_SEGMENT_ONLY'
    stages[4]['result']['predicates'].append({'path':['audited_completed_prefix'],'op':'eq','value':end})
    direct=[Path(__file__).resolve(),Path(__file__).with_name('formal_audit_chain_v1.py').resolve()]
    if start:direct += [Path(p) for p in chain['latest_direct_CPU_paths']]
    for path in direct:
        v2.check(path.is_file() and not path.is_symlink() and path.stat().st_size<=16*1024**2,'Small direct v3 dependency required');result['frozen_files'][str(path)]=v2.sha(path)
    for s in stages:
        if '-c' in s['argv']:compile(s['argv'][s['argv'].index('-c')+1],'<v3 CPU guard>','exec')
    v2.check(stages[1]==original_gpu and len(stages)==5,'GPU/5stage semantics changed')
    result['root_reviewed']=False;result['review_status']='v3 candidate only; recursively verified independent CPU audit chain, Root approval required'
    return result


def generate(plan_path,phase_path,segment_index,prior_receipts,output,previous_pointer=None):
    start=segment_index*3
    if start:
        v2.check(previous_pointer is not None,'Nonzero segment requires explicit previous independent pointer')
        chain=verify_prior_chain(previous_pointer,start,plan_path)
    else:
        v2.check(previous_pointer is None,'Genesis must not supply previous pointer');chain={'passed':True,'audited_completed_prefix':0,'nodes':[],'latest_direct_CPU_paths':[],'raw_payloads_rehashed':False}
    note=v2.generate(plan_path,phase_path,segment_index,prior_receipts,output)
    output=Path(output).resolve();original=v2.read(output/'candidate_plan.json');driver=upgrade(original,segment_index,previous_pointer,chain)
    receipt={'schema':'pi05_g1_mobile_formal_segment_candidate_preparation_v3','status':'pending_root_review','root_reviewed':False,'segment_index':segment_index,'start_index':start,'stop_index_exclusive':start+3,'v2_preparation':str(output/'preparation_receipt.json'),'v2_preparation_sha256':v2.sha(output/'preparation_receipt.json'),'v2_candidate':str(output/'candidate_plan.json'),'v2_candidate_sha256':v2.sha(output/'candidate_plan.json'),'previous_independent_chain':chain,'driver_snapshot_file_count':len(driver['frozen_files']),'driver_snapshot_source_bytes':sum(Path(p).stat().st_size for p in driver['frozen_files']),'snapshot_scope':'v2 direct inputs/producer receipts plus exactlytwo v3sources and latest3CPU evidence(pointer,guard,audit). Earlier CPUchain smallSHA verified recursively but not directlysnapshotted. Existingv2 fullrawprefix verification unchanged.','GPU_argv_and_entire_GPU_stage_equal_v2':driver['stages'][1]==original['stages'][1],'GPU_started':False,'queue_requested':False,'approved':False,'actual_model_payloads_rehashed':False,'required_root_checks':note['required_root_checks']}
    for name,value in [('candidate_plan_v3.json',driver),('preparation_receipt_v3.json',receipt)]:
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--phase-start',type=Path,required=True);p.add_argument('--segment-index',type=int,required=True);p.add_argument('--resume-receipt',type=Path,action='append',default=[]);p.add_argument('--previous-pointer',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=generate(a.plan,a.phase_start,a.segment_index,a.resume_receipt,a.output,a.previous_pointer);print(json.dumps({'status':r['status'],'root_reviewed':False,'segment_index':a.segment_index}))
if __name__=='__main__':main()
