"""Read-only V5 stored/logical byte audit; no restore, physics or model loading."""
import argparse,gzip,hashlib,json,os,re,stat,datetime
from pathlib import Path

CHUNK=1024**2
HEX=re.compile('[0-9a-f]{64}')


def check(value,message):
    if not value:raise ValueError(message)

def snapshot(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_gid,s.st_mode,s.st_nlink,s.st_blocks)

def safe(root,name):
    check(isinstance(name,str),'Path must be string');rel=Path(name)
    check(not rel.is_absolute() and rel.parts and str(rel)==name and '..' not in rel.parts,'Unsafe/noncanonical path')
    p=root/rel;check(p.resolve().is_relative_to(root.resolve()) and not any(x.is_symlink() for x in [p,*p.parents]),'Symlink/path escape')
    return p

def hash_file(path,cache):
    before=snapshot(path);check(stat.S_ISREG(before[7]),'Stored file not regular')
    if before not in cache:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
        check(snapshot(path)==before,'Stored file mutated during SHA');cache[before]=h.hexdigest()
    check(snapshot(path)==before,'Stored file metadata changed');return cache[before]

def logical_stream(path,row):
    opener=gzip.open if row['type']=='gzip' else open
    with opener(path,'rb') as f:
        size=0
        while True:
            b=f.read(min(CHUNK,row['bytes']-size+1))
            if not b:break
            size+=len(b);check(size<=row['bytes'],'Logical bytes exceed original bound');yield b
        check(size==row['bytes'],'Logical byte count differs')

def archived_json(row,root):
    check(row['bytes']<=16*1024**2,'Lightweight archived JSON exceeds16MiB');p=safe(root,row['relative_path']);before=snapshot(p)
    value=json.loads(b''.join(logical_stream(p,row)));check(snapshot(p)==before,'Archived JSON mutated');return value


def inspect(root):
    cache={};receipt_path=safe(root,'archive_receipt.json');inventory=safe(root,'inventory.jsonl');plan_path=safe(root,'archive_plan.json')
    receipt_sha=hash_file(receipt_path,cache);receipt_before=snapshot(receipt_path);receipt=json.loads(receipt_path.read_text())
    check(receipt.get('schema')=='pi05_g1_mobile_archive_receipt_v5' and receipt.get('archive_complete') is True and receipt.get('status') in ('retained_final','retained_partial'),'Complete V5 receipt required')
    check(receipt.get('phase') in ('final','partial') and receipt['status']=='retained_'+receipt['phase'],'Receipt phase/status differ')
    check(hash_file(inventory,cache)==receipt.get('inventory_sha256'),'Inventory SHA differs');inventory_before=snapshot(inventory)
    plan_before=snapshot(plan_path);plan=json.loads(plan_path.read_text());check(plan.get('schema')=='pi05_g1_mobile_archive_plan_v1','Archived plan schema differs')
    logical={};stored=set();source_map={};rows=[];verified={};stats={'files':0,'directories':0,'gzip_files':0,'shared_inode_hardlinks':0,'logical_bytes':0,'stored_file_bytes_counted_by_path':0}
    with inventory.open() as stream:
        for line in stream:
            row=json.loads(line);kind=row.get('type');check(kind in ('directory','copy','hardlink','gzip'),'Unsupported inventory type')
            name=row.get('logical_relative_path');target=safe(root,name);p=safe(root,row.get('relative_path'))
            check(Path(name).parts[0]=='retained' and name not in logical and row['relative_path'] not in stored,'Duplicate/non-retained logical or stored path')
            logical[name]=row;stored.add(row['relative_path'])
            if kind=='directory':
                check(row['relative_path']==name and p.is_dir(),'Missing/mapped directory');stats['directories']+=1;continue
            check(type(row.get('bytes')) is int and row['bytes']>=0 and HEX.fullmatch(str(row.get('sha256'))),'Original size/SHA missing')
            before=snapshot(p);check(stat.S_ISREG(before[7]),'Stored file is not regular');stored_sha=hash_file(p,cache)
            if kind=='gzip':
                check(row['bytes']>=65536 and Path(row['source_path']).suffix.lower() in {'.json','.jsonl','.log','.txt','.md','.csv'} and row.get('symlink_target') is None,'Ineligible compressed source')
                expected='text_gzip/'+hashlib.sha256(name.encode()).hexdigest()+'.gz'
                check(row['relative_path']==row.get('stored_relative_path')==expected,'Gzip flat path differs')
                check(type(row.get('stored_bytes')) is int and before[2]==row['stored_bytes'] and stored_sha==row.get('stored_sha256'),'Stored gzip size/SHA differs')
                check(row.get('gzip_level')==6 and row.get('gzip_mtime')==0 and row.get('roundtrip_original_sha256')==row['sha256'],'Original gzip metadata differs')
                with p.open('rb') as f:header=f.read(10)
                check(len(header)==10 and header[:3]==b'\x1f\x8b\x08' and header[3]==0 and header[4:8]==bytes(4),'Gzip mtime/empty filename/header differs')
                h=hashlib.sha256()
                for b in logical_stream(p,row):h.update(b)
                check(h.hexdigest()==row['sha256'],'Decompressed original SHA differs');stats['gzip_files']+=1
            else:check(row['relative_path']==name and before[2]==row['bytes'] and stored_sha==row['sha256'],'Plain stored/original identity differs')
            check(all(type(row.get(k)) is int for k in ('source_uid','source_device','source_inode','source_mtime_ns')),'Source stat provenance missing')
            shared=(before[0],before[1])==(row['source_device'],row['source_inode'])
            check(not shared or kind=='hardlink','Copy/gzip unexpectedly shares original source inode')
            if kind=='hardlink':
                check(row.get('declared_immutable') is True and row.get('source_uid')==os.getuid()!=0 and before[5]==row['source_uid'] and (before[0],before[1])==(row.get('source_device'),row.get('source_inode')) and before[8]>=2,'Hardlink UID/immutable/shared inode identity differs')
                stats['shared_inode_hardlinks']+=1
            check(snapshot(p)==before,'Stored file changed during logical verification');verified[str(p)]=before
            for source in set((row.get('source_path'),row.get('source_resolved_path')))-{None}:
                source_map.setdefault(source,[]).append(row)
            rows.append(row);stats['files']+=1;stats['logical_bytes']+=row['bytes'];stats['stored_file_bytes_counted_by_path']+=before[2]
    check(snapshot(inventory)==inventory_before and snapshot(plan_path)==plan_before,'Inventory/plan changed during audit')
    check(stats['files']==receipt.get('files'),'Inventory filecount differs from complete receipt')
    for name in logical:
        for parent in Path(name).parents:
            check(str(parent) not in logical or logical[str(parent)]['type']=='directory','File ancestor/nested logical collision')
    def retained(source,digest=None):
        matches=[r for r in source_map.get(source,[]) if digest is None or r['sha256']==digest];check(matches,'Required source absent/wrong logicalSHA:'+source);return matches[0]
    for item in plan.get('required_files',[]):retained(item['path'],item['sha256'])
    check(bool(plan.get('required_files')) and bool(receipt.get('input_bindings')),'Required closure/bindings missing')
    for source,h in receipt['input_bindings'].items():retained(source,h)
    original=retained(receipt['plan_path'],receipt['input_bindings'].get(receipt['plan_path']))
    check(archived_json(original,root)==plan,'Normalized archive_plan differs semantically from retained original plan')
    final={}
    if receipt['phase']=='final':
        training=archived_json(retained(plan['training_result']),root);benchmark=archived_json(retained(plan['benchmark']),root)
        check(training.get('schema')=='pi05_g1_mobile_training_result_v1' and training.get('status')=='completed' and training.get('training_stopped',{}).get('complete_optimizer_update_boundary') is True,'Archived training not completed')
        check(benchmark.get('schema')=='pi05_g1_mobile_benchmark_summary_v1' and all(benchmark.get(k) is True for k in ('complete','evidence_qualified','formal_verdict_available')) and not benchmark.get('errors'),'Archived benchmark not qualified')
        cases=benchmark.get('case_ids',[]);policies=benchmark.get('policies',{});episodes=benchmark.get('episodes',[])
        check(len(cases)==len(set(cases))==20 and len(policies)==3 and all(x.get('completed_evidence_cases')==20 for x in policies.values()) and len(episodes)==60 and {(e['policy'],e['case_id']) for e in episodes}=={(p,c) for p in policies for c in cases},'Archived benchmark not exact3x20')
        offline=[]
        for row in rows:
            if Path(row['source_path']).name!='offline_result.json':continue
            value=archived_json(row,root)
            if value.get('schema')=='pi05_g1_mobile_offline_error_result_v1' and value.get('status')=='completed':offline.append(value)
        check(any(o.get('full_test_episode_count')==10 and o.get('full_test_chunk_count')==200 and o.get('baseline',{}).get('chunks')==o.get('finetuned',{}).get('chunks')==200 and o.get('test_used_for_checkpoint_selection') is False and o.get('identity',{}).get('selected',{}).get('training_result_sha256')==retained(plan['training_result'])['sha256'] and o.get('identity',{}).get('weights_sha256',{}).get('finetuned')==retained(str(Path(training['selected_checkpoint'])/'model.safetensors'))['sha256'] for o in offline),'Completed archived400 paired offline diagnostics absent')
        final={'training_completed':True,'benchmark_primary3x20':True,'paired_offline400_complete':True,'physical_acceptance_recomputed':False}
    check(hash_file(receipt_path,{})==receipt_sha and hash_file(inventory,{})==receipt['inventory_sha256'],'Archive receipt/inventory changed at final boundary')
    check(snapshot(receipt_path)==receipt_before and snapshot(inventory)==inventory_before and snapshot(plan_path)==plan_before,'Archive controlmetadata changed before completion')
    for row in rows:
        path=safe(root,row['relative_path']);check(snapshot(path)==verified[str(path)] and hash_file(path,cache)==(row['stored_sha256'] if row['type']=='gzip' else row['sha256']),'Stored file changed before completion')
    return {'inventory_sha256':receipt['inventory_sha256'],'archive_receipt_sha256':receipt_sha,'phase':receipt['phase'],'counts':stats,'required_closure_checked':len(plan['required_files']),'input_bindings_checked':len(receipt['input_bindings']),'original_plan_semantic_equal':True,'all_stored_and_logical_SHA_verified':True,'shared_hardlinks_are_not_independent_copies':True,'physical_inode_SHA_cache_entries':len(cache),'final_light_fields':final}


def audit(archive_root,output):
    root=Path(archive_root).absolute();out=Path(output).absolute()
    check(not out.exists() and not out.is_symlink() and not any(x.is_symlink() for x in out.parents),'Fresh nonsymlink report path required')
    check(not out.resolve().is_relative_to(root.resolve()),'Report cannot modify archive tree');out.parent.mkdir(parents=True,exist_ok=True)
    report={'schema':'pi05_g1_mobile_archive_independent_audit_v5','status':'failed','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(root),'read_only':True,'restore_or_GPU_started':False,'errors':[]}
    try:
        check(root.is_dir() and not any(x.is_symlink() for x in [root,*root.parents]),'Archive root symlink/missing');report.update(inspect(root));report['status']='passed'
    except BaseException as error:report['errors'].append(type(error).__name__+': '+str(error))
    report['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.archive,a.output);print(json.dumps({'status':r['status'],'output':str(a.output)}));return 0 if r['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
