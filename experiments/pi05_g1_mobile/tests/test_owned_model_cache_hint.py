"""Tiny owned temp files only; no actual models,queue,CUDAs or quiescence probe."""
import datetime,hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import owned_model_cache_hint as c

def write(path,value):path.write_text(json.dumps(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class HintTests(unittest.TestCase):
    def fixture(self,root,dedup=False):
        base=root/'base/model.safetensors';base.parent.mkdir();base.write_bytes(b'official tiny model'*50)
        train=root/'training_carry_run01';chosen=train/'step_000400';chosen.mkdir(parents=True);fine=chosen/'model.safetensors'
        if dedup:os.link(base,fine)
        else:fine.write_bytes(b'fine tiny model'*50)
        native=root/'fixed_native_revision';native.mkdir()
        weights=[]
        for i in range(2):
            path=native/f'model-{i}.safetensors';path.write_bytes(bytes([i])*1000);weights.append(path)
        nr=root/'native_receipt.json';write(nr,{'profile':'mobile_box','revision':native.name,'model_root':str(native),'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in weights]})
        mp=chosen/'metadata.json';write(mp,{'weights_sha256':sha(fine)})
        selection=root/'selection.json';write(selection,{'recorded_selected':{'path':str(chosen),'metadata_sha256':sha(mp)}})
        def artifact(p):return {'path':str(p),'sha256':sha(p),**{k:v for k,v in c.metadata(p).items() if k not in ('mode','uid')}}
        artifacts={'official_pi05_base':artifact(base),'pi05_mobile':artifact(fine),'gr00t_n16':[artifact(p) for p in weights]}
        plan=root/'plan.json';write(plan,{'schema':'pi05_g1_mobile_benchmark_plan_v1','selection_receipt':str(selection),'model_artifacts':artifacts,'frozen_files':{str(selection):sha(selection),str(nr):sha(nr)}})
        phase=root/'phase.json';flat=[artifacts['official_pi05_base'],artifacts['pi05_mobile'],*artifacts['gr00t_n16']];write(phase,{'schema':'pi05_g1_mobile_execution_phase_sha_v1','status':'verified','plan':str(plan),'plan_sha256':sha(plan),'frozen_files':json.loads(plan.read_text())['frozen_files'],'model_artifacts_observed':{x['path']:x for x in flat}})
        ledger=root/'gpu_budget.json';write(ledger,{'runs':[]});now=time.time();q=root/'quiescence.json';write(q,{'schema':'pi05_g1_mobile_owned_quiescence_v1','passed':True,'owned_closed':True,'ledger':str(ledger),'ledger_sha256':sha(ledger),'checked_utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),'active_owned_units':[],'active_owned_container_ids':[],'accounting_open_tokens':[]})
        patcher=patch.multiple(c,MOBILE=root,BASE=base,TRAIN=train,NATIVE=native,NATIVE_RECEIPT=nr,BASE_SHA=sha(base));patcher.start();self.addCleanup(patcher.stop)
        approval=root/'approval.json';proposal=c.proposal(plan,phase,q,approval);proposal['root_approved']=True;write(approval,proposal)
        return approval,[base,fine,*weights],now+.01
    def test_real_tinyfile_hint_preserves_bytes_metadata_and_deduplicates_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval,weights,now=self.fixture(root,dedup=True);before={str(p):(sha(p),c.metadata(p)) for p in weights};calls=[]
            def advise(fd,start,length,kind):calls.append((fd,start,length,kind));os.posix_fadvise(fd,start,length,kind)
            result=c.hint(approval,root/'result.json',fadvise=advise,now=now)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(calls),3);self.assertFalse(result['model_payload_read_or_hashed']);self.assertFalse(result['eviction_guaranteed'])
            for p in weights:self.assertEqual((sha(p),c.metadata(p)),before[str(p)])
    def test_peer_path_denied_even_with_valid_stat_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval,weights,now=self.fixture(root);a=json.loads(approval.read_text());plan=Path(a['context']['plan_path']);p=json.loads(plan.read_text());peer=root/'peer.safetensors';peer.write_bytes(b'peer')
            p['model_artifacts']['peer']={'path':str(peer)};write(plan,p)
            with self.assertRaises(ValueError):c.context(plan,a['context']['phase_path'],a['context']['quiescence_path'],now=now)
    def test_unapproved_modechange_or_stale_quiescence_fail_closed(self):
        for kind in ('unapproved','modechange','stale'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);approval,weights,now=self.fixture(root);a=json.loads(approval.read_text())
                if kind=='unapproved':a['root_approved']=False;write(approval,a)
                elif kind=='modechange':os.chmod(weights[0],0o400)
                result=c.hint(approval,root/'result.json',fadvise=lambda *x:self.fail('must not hint'),now=now+61 if kind=='stale' else now)
                self.assertEqual(result['status'],'failed');self.assertEqual(result['hints'],[])
    def test_unsupported_hint_errno_preserved_without_metadata_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approval,weights,now=self.fixture(root);before=c.metadata(weights[0])
            def unsupported(*a):raise OSError(95,'synthetic unsupported')
            result=c.hint(approval,root/'result.json',fadvise=unsupported,now=now)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['hints'][0]['errno'],95);self.assertEqual(c.metadata(weights[0]),before)

if __name__=='__main__':unittest.main()
