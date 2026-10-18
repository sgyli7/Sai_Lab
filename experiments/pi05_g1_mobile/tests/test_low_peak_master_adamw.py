"""CPU-only candidate equivalence; no actual CUDA model allocation."""
import copy
import unittest
from unittest.mock import patch
import torch
from experiments.pi05_g1.training import MasterAdamW as Original
from experiments.pi05_g1_mobile.low_peak_master_adamw import MasterAdamW,release_unused_cuda_cache


def equal_tree(case,a,b):
    if torch.is_tensor(a):case.assertTrue(torch.equal(a,b));case.assertEqual(a.dtype,b.dtype)
    elif isinstance(a,dict):
        case.assertEqual(a.keys(),b.keys())
        for key in a:equal_tree(case,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        case.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal_tree(case,x,y)
    else:case.assertEqual(a,b)


class LowPeakTests(unittest.TestCase):
    def test_mixed_precision_none_grad_multistep_bitwise(self):
        torch.manual_seed(9)
        before=torch.get_rng_state().clone()
        p=[torch.nn.Parameter(torch.tensor([1.125,-.25],dtype=d)) for d in (torch.bfloat16,torch.float32,torch.bfloat16)]
        q=[torch.nn.Parameter(x.detach().clone()) for x in p]
        old=Original(p,lr=1e-5);new=MasterAdamW(q,lr=1e-5)
        for step in range(12):
            old.zero_grad();new.zero_grad()
            loss1=sum((x.float().square().sum()*(step+1)) for x in p[:2]);loss2=sum((x.float().square().sum()*(step+1)) for x in q[:2])
            self.assertTrue(torch.equal(loss1,loss2));loss1.backward();loss2.backward()
            torch.nn.utils.clip_grad_norm_(p,1.);torch.nn.utils.clip_grad_norm_(q,1.)
            alias=q[1].grad;old_step=new.optimizer.step
            def checked_step():
                self.assertTrue(all(x.grad is None for x in q));self.assertEqual(new.master_parameters[1].grad.data_ptr(),alias.data_ptr());self.assertIsNone(new.master_parameters[2].grad)
                return old_step()
            with patch.object(new.optimizer,'step',side_effect=checked_step):old.step();new.step()
            for x,y in zip(p,q):self.assertTrue(torch.equal(x,y))
            equal_tree(self,old.state_dict(),new.state_dict())
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        saved=copy.deepcopy(new.state_dict());new.load_state_dict(saved)
        equal_tree(self,old.state_dict(),new.state_dict())
        old.zero_grad(set_to_none=False);new.zero_grad(set_to_none=False)
        equal_tree(self,old.state_dict(),new.state_dict())

    def test_fake_cuda_release_order_no_context_realization(self):
        events=[]
        class Context:
            def __enter__(self):events.append('enter')
            def __exit__(self,*args):events.append('exit')
        with patch.object(torch.cuda,'synchronize',side_effect=lambda d:events.append(('sync',str(d)))),patch.object(torch.cuda,'device',side_effect=lambda d:Context()),patch.object(torch.cuda,'empty_cache',side_effect=lambda:events.append('empty')):
            release_unused_cuda_cache([torch.device('cuda:0')])
        self.assertEqual(events,[('sync','cuda:0'),'enter','empty','exit',('sync','cuda:0')])

    def test_cpu_cache_release_never_queries_cuda(self):
        with patch.object(torch.cuda,'synchronize',side_effect=AssertionError('No CUDA')):
            p=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16));optimizer=MasterAdamW([p],lr=1e-5);p.grad=torch.ones_like(p);optimizer.step()
        self.assertIsNone(p.grad)

if __name__=='__main__':unittest.main()
