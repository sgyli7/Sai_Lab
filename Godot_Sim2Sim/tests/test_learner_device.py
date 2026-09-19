import unittest
import torch
from sim2sim.research.learner_device import LearnerDevice


class Anchor:
    def __deepcopy__(self,memo):raise RuntimeError('ORT sessions must stay shared')


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=Anchor()
        self.delta=torch.nn.Linear(4,2,dtype=torch.float64)
    def forward(self,x):return self.delta(x)


class LearnerDeviceTests(unittest.TestCase):
    def test_cpu_retains_single_models_and_does_not_consume_rng(self):
        p=Actor();c=torch.nn.Linear(4,1);rng=torch.get_rng_state().clone()
        device=LearnerDevice(p,c,'cpu')
        self.assertIs(device.collect_policy,p);self.assertIs(device.collect_critic,c)
        device.sync_collectors();self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device required')
    def test_gpu_updates_sync_to_cpu_without_changing_precision_or_copying_ort(self):
        p=Actor();c=torch.nn.Linear(4,1);device=LearnerDevice(p,c,'cuda')
        self.assertIs(device.collect_policy.anchor,p.anchor)
        self.assertEqual(next(p.parameters()).dtype,torch.float64)
        self.assertEqual(next(c.parameters()).dtype,torch.float32)
        x=torch.ones(5,4,dtype=torch.float64);before=device.collect_policy(x).detach().clone()
        optimizer=torch.optim.Adam(p.parameters(),lr=.01)
        optimizer.zero_grad();p(x.cuda()).square().mean().backward();optimizer.step()
        torch.testing.assert_close(device.collect_policy(x),before,rtol=0,atol=0)
        self.assertFalse(torch.equal(before,p(x.cuda()).detach().cpu()))
        device.sync_collectors()
        torch.testing.assert_close(device.collect_policy(x),p(x.cuda()).cpu(),rtol=1e-10,atol=1e-10)
        self.assertEqual(next(device.collect_policy.parameters()).device.type,'cpu')
        self.assertEqual(device.batch({'obs':x})['obs'].device.type,'cuda')


if __name__=='__main__':unittest.main()
