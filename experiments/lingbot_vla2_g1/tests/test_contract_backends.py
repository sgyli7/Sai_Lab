import unittest
import numpy as np
import torch
from experiments.lingbot_vla2_g1 import contract as c
from experiments.lingbot_vla2_g1.backends import sdpa_attention,dense_stored_experts

class ContractTests(unittest.TestCase):
    def test_all_named_actions_and_units_roundtrip(self):
        action=np.arange(32,dtype=np.float32)/7-1
        np.testing.assert_array_equal(c.unpack_actions(c.pack_actions(action)),action)
        self.assertEqual(c.ACTION_MASK.sum(),32)
        self.assertEqual(c.pack_actions(action)[30],action[28])
        np.testing.assert_array_equal(c.pack_actions(action)[36:39],action[29:])
        np.testing.assert_array_equal(c.pack_actions(action)[39:53],action[14:28])
    def test_state_waist_is_not_height_and_padding_invalid(self):
        state=np.arange(32,dtype=np.float32);state[-1]=0
        out=c.pack_state(state);np.testing.assert_array_equal(c.unpack_state(out),state)
        self.assertEqual(c.STATE_MASK.sum(),31)
        self.assertEqual(out[30],0)
        np.testing.assert_array_equal(out[31:34],state[28:31])
        self.assertFalse(c.ACTION_MASK[31]);self.assertFalse(c.STATE_MASK[30])
        state[-1]=1
        with self.assertRaises(ValueError):c.pack_state(state)
    def test_unused_channels_do_not_change_robot_action(self):
        action=np.linspace(-1,1,32,dtype=np.float32);out=c.pack_actions(action)
        out[~c.ACTION_MASK]=100000
        np.testing.assert_array_equal(c.unpack_actions(out),action)

class BackendTests(unittest.TestCase):
    def test_source_scientific_notation_config_has_numeric_coefficients(self):
        from experiments.lingbot_vla2_g1.runtime import components
        config=components()[0]
        for name in ['sequence_wise_loss_coeff','router_z_loss_coeff','routed_scaling_factor','bias_update_speed']:
            self.assertIsInstance(getattr(config,name),(int,float))
            self.assertGreaterEqual(getattr(config,name),0)
    def test_sdpa_matches_original_eager_outputs_and_gradients(self):
        # Independent original implementation, including a wholly masked row.
        from lingbotvla.models.vla.lingbot_vla.utils import our_eager_attention_forward
        torch.manual_seed(7)
        values=[torch.randn(2,7,4,8,requires_grad=True),torch.randn(2,9,2,8,requires_grad=True),torch.randn(2,9,2,8,requires_grad=True)]
        mask=torch.rand(2,7,9)>.25;mask[:,0]=False
        ref=our_eager_attention_forward(*values,mask)
        grad=torch.randn_like(ref);expected=torch.autograd.grad(ref,values,grad)
        actual=sdpa_attention(*values,mask)
        got=torch.autograd.grad(actual,values,grad)
        torch.testing.assert_close(actual,ref,atol=1e-6,rtol=1e-5)
        for a,b in zip(got,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    def test_stored_expert_backend_matches_independent_token_loop_and_gradients(self):
        torch.manual_seed(9)
        module=torch.nn.Module()
        for name,shape in [('gate_proj',(3,5,4)),('up_proj',(3,5,4)),('down_proj',(3,4,5))]:
            module.register_parameter(name,torch.nn.Parameter(torch.randn(*shape)))
        x=torch.randn(4,4,requires_grad=True);routing=torch.rand(4,2,requires_grad=True)
        selected=torch.tensor([[0,1],[1,2],[0,2],[1,0]])
        rows=[]
        for token in range(4):
            out=0
            for slot in range(2):
                e=selected[token,slot]
                hidden=torch.nn.functional.silu(torch.nn.functional.linear(x[token],module.gate_proj[e]))*torch.nn.functional.linear(x[token],module.up_proj[e])
                out=out+torch.nn.functional.linear(hidden,module.down_proj[e])*routing[token,slot]
            rows.append(out)
        reference=torch.stack(rows);params=[x,routing,*module.parameters()]
        grad=torch.randn_like(reference);expected=torch.autograd.grad(reference,params,grad)
        actual=dense_stored_experts(module,None,3,routing,selected,x)
        got=torch.autograd.grad(actual,params,grad)
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=1e-5)
        for a,b in zip(got,expected):torch.testing.assert_close(a,b,atol=8e-6,rtol=3e-5)

if __name__=='__main__':unittest.main()
