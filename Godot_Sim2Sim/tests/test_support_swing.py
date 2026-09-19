import unittest
import torch
from sim2sim.research.support_swing import SupportSwing


class SupportSwingObjective(unittest.TestCase):
    def test_only_forward_supported_swing_pays_and_resets_are_independent(self):
        state=SupportSwing(6,'cpu')
        contact=torch.tensor([[True,False]]*6)
        contact[1]=False  # flight cannot pay
        for _ in range(40):state.update(contact,torch.full((6,2),.01))
        command=torch.zeros(6,13);command[:,0]=.45
        speed=torch.tensor([.45,.45,0.,-.3,.45,.45])
        tilt=torch.tensor([0.,0.,0.,0.,1.,0.]);height=torch.full((6,),.12)
        sprint=torch.tensor([True,True,True,True,True,False])
        torch.testing.assert_close(state.reward(command,speed,tilt,height,sprint),torch.tensor([1.,0.,0.,0.,0.,0.]))
        # Keep the otherwise eligible foot airborne beyond the reward window.
        for _ in range(30):state.update(contact)
        self.assertEqual(float(state.reward(command,speed,tilt,height,sprint)[0]),0.)
        state.reset(torch.tensor([0]));self.assertEqual(float(state.air_time[0].sum()),0.)
        self.assertGreater(float(state.air_time[2].sum()),.19)

    def test_contact_resets_timer_and_short_scuffs_cannot_farm_airtime(self):
        state=SupportSwing(1,'cpu')
        for _ in range(20):state.update(torch.tensor([[True,False]]),torch.full((1,2),.01))
        cmd=torch.zeros(1,13);cmd[:,0]=.45
        args=(cmd,torch.tensor([.45]),torch.zeros(1),torch.tensor([.12]),torch.tensor([True]))
        self.assertEqual(float(state.reward(*args)),0.)
        for _ in range(20):state.update(torch.tensor([[True,False]]),torch.zeros(1,2))
        self.assertEqual(float(state.reward(*args)),0.,'near-ground contact flicker must not count as a swing')
        state.update(torch.ones(1,2,dtype=torch.bool));self.assertEqual(float(state.air_time.sum()),0.)
