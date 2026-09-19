import math
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from sim2sim.motion_control import MotionControl
from sim2sim.research.torch_walking import WalkingControl


SETTINGS = dict(walk_heading_gain=6., walk_heading_limit=.6,
                walk_sprint_turn_tracking=True)


def state(yaw):
    return SimpleNamespace(base_pos=np.zeros(3), base_linvel=np.zeros(3),
        base_quat_wxyz=np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]))


class SprintTurnTracking(unittest.TestCase):
    def test_stalled_turn_is_bounded_and_reversal_releases_old_lag(self):
        c=MotionControl(SETTINGS);command=np.zeros(13,np.float32);command[0]=.3;command[2]=.8
        outputs=[float(c.command(command,state(3.12),'sprint')[2]) for _ in range(100)]
        self.assertAlmostEqual(outputs[0],.8,places=6)
        self.assertAlmostEqual(outputs[-1],1.4,places=6)
        self.assertLessEqual(max(outputs),1.4+1e-6)
        command[2]=-.8
        self.assertAlmostEqual(float(c.command(command,state(3.12),'sprint')[2]),-.8,places=6)
        command[:]=0.
        self.assertAlmostEqual(float(c.command(command,state(3.12),'walking')[2]),0.,places=6)
        c.reset();command[2]=.8
        self.assertAlmostEqual(float(c.command(command,state(-3.12),'sprint')[2]),.8,places=6)

    def test_normal_turns_retain_the_existing_controller(self):
        old=MotionControl({**SETTINGS,'walk_sprint_turn_tracking':False});new=MotionControl(SETTINGS)
        command=np.zeros(13,np.float32);command[0]=.25
        for i in range(120):
            command[2]=.8 if i<40 else (-.8 if i<80 else 0.)
            s=state(3.+i*.008)
            np.testing.assert_array_equal(new.command(command,s,'walking'),old.command(command,s,'walking'))

    def test_batched_controller_matches_wrap_reversal_handoff_and_partial_reset(self):
        settings={**SETTINGS,'walk_path_gain':4.,'walk_path_lookahead':.2,'walk_reanchor_on_start':True}
        reference=[MotionControl(settings),MotionControl(settings)];batch=WalkingControl(2,settings,'cpu')
        for i in range(360):
            if i==240:
                reference[0].reset();batch.reset(torch.tensor([0]))
            commands=np.zeros((2,13),np.float32);commands[:,0]=.3
            commands[:,2]=.8 if i<100 else (-.8 if i<210 else 0.)
            sprint=np.array([i<280,i<150])
            states=[state(3.12+i*.009),state(-3.12-i*.006)]
            for j,s in enumerate(states):s.base_pos[:2]=[i*.001,j*.01]
            actual=batch.command(torch.from_numpy(commands),torch.tensor(np.array([s.base_pos for s in states])),
                torch.tensor(np.array([s.base_quat_wxyz for s in states])),torch.zeros((2,3)),torch.from_numpy(sprint)).numpy()
            expected=np.array([c.command(cmd,s,'sprint' if selected else 'walking') for c,cmd,s,selected in zip(reference,commands,states,sprint)])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
