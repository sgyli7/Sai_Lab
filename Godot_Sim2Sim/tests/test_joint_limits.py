"""Physical end stops must be invariant to reset pose and previous episode."""
import unittest
from pathlib import Path
import numpy as np
from sim2sim.paths import load_robot_json,sim2sim_root
from sim2sim.train.reset_poses import HomePoseSampler
from sim2sim.backends.godot_backend import GodotBackend

class HingeLimits(unittest.TestCase):
    def test_asymmetric_limits_survive_repeated_nonzero_resets(self):
        cfg=load_robot_json(sim2sim_root()/"robots/microduck.json")
        sampler=HomePoseSampler(cfg)
        be=GodotBackend(Path(cfg["godot_spec"]),current_limit_a=1.75)
        try:
            # Includes an asymmetric neck range, a nonzero hip home, and ankle.
            # Reuse the same process, alternating the previous end-stop state.
            for joint in (1,5,4):
                for initial in (0.,float(sampler.home[joint]),.2):
                    q=sampler.home.copy();q[joint]=initial
                    poses,_,_=sampler.sample(np.random.default_rng(0),yaw_range=(0,0),joint_noise_rad=0,z=.5,q_base=q)
                    for direction,bound in ((-1,sampler.joint_lo[joint]),(1,sampler.joint_hi[joint])):
                        with self.subTest(joint=joint,initial=initial,direction=direction):
                            be.reset(ctrl=sampler.home,bodies=poses,pin_base=True)
                            ctrl=sampler.home.copy();ctrl[joint]=bound+direction*.4
                            for _ in range(80):
                                be.send_step(ctrl,n_substeps=4,report="lite");state=be.recv_step()
                            self.assertAlmostEqual(state.q[joint],bound,delta=1e-3)
        finally:be.close();sampler.close()

if __name__=="__main__":unittest.main()
