import unittest
from pathlib import Path
import numpy as np
import mujoco
from sim2sim.paths import load_robot_json
from sim2sim.research.joint_compliance import add_joint_compliance, add_joint_compliance_direct


class JointComplianceTests(unittest.TestCase):
    def test_zero_deflection_preserves_robot_geometry_mass_and_actuators(self):
        cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
        original=mujoco.MjSpec.from_file(cfg['mjcf']).compile()
        spec=mujoco.MjSpec.from_file(cfg['mjcf'])
        report=add_joint_compliance(spec,cfg['godot_spec'],.01)
        modified=spec.compile()
        self.assertEqual(modified.nu,original.nu)
        np.testing.assert_array_equal(modified.body_mass,original.body_mass)
        np.testing.assert_array_equal(modified.body_inertia,original.body_inertia)
        self.assertEqual(modified.neq-original.neq,42)
        old=mujoco.MjData(original);new=mujoco.MjData(modified)
        rng=np.random.default_rng(956001)
        for _ in range(8):
            angles=np.array(cfg['home'])+rng.normal(0,.03,14)
            for model,data in [(original,old),(modified,new)]:
                qi=[model.jnt_qposadr[model.actuator_trnid[i,0]] for i in range(model.nu)]
                data.qpos[qi]=angles
                mujoco.mj_kinematics(model,data)
            np.testing.assert_allclose(new.xipos,old.xipos,atol=1e-12,rtol=0)
            np.testing.assert_allclose(new.ximat,old.ximat,atol=1e-12,rtol=0)
        # A passive coordinate can move the foot while every commanded hinge
        # stays fixed. This is the state dimension missing from the rigid proxy.
        name=next(n for n in report['joint_names'] if n=='proxy_compliance_left_ankle_0')
        j=mujoco.mj_name2id(modified,mujoco.mjtObj.mjOBJ_JOINT,name)
        bid=mujoco.mj_name2id(modified,mujoco.mjtObj.mjOBJ_BODY,'ankle_left')
        before=new.xipos[bid].copy()
        new.qpos[modified.jnt_qposadr[j]]=.002
        mujoco.mj_kinematics(modified,new)
        self.assertAlmostEqual(np.linalg.norm(new.xipos[bid]-before),.002,places=10)

    def test_invalid_time_is_rejected_before_asset_mutation(self):
        for value in [0.,.005,-.01,float('nan'),float('inf'),.1]:
            with self.assertRaises(ValueError):
                add_joint_compliance(None,'not-read.json',value)

    def test_direct_stiffness_is_compiled_as_negative_solref(self):
        cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
        spec=mujoco.MjSpec.from_file(cfg['mjcf'])
        add_joint_compliance_direct(spec,cfg['godot_spec'],20000,200)
        model=spec.compile()
        np.testing.assert_array_equal(model.eq_solref,np.tile([-20000, -200],(42,1)))
        self.assertEqual(model.nu,14)
        for stiffness,damping in [(0,200),(20000,-1),(float('nan'),200),(20000,float('inf'))]:
            with self.assertRaises(ValueError):
                add_joint_compliance_direct(None,'not-read.json',stiffness,damping)
