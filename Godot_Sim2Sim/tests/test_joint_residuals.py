import json
from pathlib import Path
import unittest
import numpy as np
import mujoco
from sim2sim.coords import mat_to_quat_wxyz
from sim2sim.paths import load_robot_json
from sim2sim.research.joint_residuals import JointResidualModel, axis_rotation


class JointResidualTests(unittest.TestCase):
    def setUp(self):
        cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
        self.spec=json.loads(Path(cfg['godot_spec']).read_text())
        self.measure=JointResidualModel(self.spec)
        m=mujoco.MjModel.from_xml_path(cfg['mjcf']);d=mujoco.MjData(m)
        self.angles=np.asarray(cfg['home'])+np.random.default_rng(957000).normal(0,.02,14)
        qi=[m.jnt_qposadr[m.actuator_trnid[i,0]] for i in range(14)];d.qpos[qi]=self.angles
        mujoco.mj_kinematics(m,d)
        self.poses=[dict(name=m.body(i).name,pos=d.xipos[i].copy(),quat=mat_to_quat_wxyz(d.ximat[i].reshape(3,3))) for i in range(1,m.nbody)]

    def test_ideal_hinges_have_no_constraint_residual(self):
        poses=self.measure.body_poses(self.poses);r=self.measure.measure(poses,self.angles)
        np.testing.assert_allclose(r['translation_parent'],0,atol=1e-14)
        np.testing.assert_allclose(r['swing_parent'],0,atol=1e-13)
        np.testing.assert_allclose(r['q_error'],0,atol=1e-13)

    def perturb_subtree(self, translate=False, rotate=False):
        h=next(x for x in self.measure.hinges if x.name=='left_knee')
        poses=self.measure.body_poses(self.poses);pp,rp=poses[h.parent]
        anchor=pp+rp@h.anchor_parent
        shift=rp@np.array([.001,.002,-.001]) if translate else np.zeros(3)
        axis=np.cross(h.axis,[1.,0,0])
        if np.linalg.norm(axis)<.1:axis=np.cross(h.axis,[0.,1,0])
        rot=rp@axis_rotation(axis,.003)@rp.T if rotate else np.eye(3)
        children={h.child}
        for item in self.measure.hinges:
            if item.parent in children:children.add(item.child)
        for pose in self.poses:
            if pose['name'] in children:
                from sim2sim.research.joint_residuals import rotation
                pose['pos']=anchor+rot@(pose['pos']-anchor)+shift
                pose['quat']=mat_to_quat_wxyz(rot@rotation(pose['quat']))
        return h

    def check_reconstruction(self, translation, swing):
        poses=self.measure.body_poses(self.poses);r=self.measure.measure(poses,self.angles)
        np.testing.assert_allclose(r['q_error'],0,atol=1e-13)
        recovered=self.measure.reconstruct(poses,self.angles,r,translation=translation,swing=swing)
        for pose in self.poses:
            np.testing.assert_allclose(recovered[pose['name']][0],pose['pos'],atol=1e-13)
        return r

    def test_known_translation_is_localized_to_perturbed_hinge(self):
        h=self.perturb_subtree(translate=True);r=self.check_reconstruction(True,False)
        i=self.measure.hinges.index(h)
        np.testing.assert_allclose(r['translation_parent'][i],[.001,.002,-.001],atol=1e-13)
        np.testing.assert_allclose(np.delete(r['translation_parent'],i,axis=0),0,atol=1e-13)
        np.testing.assert_allclose(r['swing_parent'],0,atol=1e-13)

    def test_constrained_axis_rotation_is_distinct_from_translation(self):
        h=self.perturb_subtree(rotate=True);r=self.check_reconstruction(False,True)
        np.testing.assert_allclose(r['translation_parent'],0,atol=1e-13)
        self.assertAlmostEqual(np.linalg.norm(r['swing_parent'][self.measure.hinges.index(h)]),.003,places=12)
