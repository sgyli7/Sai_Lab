"""Continuous terrain must not select a periodic stair gait or reduce the command."""
import unittest
import importlib.util
import numpy as np
SAI_AVAILABLE = importlib.util.find_spec("sai_agent") is not None
if SAI_AVAILABLE:
    from sai_agent.paths import resource_root
    from sim2sim.sai_controller import MotionController
    from scripts.sai_mujoco_motion import rollout

@unittest.skipUnless(SAI_AVAILABLE, "Install the optional Sai extra")
class SuspensionRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = rollout(MotionController(resource_root()), 'stop', .02)['samples'][-1]

    def request(self, height, *, dense=True, **extra):
        s = dict(self.state, command=[.5, 0., 0.],
                 terrain_heights=[height(x,y) for x in np.arange(8)*.18-.36 for y in [-.24,0.,.24]],
                 terrain_path_heights=[height(x,y) for x in [-.18,0.,.18,.36,.54] for y in [-.16,0.,.16]])
        if dense:
            s['terrain_edge_heights']=[height(x,y) for x in np.arange(46)*.02-.30 for y in [-.146,0.,.146]]
        s.update(extra)
        return s

    def test_continuous_terrain_keeps_requested_speed(self):
        terrains = [lambda x,y: .04*x, lambda x,y: .25*x+.13*y,
                    lambda x,y: .02*np.sin(5*x+2*y), lambda x,y: .12*abs(x-.21)]
        for h in terrains:
            with self.subTest(terrain=h):
                r=MotionController(resource_root()).command(self.request(h))
                self.assertNotEqual(r['stage'],'stairs')
                self.assertAlmostEqual(r['policy_observation'][9],.5)

    def test_legacy_linear_ramp(self):
        r=MotionController(resource_root()).command(self.request(lambda x,y: .04*x, dense=False))
        self.assertNotEqual(r['stage'],'stairs')

    def test_real_edges_and_small_seams(self):
        for edge in [.055,.17,.315,.505]:
            for rise in [.02,.04,-.02,-.04]:
                with self.subTest(edge=edge,rise=rise):
                    r=MotionController(resource_root()).command(self.request(lambda x,y: .03*x+rise*(x>edge)))
                    self.assertEqual(r['stage'],'stairs' if rise > 0 else 'descending')
                    self.assertAlmostEqual(r['policy_observation'][9], .12 if rise > 0 else .16, places=6)
        r=MotionController(resource_root()).command(self.request(lambda x,y: .006*(x>.17)))
        self.assertNotEqual(r['stage'],'stairs')

    def test_recovery_crouch_and_reverse(self):
        c=MotionController(resource_root())
        step=self.request(lambda x,y:.02*(x>.17))
        self.assertEqual(c.command(step)['stage'],'stairs')
        self.assertNotEqual(c.command(self.request(lambda x,y:0.))['stage'],'stairs')
        self.assertEqual(c.command(dict(step,command=[.5,0.,1.]))['stage'],'crouch_blocked')
        self.assertNotEqual(c.command(dict(step,command=[-.5,0.,0.]))['stage'],'stairs')

    def test_bad_dense_scan_rejected(self):
        for bad in [[0.], [float('nan')]*138]:
            with self.assertRaises(ValueError):
                MotionController(resource_root()).command(self.request(lambda x,y:0.,terrain_edge_heights=bad))


@unittest.skipUnless(SAI_AVAILABLE, "Install the optional Sai extra")
class SuspensionTargetTests(unittest.TestCase):
    def setUp(self):
        from sim2sim.sai_suspension import Suspension
        self.s = Suspension([1.,.5,.08])
        self.result=dict(stage='rolling',effective_crouch=0.,target_leg=list(np.zeros(16)))
        self.state=dict(wheel_ground_heights=[0.]*4,base_rotation_columns=np.eye(3).tolist())

    def test_flat_is_exact_and_wheel_targets_unchanged(self):
        self.result['target_leg']=list(np.linspace(-1,1,16))
        out=self.s.apply(self.result,self.state)
        np.testing.assert_array_equal(out['target_leg'],self.result['target_leg'])
        for _ in range(60):out=self.s.apply(self.result,dict(self.state,wheel_ground_heights=[.03,-.02,.01,-.01]))
        np.testing.assert_array_equal(np.array(out['target_leg'])[3::4],np.array(self.result['target_leg'])[3::4])
        self.assertLessEqual(max(abs(v) for v in out['suspension_offset_m']),.025)

    def test_height_datum_does_not_change_suspension(self):
        from sim2sim.sai_suspension import Suspension
        a=self.s.apply(self.result,dict(self.state,wheel_ground_heights=[.02,0.,.01,0.]))
        b=Suspension([1.,.5,.08]).apply(self.result,dict(self.state,wheel_ground_heights=[10.02,10.,10.01,10.]))
        np.testing.assert_allclose(a['target_leg'],b['target_leg'],atol=1e-12)

    def test_stairs_are_exact_and_reset_correction(self):
        self.s.apply(self.result,dict(self.state,wheel_ground_heights=[.02,0.,.01,0.]))
        stair=dict(self.result,stage='stairs')
        self.assertIs(self.s.apply(stair,self.state),stair)
        np.testing.assert_array_equal(self.s.offset,np.zeros(4))

    def test_invalid_ground_rejected(self):
        for bad in [[0.],[0.,0.,0.,float('nan')]]:
            with self.assertRaises(ValueError):self.s.apply(self.result,dict(self.state,wheel_ground_heights=bad))



class DescentTerrainTests(unittest.TestCase):
    def test_only_validated_downward_edges_select_rolling_descent(self):
        from sim2sim.sai_terrain import EDGE_X, descending_in_path
        for height,expected in [(-.02,True),(-.04,True),(.02,False),(.04,False),(-.06,False)]:
            h=np.repeat(np.where(EDGE_X>.2,height,0.)[:,None],3,axis=1)
            self.assertEqual(descending_in_path({'terrain_edge_heights':h.ravel().tolist()}),expected)
        plane=np.repeat((EDGE_X*.3)[:,None],3,axis=1)
        self.assertFalse(descending_in_path({'terrain_edge_heights':plane.ravel().tolist()}))

if __name__=='__main__': unittest.main()
