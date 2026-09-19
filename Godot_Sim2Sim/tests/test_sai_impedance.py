"""Contract and gravity support sign tests using the real articulated model."""
import importlib.util
import unittest
import numpy as np

SAI_AVAILABLE=importlib.util.find_spec('sai_agent') is not None

@unittest.skipUnless(SAI_AVAILABLE,'Install the optional Sai extra')
class ImpedanceTests(unittest.TestCase):
    def test_motor_contract_matches_released_adapter_when_disabled(self):
        import mujoco
        from sai_agent.paths import resource_root
        from sai_agent.runtime import JointAdapter
        from sim2sim.sai_compliance import apply_impedance
        model=mujoco.MjModel.from_xml_path(str(resource_root()/'models/full/locomotion-articulated.xml'))
        adapter=JointAdapter(model);data=mujoco.MjData(model)
        target=np.random.default_rng(91).normal(0,.5,16)
        adapter.apply(data,target);expected=data.ctrl.copy()
        result=dict(target_leg=target,impedance_contract='sai-joint-impedance-v1',leg_kp=np.full(16,80.),leg_kd=np.full(16,2.),leg_feedforward=np.zeros(16))
        apply_impedance(adapter,data,result)
        np.testing.assert_allclose(data.ctrl,expected,atol=1e-12)

    def test_static_support_force_and_moment(self):
        from sai_agent.paths import resource_root
        from sim2sim.sai_compliance import CompliantController
        from scripts.sai_mujoco_motion import rollout
        c=CompliantController(resource_root(),[30,1.5,1,0,0,.2])
        # A published control observation from the unmodified CPU articulation.
        from sim2sim.sai_controller import MotionController
        state=rollout(MotionController(resource_root()),'stop',.02)['samples'][-1]
        state.update(command=[0,0,0],terrain_heights=[0.]*24,terrain_path_heights=[0.]*15,wheel_ground_heights=[0.]*4)
        result=c.command(state);weights=np.array(result['stance_weights'])
        self.assertTrue((weights>=0).all());self.assertAlmostEqual(weights.sum(),1.,places=6)
        xy=c.data.xpos[c.wheel_ids,:2]-c.data.subtree_com[c.model.body('chassis').id,:2]
        np.testing.assert_allclose(weights@xy,[0,0],atol=1e-7)
        self.assertAlmostEqual(result['support_force_N'],c.robot_mass*9.81,places=6)


@unittest.skipUnless(SAI_AVAILABLE,'Install the optional Sai extra')
class ProfileTests(unittest.TestCase):
    def test_packaged_controller_matches_experimental_motor_commands(self):
        import json,tempfile
        from pathlib import Path
        from sai_agent.paths import resource_root
        from sim2sim.sai_controller import MotionController
        from sim2sim.sai_compliance import BASE_GEOMETRY,CompliantController
        from scripts.sai_mujoco_motion import rollout
        params=[40.91158566171846,1.1964986190787574,.9392462048772455,246.57832155769,64.99753838358409,.11777877415577209]
        state=rollout(MotionController(resource_root(),suspension_profile='off'),'stop',.02)['samples'][-1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json';path.write_text(json.dumps(dict(schema_version=2,id='test',parameters=params,geometry_parameters=BASE_GEOMETRY,descent_control='contact_following')))
            actual=MotionController(resource_root(),suspension_profile=str(path));expected=CompliantController(resource_root(),params)
            for i in range(12):
                s=dict(state,time=i*.02,command=[.5,0.,0.],terrain_heights=[0.]*24,terrain_path_heights=[0.]*15,wheel_ground_heights=[.002*(i%3),0.,-.001,0.])
                a,b=actual.command(s),expected.command(s)
                for key in ['target_leg','leg_kp','leg_kd','leg_feedforward','stance_weights','support_force_N']:
                    np.testing.assert_array_equal(a[key],b[key],err_msg=key)

    def test_legacy_state_without_extra_rays_keeps_original_motor_contract(self):
        from sai_agent.paths import resource_root
        from sim2sim.sai_compliance import CompliantController
        from sim2sim.sai_controller import MotionController
        from scripts.sai_mujoco_motion import rollout
        state=rollout(MotionController(resource_root(),suspension_profile='off'),'stop',.02)['samples'][-1]
        c=CompliantController(resource_root(),[30.,1.5,1.,0.,0.,.2])
        result=c.command(state)
        self.assertNotIn('impedance_contract',result)

if __name__=='__main__':unittest.main()
