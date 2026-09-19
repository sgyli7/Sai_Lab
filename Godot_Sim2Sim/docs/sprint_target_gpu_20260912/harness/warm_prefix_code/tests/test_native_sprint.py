import json
import math
from pathlib import Path
import tempfile
import unittest

from sim2sim.research.native_sprint import dataset


class NativeSprintDatasetTests(unittest.TestCase):
    def fixture(self, directory, *, heights=(.125,.125,.125), velocities=(0.,1.,0.), sigma=.02, explore_after=0.):
        root=Path(directory)
        rows=[]
        for i,(z,v) in enumerate(zip(heights[:-1],velocities[:-1])):
            rows.append(dict(t=i*.02,obs=[0.]*61,action=[.2]*14,last_action=[0.]*14 if i==0 else [.2]*14,
                             requested_command=[0.]*13,command=[0.]*13,skill='sprint',
                             body=dict(base_pos=[0.,0.,z],base_quat=[1.,0.,0.,0.],base_linvel=[v,0.,0.]),
                             raw=dict(base_angvel_local=[0.,0.,0.])))
        raw=dict(t=len(rows)*.02,q=[0.]*14,qd=[0.]*14,base_pos=[0.,0.,heights[-1]],
                 base_quat=[1.,0.,0.,0.],base_linvel=[velocities[-1],0.,0.],base_angvel_local=[0.,0.,0.])
        trace=root/'trace.json';case=root/'case.json'
        trace.write_text(json.dumps(dict(rows=rows,summary=dict(training_exploration=sigma,training_exploration_after=explore_after,models={},final_raw=raw))))
        case.write_text(json.dumps(dict(case='sprint_shift_first')))
        rollouts=dict(rows=[dict(trace=str(trace),case=str(case),steps=len(rows),sigma=sigma,explore_after=explore_after)])
        deployment=dict(policies={},control_config=dict(walk={}),robots=dict(walk=dict(ipos=[0.,0.,0.],iquat=[1.,0.,0.,0.])))
        return rollouts,deployment

    def test_reward_uses_response_after_action_and_final_state_terminates(self):
        with tempfile.TemporaryDirectory() as folder:
            d,a=dataset(*self.fixture(folder),'cpu')
        self.assertAlmostEqual(float(d['reward'][0,0]),math.exp(-4)+.5-.05*14*.2**2,places=6)
        self.assertAlmostEqual(float(d['reward'][1,0]),1.5,places=6)
        self.assertEqual(d['done'][:,0].tolist(),[False,True])
        self.assertEqual(a['rows'],2)

    def test_fall_transition_kept_but_subsequent_recovery_not_used(self):
        with tempfile.TemporaryDirectory() as folder:
            d,a=dataset(*self.fixture(folder,heights=(.125,.04,.125,.125),velocities=(0.,0.,0.,0.)),'cpu')
        self.assertEqual(d['valid'][:,0].tolist(),[True,False,False])
        self.assertEqual(d['mask'][:,0].tolist(),[True,False,False])
        self.assertTrue(bool(d['done'][0,0]))
        self.assertEqual(a['discarded_after_fall'],2)
        self.assertLess(float(d['reward'][0,0]),0.)

    def test_deterministic_controls_are_not_stochastic_training_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'stochastic'):
                dataset(*self.fixture(folder,sigma=0.),'cpu')

    def test_deterministic_prefix_is_available_to_critic_but_not_policy_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            d,a=dataset(*self.fixture(folder,explore_after=.02),'cpu')
        self.assertEqual(d['valid'][:,0].tolist(),[True,True])
        self.assertEqual(d['sprint'][:,0].tolist(),[True,True])
        self.assertEqual(d['mask'][:,0].tolist(),[False,True])
        self.assertEqual(a['deterministic_prefix_sprint_rows'],1)
