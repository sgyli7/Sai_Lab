import unittest
import numpy as np
from test_research import fake_world
from sim2sim.research.rewards import Objective
from sim2sim.research.sprint_tasks import commands
from sim2sim.research.world import World
from sim2sim.motion_control import MotionControl
from types import SimpleNamespace


class SprintLearning(unittest.TestCase):
    def test_native_turn_exit_curriculum_keeps_sprinting_after_turn(self):
        for direction,sign in [('nativeleft',1),('nativeright',-1)]:
            with self.assertRaisesRegex(ValueError,'14 seconds'):commands('sprint_030_'+direction)
            tape,selected=commands('sprint_030_'+direction,seconds=14,include_selection=True)
            self.assertEqual(len(tape),700)
            self.assertAlmostEqual(float(tape[350,2]),.8*sign,places=6)
            self.assertTrue(selected[450]);self.assertAlmostEqual(float(tape[450,0]),.3,places=6)
            self.assertAlmostEqual(float(tape[450,2]),0.,places=6)
            self.assertFalse(selected[600]);np.testing.assert_array_equal(tape[600],np.zeros(13))

    def test_path_lookahead_turns_toward_path_and_respects_manual_turns(self):
        import math
        cmd=np.array([.3,0,0,*([0]*10)],np.float32)
        for yaw in [0.,math.pi/2]:
            control=MotionControl({'walk_path_gain':4.,'walk_heading_gain':6.,
                'walk_heading_limit':.6,'walk_path_lookahead':.3})
            state=SimpleNamespace(base_pos=np.zeros(3),base_quat_wxyz=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)]))
            control.command(cmd,state,'sprint')
            state.base_pos[:2]=[-math.sin(yaw)*.03,math.cos(yaw)*.03]
            corrected=control.command(cmd,state,'sprint')
            self.assertAlmostEqual(float(corrected[2]),-6*math.atan2(.03,.3),places=6)
            self.assertAlmostEqual(control.target_yaw,yaw-math.atan2(.03,.3),places=6)
            turn=cmd.copy();turn[2]=.8
            np.testing.assert_array_equal(control.command(turn,state,'sprint'),turn)
            self.assertFalse(control.walk_path_started)

    def test_training_handoff_uses_deployed_normal_speed_and_acceleration(self):
        tape,selected=commands('sprint_030_wfirst',include_selection=True,
            twist_limits={'vmax_x':.25,'vmax_ang':.8,'accel':1.2,'sprint_vmax_x':.9})
        self.assertFalse(selected[100]);self.assertTrue(selected[200]);self.assertFalse(selected[350])
        self.assertAlmostEqual(float(tape[100,0]),.25,places=6)
        self.assertAlmostEqual(float(tape[200,0]),.3,places=6)
        self.assertAlmostEqual(float(tape[350,0]),.25,places=6)
        self.assertAlmostEqual(float(tape[50,0]),.024,places=6)

    def test_composed_controller_uses_actual_skill_selection_not_velocity_magnitude(self):
        from sim2sim.research.sprint_composition import SprintComposition
        tape,selected=commands('sprint_030_release',include_selection=True)
        np.testing.assert_array_equal(tape[150],tape[250])
        self.assertTrue(selected[150]);self.assertFalse(selected[250])
        turn,phase=commands('sprint_030_turnrelease',include_selection=True,ordinary_turn_rate=.8)
        self.assertFalse(phase[250]);self.assertGreater(turn[250,2],0.)
        self.assertAlmostEqual(float(turn[250,2]),.8,places=6)
        for name in ['wfirst','turnrelease','repeat']:
            _,phase=commands('sprint_030_'+name,include_selection=True)
            self.assertTrue(phase.any());self.assertTrue((~phase).any())
        controller=object.__new__(SprintComposition)
        controller.learn_all=False
        controller.actor=lambda obs:np.full((len(obs),14),.75,np.float32)
        controller.counts={'learned':0,'ordinary':0}
        worlds=[SimpleNamespace(sprint_active=lambda:True,obs=lambda:np.zeros(61)),
                SimpleNamespace(sprint_active=lambda:False,obs=lambda:np.zeros(61))]
        proposed=np.full((2,14),-.1,np.float32)
        actual=controller.actions(worlds,proposed)
        np.testing.assert_array_equal(actual[0],proposed[0])
        np.testing.assert_array_equal(actual[1],np.full(14,.75,np.float32))
        np.testing.assert_array_equal(proposed,np.full((2,14),-.1,np.float32))
        self.assertEqual(controller.mask(worlds).tolist(),[True,False])
        self.assertEqual(controller.counts,{'learned':1,'ordinary':1})
        controller.learn_all=True
        np.testing.assert_array_equal(controller.actions(worlds,proposed),proposed)

    def test_ordinary_feedback_override_preserves_sprint_and_legacy_defaults(self):
        state=SimpleNamespace(base_pos=np.zeros(3),base_quat_wxyz=np.array([1.,0,0,0]))
        cmd=np.zeros(13,np.float32);cmd[0]=.3
        controls=[MotionControl({'walk_path_gain':4.,'walk_ordinary_path_gain':6.}),MotionControl({'walk_path_gain':4.})]
        for c in controls:c.command(cmd,state,'walking')
        state.base_pos[1]=.02
        self.assertAlmostEqual(float(controls[0].command(cmd,state,'walking')[1]),-.12,places=6)
        self.assertAlmostEqual(float(controls[0].command(cmd,state,'sprint')[1]),-.08,places=6)
        np.testing.assert_array_equal(controls[1].command(cmd,state,'walking'),controls[1].command(cmd,state,'sprint'))

    def test_entry_fingerprint_detects_models_and_control_changes(self):
        import json,tempfile
        from pathlib import Path
        from sim2sim.research.sprint_entry import SprintEntryBank
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actor=root/'actor.onnx';actor.write_bytes(b'original')
            bank=root/'bank.json';config=dict(version='sprint_entry_v1',walking=str(actor),sprint=str(actor),twist_limits={})
            bank.write_text(json.dumps(config));before=SprintEntryBank.fingerprint(bank)
            actor.write_bytes(b'candidate');changed=SprintEntryBank.fingerprint(bank)
            self.assertNotEqual(before['walking'],changed['walking']);self.assertEqual(before['config'],changed['config'])
            config['twist_limits']['vmax_ang']=.8;bank.write_text(json.dumps(config))
            self.assertNotEqual(changed['config'],SprintEntryBank.fingerprint(bank)['config'])

    def test_restart_can_capture_current_heading_without_resetting_actor_history(self):
        control=MotionControl({'walk_heading_gain':6.,'walk_reanchor_on_start':True})
        state=SimpleNamespace(base_pos=np.zeros(3),base_quat_wxyz=np.array([1.,0,0,0]))
        idle=np.zeros(13,np.float32);forward=idle.copy();forward[0]=.3
        control.command(forward,state,'walking')
        state.base_quat_wxyz=np.array([np.cos(.2),0,0,np.sin(.2)])
        self.assertLess(float(control.command(idle,state,'walking')[2]),0.)
        restarted=control.command(forward,state,'walking')
        self.assertAlmostEqual(float(restarted[2]),0.)
        self.assertAlmostEqual(control.target_yaw,.4)

    def test_feedback_reward_tracks_corrective_lateral_request(self):
        scores={}
        for version in ['sprint_v1','sprint_v2']:
            scores[version]=[]
            for speed in [0.,-.1]:
                world=fake_world('walking');world.executed_command[1]=-.1
                world.features['vel'][1]=speed
                objective=Objective(world,walking_objective=version)
                objective.smooth[1]=speed
                scores[version].append(objective.compute()[0])
        self.assertGreater(scores['sprint_v1'][0],scores['sprint_v1'][1])
        self.assertLess(scores['sprint_v2'][0],scores['sprint_v2'][1])

    def test_feedback_is_sampled_once_per_physical_state(self):
        world=object.__new__(World)
        world.motion=MotionControl({'walk_path_gain':2.})
        world._command_stamp=None;world.t=0.
        world.state=SimpleNamespace(base_pos=np.array([0.,0.,.115]),base_quat_wxyz=np.array([1.,0,0,0]))
        world.requested_command=lambda: np.array([.35,0.,0.,*([0.]*10)],np.float32)
        first=world.command();np.testing.assert_array_equal(first,world.command())
        world.state.base_pos[1]=.04;world.t=.02
        corrected=world.command();self.assertAlmostEqual(float(corrected[1]),-.08,places=6)
        np.testing.assert_array_equal(corrected,world.command())
        world.requested_command=lambda: np.array([.35,0.,.8,*([0.]*10)],np.float32)
        world.t=.04;turn=world.command()
        self.assertEqual(float(turn[1]),0.);self.assertFalse(world.motion.walk_path_started)

    def test_target_speed_has_progress_gradient_and_overrun_cost(self):
        rewards=[]
        for speed in [0.,.1,.2,.3,.4,.5]:
            world=fake_world('walking');world.executed_command[0]=.4
            world.features['vel'][0]=speed
            objective=Objective(world,walking_objective='sprint_v1')
            objective.smooth[0]=speed
            rewards.append(objective.compute()[0])
        self.assertTrue(all(a<b for a,b in zip(rewards[:4],rewards[1:5])))
        self.assertLess(rewards[5],rewards[4])

    def test_fall_is_terminal_and_heading_is_critic_state(self):
        world=fake_world('walking');objective=Objective(world,walking_objective='sprint_v1')
        world.features['yaw']=.2
        np.testing.assert_allclose(objective.extra()[8:10],[np.sin(-.2),np.cos(-.2)],atol=1e-7)
        world.features['tilt']=61.
        self.assertTrue(objective.compute()[1])
        self.assertFalse(Objective(world).compute()[1])

    def test_curriculum_has_idle_acceleration_both_turns_and_release(self):
        for direction,turn in [('straight',0.),('left',.8),('right',-.8)]:
            tape=commands('sprint_040_'+direction)
            self.assertEqual(tape.shape,(500,13))
            np.testing.assert_allclose(tape[80,:3],[.4,0,turn],atol=1e-6)
            np.testing.assert_array_equal(tape[0],np.zeros(13))
            np.testing.assert_array_equal(tape[-1],np.zeros(13))
        self.assertAlmostEqual(float(commands('sprint_040_release')[300,0]),.3)
