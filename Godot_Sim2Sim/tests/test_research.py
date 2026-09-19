"""Regression checks for task semantics, real telemetry and deployable actors."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sim2sim.research.tasks import TASKS,command
from sim2sim.research.models import Policy,export_policy,parity
from sim2sim.research.world import World
from sim2sim.research.rewards import Objective,EXTRA_DIM
from sim2sim.research.evaluate import summarize

def fake_world(name):
    f=dict(z=.115,xy=np.zeros(2),rot=np.eye(3),yaw=0.,up=1.,tilt=0.,
           vel=np.zeros(3),gyro=np.zeros(3),contact=np.ones(2),foot_pos=np.zeros((2,3)),
           foot_vel=np.zeros((2,3)),mouth_pos=np.array([.08,0,.17]),mouth_down=.5,
           head_contact=False,head_up=1.,supported=True,ball_pos=np.array([.09,.042,.035]),
           ball_vel=np.zeros(3),kick_contacts=[])
    w=SimpleNamespace(task=TASKS[name],features=f,t=.02,home=np.zeros(14),last=np.zeros(14),old_last=np.zeros(14),heading=np.array([1.,0.]),motion=None)
    w.state=SimpleNamespace(q=np.zeros(14),qd=np.zeros(14),base_pos=np.array([0,0,.115]))
    w.executed_command=command(w.task,0)
    return w

class TaskSemantics(unittest.TestCase):
    def test_native_roller_command_uses_throttle_and_actual_heading_error(self):
        from sim2sim.research.roller_tasks import command as roller_command
        w=fake_world('roller');w.condition='brake';w.roller_target_yaw=.5;w.features['yaw']=.2
        cmd=roller_command(w);self.assertEqual(cmd[0],-.5);self.assertAlmostEqual(float(cmd[2]),.3,places=6)
        w.features['yaw']=.6
        self.assertAlmostEqual(float(roller_command(w)[2]),-.1,places=6)
        w.condition='coast';self.assertEqual(roller_command(w)[0],0.)

    def test_native_roller_brake_rewards_stopping_and_penalizes_reverse(self):
        w=fake_world('roller');w.roller_contract=True;w.roller_target_yaw=0.;w.executed_command[0]=-.5
        stopped=Objective(w).compute()[2]
        w.features['vel']=np.array([-.3,0.,0.]);r=Objective(w);r.smooth=np.array([-.3,0.,0.])
        reverse=r.compute()[2]
        self.assertGreater(stopped['brake'],reverse['brake']);self.assertLess(reverse['reverse'],0.)
        self.assertEqual(stopped['push'],0.)

    def test_brake_speed_tolerance_only_changes_the_declared_reward_term(self):
        w=fake_world('roller');w.roller_contract=True;w.roller_target_yaw=0.;w.executed_command[0]=-.5
        w.features['vel']=np.array([.1,0.,0.])
        def terms(params):
            reward=Objective(w,params=params);reward.smooth=np.array([.1,0.,0.])
            return reward.compute()[2]
        broad=terms({});narrow=terms({'brake_velocity_variance':.0025})
        self.assertLess(narrow['brake'],broad['brake'])
        for key in broad:
            if key!='brake':self.assertEqual(narrow[key],broad[key])
        for invalid in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):Objective(w,params={'brake_velocity_variance':invalid})
        with self.assertRaises(ValueError):
            Objective(fake_world('walking'),params={'brake_velocity_variance':.0025})

    def test_dense_brake_cost_is_explicit_and_inactive_during_push_or_coast(self):
        w=fake_world('roller');w.roller_contract=True;w.roller_target_yaw=0.
        for throttle in [-.5,0.,.6]:
            w.executed_command[0]=throttle;w.features['vel']=np.array([.3,.4,0.])
            reward=Objective(w,{'brake_speed_cost':4.});reward.smooth=np.array([.3,.4,0.])
            terms=reward.compute()[2]
            self.assertAlmostEqual(terms['brake_speed_cost'],-2. if throttle<0 else 0.)
            self.assertNotIn('brake_speed_cost',Objective(w).compute()[2])

    def test_reflection_preserves_phase_and_heading_cosine(self):
        from sim2sim.research.mirror import reflect_obs
        x=np.zeros(61,np.float32);x[48:51]=[.2,.3,.4]
        np.testing.assert_array_equal(reflect_obs(x,heading_input=True)[48:51],[np.float32(.2),np.float32(-.3),np.float32(.4)])
        for task in ('ground_pick','roller_crouch'):
            np.testing.assert_array_equal(reflect_obs(x,task=task)[48:50],x[48:50])
        np.testing.assert_array_equal(reflect_obs(x)[48:51],x[48:51]*[1,-1,-1])

    def test_relative_heading_survives_forward_inversion_and_reflects_correctly(self):
        from sim2sim.policy_time import time_command
        a=.7;heading=np.array([np.cos(a),np.sin(a)])
        for error in [-.4,0.,.4]:
            yaw=a+error;c,s=np.cos(yaw),np.sin(yaw);rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            for pitch in [0,np.pi/2,np.pi,1.5*np.pi,2*np.pi]:
                c,s=np.cos(pitch),np.sin(pitch);ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
                cmd=time_command(1.,5.,rz@ry,heading)
                np.testing.assert_allclose(cmd[1:3],[np.sin(error),np.cos(error)],atol=1e-7)

    def test_roll_yaw_spin_cost_preserves_forward_rotation(self):
        w=fake_world('roulade');r=Objective(w,{'yaw_spin':2.})
        w.features['rot']=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        w.features['gyro']=np.array([0.,8.,0.])
        self.assertEqual(r.compute()[2]['yaw_spin'],0.)
        w.features['rot']=np.eye(3);w.features['gyro']=np.array([0.,0.,100.])
        self.assertEqual(r.compute()[2]['yaw_spin'],-8.)

    def test_kick_quality_target_caps_speed_incentive_and_charges_spin(self):
        w=fake_world('kick_left');w.features['ball_vel']=np.array([.9,0.,0.]);w.features['gyro']=np.array([0.,0.,2.])
        standard=Objective(w).compute()[2]
        quality=Objective(w,{'yaw_spin':2.},params={'ball_speed_target':.7}).compute()[2]
        self.assertLess(quality['ball_progress'],standard['ball_progress'])
        self.assertAlmostEqual(quality['overspeed'],-.8)
        self.assertEqual(quality['yaw_spin'],-8.)

    def test_motion_feedback_uses_actual_pose_and_retained_source_time(self):
        w=fake_world("roulade")
        with self.assertRaises(ValueError):Objective(w,{"motion_pose":1.})
        w.time_input_s=5.;w.time_offset=1.25;r=Objective(w,{"motion_pose":1.})
        target=r.motion.sample(w.t+w.time_offset);w.state.q=target["q"].copy()
        exact=r.compute()[2]["motion_pose"]
        w.state.q+=1.;wrong=r.compute()[2]["motion_pose"]
        self.assertAlmostEqual(exact,10.);self.assertLess(wrong,.1)

    def test_play_and_training_agree_on_one_shot_time(self):
        from sim2sim.play_input import PlayBrain
        from sim2sim.policy_time import time_command
        for skill in ('roulade','kick_left','kick_right'):
            brain=PlayBrain()
            for k in range(20):
                out=brain.tick(set(),[skill] if k==0 else [],.02)
                self.assertEqual(out.policy,skill)
                duration=brain.roulade_duration if skill=='roulade' else brain.kick_duration
                elapsed=duration-brain.behavior_t
                np.testing.assert_allclose(time_command(elapsed,5)[0],k*.02/5,atol=1e-7)
                np.testing.assert_array_equal(out.command,0.) # Existing zero-command policies stay unchanged.
        self.assertEqual(time_command(7.,5.)[0],1.)

    def test_interactive_command_tapes_are_reproducible_bounded_and_ramped(self):
        from sim2sim.research.schedules import random_schedule,scheduled_command
        for skill in ("walking","roller"):
            task=TASKS[skill];a=random_schedule(task,62000);b=random_schedule(task,62000)
            np.testing.assert_array_equal([x[0] for x in a],[x[0] for x in b])
            np.testing.assert_array_equal([x[1] for x in a],[x[1] for x in b])
            values=np.array([scheduled_command(a,t) for t in np.arange(0,task.seconds,.02)])
            self.assertEqual(values.shape[1],13)
            self.assertLessEqual(values[:,0].max(),.6 if skill=="roller" else .4)
            self.assertGreaterEqual(values[:,0].min(),-.3)
            self.assertLessEqual(np.abs(values[:,2]).max(),1.)
            np.testing.assert_array_equal(values[:,3:],0.)
            if skill=="roller":np.testing.assert_array_equal(values[:,1],0.)
            for i,(t,target) in enumerate(a[1:],1):
                np.testing.assert_allclose(scheduled_command(a,t)[:3],a[i-1][1],atol=1e-6)
                np.testing.assert_allclose(scheduled_command(a,t+.1)[:3],target,atol=1e-6)

    def test_tracking_variance_changes_tolerance_without_changing_task_command(self):
        w=fake_world("walking");w.executed_command[2]=.4
        broad=Objective(w).compute()[2]["yaw"]
        narrow=Objective(w,params={"yaw_variance":.04}).compute()[2]["yaw"]
        self.assertLess(narrow,broad)
        with self.assertRaises(ValueError):Objective(w,params={"yaw_variance":0})

    def test_phase_commands_and_posture_are_not_velocity(self):
        c=command(TASKS["roller_crouch"],1.25)
        np.testing.assert_allclose(c[:2],[0,1],atol=1e-6)
        w=fake_world("sitstand");obj=Objective(w)
        first=obj.compute()[0];w.features["vel"][:]=[1.,0,0]
        self.assertEqual(obj.compute()[0],first)

    def test_inverted_is_not_upright(self):
        w=fake_world("standing");r=Objective(w);upright=r.compute()[2]["upright"]
        w.features["up"]=-1.;w.features["tilt"]=180.
        reward,terminated,terms=r.compute()
        self.assertGreater(upright,terms["upright"]+1.9)
        self.assertTrue(terminated)

    def test_kick_requires_real_correct_foot_contact(self):
        w=fake_world("kick_left");r=Objective(w)
        w.features["foot_vel"][0]=[2,0,1]
        terms=r.compute()[2]
        self.assertEqual(terms["touch"],0);self.assertEqual(terms["ball_forward"],0)
        w.features["kick_contacts"]=["ankle_left"];w.features["ball_vel"][0]=.6
        terms=r.compute()[2]
        self.assertGreater(terms["touch"],0);self.assertGreater(terms["ball_forward"],0)
        self.assertEqual(r.extra().shape,(EXTRA_DIM,))

    def test_rocking_does_not_count_as_full_roll(self):
        rows=[]
        for k in range(250):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.zeros(2),up=-1. if k==30 else 1.,
                 tilt=180. if k==30 else 0.,yaw=0.,vel=np.zeros(3),gyro=np.array([0,2. if k%20<10 else -2.,0]),
                 contact=np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),q=np.zeros(14),mouth_z=.1,mouth_down=0.,
                 head_contact=k==20,head_up=-1.,supported=True,lateral_z=0.,ball_pos=np.zeros(3),ball_vel=np.zeros(3),correct_kick=False,wrong_kick=False))
        result=summarize(TASKS["roulade"],rows,np.array([1,0]))
        self.assertTrue(result["ordered_roll_events"])
        self.assertLess(result["supported_forward_rotation"],.5)
        self.assertFalse(result["success"])

    def test_roll_reward_stops_at_one_revolution(self):
        w=fake_world("roulade");r=Objective(w);r.net=7.;r.frontier=2*np.pi;r.pivot=True
        w.features["gyro"][1]=4.
        self.assertEqual(r.compute()[2]["forward_progress"],0)

    def test_excess_rotation_penalty_is_bounded(self):
        w=fake_world("roulade");r=Objective(w,{"over_rotation":5})
        r.net=100.;r.frontier=2*np.pi
        self.assertEqual(r.compute()[2]["over_rotation"],-5.)

    def test_fall_then_recovery_is_not_a_clean_kick(self):
        rows=[]
        for k in range(250):
            fallen=80<k<130
            rows.append(dict(time=.02*(k+1),z=.03 if fallen else .115,xy=np.zeros(2),up=-1. if fallen else 1.,
                 tilt=180. if fallen else 0.,yaw=0.,vel=np.zeros(3),gyro=np.zeros(3),
                 contact=np.zeros(2) if fallen else np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),q=np.zeros(14),mouth_z=.1,mouth_down=0.,
                 head_contact=False,head_up=1.,supported=True,lateral_z=0.,ball_pos=np.array([k*.004,0,.035]),ball_vel=np.array([.2,0,0]),correct_kick=k==1,wrong_kick=False))
        result=summarize(TASKS["kick_left"],rows,np.array([1,0]))
        self.assertTrue(result["final_standing"])
        self.assertTrue(result["correct_foot_contact"])
        self.assertFalse(result["success"])

    def test_slow_cumulative_idle_drift_is_not_a_stop(self):
        rows=[]
        for k in range(500):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.array([k*.0006,0]),up=1.,tilt=0.,
                yaw=k*.001,vel=np.array([.03,0,0]),gyro=np.array([0,0,.05]),contact=np.ones(2),
                cmd=np.zeros(13),actions=np.zeros(14),head_contact=False))
        result=summarize(TASKS["walking"],rows,np.array([1,0]))
        self.assertLess(result["vel_err_1s"],.1)
        self.assertLess(result["yaw_err_1s"],.15)
        self.assertGreater(result["idle_displacement"],.25)
        self.assertGreater(result["idle_yaw_drift_deg"],20)
        self.assertFalse(result["success"])

    def test_roll_must_finish_facing_its_original_direction(self):
        rows=[]
        for k in range(250):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.zeros(2),up=-1. if k==30 else 1.,
                tilt=180. if k==30 else 0.,yaw=0.,vel=np.zeros(3),gyro=np.array([0,2*np.pi/5,0]),
                contact=np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),head_contact=k==20,
                head_up=-1.,supported=True,lateral_z=0.))
        self.assertTrue(summarize(TASKS["roulade"],rows,np.array([1,0]))["success"])
        rows[-1]["yaw"]=np.pi
        result=summarize(TASKS["roulade"],rows,np.array([1,0]))
        self.assertTrue(result["final_standing"])
        self.assertTrue(result["single_revolution"])
        self.assertFalse(result["success"])

class RealTelemetry(unittest.TestCase):
    def test_deployment_entry_uses_declared_idle_actor_and_keeps_history(self):
        from sim2sim.research.models import NativeAnchor
        source=TASKS['walking'].source;w=World(TASKS['roulade'],entry_source=source)
        try:
            w.reset(841);teacher,cmd=w.prepare_standing_entry()
            self.assertEqual(teacher.sha256,NativeAnchor(source).sha256)
            np.testing.assert_array_equal(cmd,0.)
            w.enter_from_standing(.1)
            np.testing.assert_array_equal(w.obs()[34:48],w.last)
            self.assertGreater(float(np.linalg.norm(w.last)),.01)
            self.assertEqual(w.t,0.)
        finally:w.close()

    def test_episode_generator_resume_matches_next_fresh_reset(self):
        from sim2sim.research.train import Vector
        a=Vector(TASKS['walking'],1,919,random_commands=.5);b=None
        try:
            state=a.checkpoint_state();a.reset_worlds([0])
            b=Vector(TASKS['walking'],1,919,random_commands=.5,environment_state=state)
            self.assertEqual(a.count,b.count);self.assertGreater(b.count,state['count'])
            self.assertEqual(a.worlds[0].condition,b.worlds[0].condition)
            self.assertEqual(a.checkpoint_state(),b.checkpoint_state())
            np.testing.assert_allclose(a.worlds[0].obs(),b.worlds[0].obs(),atol=2e-5)
        finally:
            a.close()
            if b is not None:b.close()

    def test_generic_runner_uses_declared_time_and_heading_inputs(self):
        from sim2sim.research.time_input import prepare,add_heading
        from sim2sim.policy import OnnxPolicy
        from sim2sim.runner import run_rollout
        from sim2sim.backends.mujoco_backend import MujocoBackend
        from sim2sim.paths import load_robot_json
        with tempfile.TemporaryDirectory() as d:
            path=add_heading(prepare(TASKS['roulade'].source,Path(d)/'time.onnx'),Path(d)/'heading.onnx')
            cfg=load_robot_json(TASKS['roulade'].robot_path);backend=MujocoBackend(Path(cfg['mjcf']))
            try:trace=run_rollout(backend,OnnxPolicy(path),cfg,schedule=[dict(name='roll',seconds=.06,vel=[0,0,0])])
            finally:backend.close()
            np.testing.assert_allclose(np.asarray(trace['obs'])[:,48],[0,.004,.008],atol=1e-7)
            np.testing.assert_allclose(np.asarray(trace['obs'])[0,49:51],[0,1],atol=1e-7)

    def test_time_input_restarts_at_trigger_and_retains_curriculum_time(self):
        w=World(TASKS["roulade"],"mujoco",time_input_s=5.)
        try:
            w.reset(73000);self.assertEqual(w.obs()[48],0.)
            w.enter_from_standing();self.assertEqual(w.obs()[48],0.)
            w.step(np.zeros(14));self.assertAlmostEqual(w.obs()[48],.02/5)
            w.reset_from_roll_state(w.mj.data.qpos,w.mj.data.qvel,w.last,w.heading,[0,0,0,0],source_time=1.25)
            self.assertEqual(w.t,0.);self.assertAlmostEqual(w.obs()[48],.25)
            w.reset(73001);self.assertEqual(w.obs()[48],0.)
        finally:w.close()

    def test_timed_kick_clock_and_heading_reset_in_native_world(self):
        for skill in ('kick_left','kick_right'):
            w=World(TASKS[skill],'mujoco',time_input_s=5.,heading_input=True)
            try:
                w.reset(73411)
                np.testing.assert_allclose(w.obs()[48:51],[0,0,1],atol=1e-6)
                w.step(np.zeros(14));self.assertAlmostEqual(w.obs()[48],.004)
                w.enter_from_standing();self.assertEqual(w.obs()[48],0.)
                w.reset(73412)
                np.testing.assert_allclose(w.obs()[48:51],[0,0,1],atol=1e-6)
            finally:w.close()

    def test_body_transfer_velocity_matches_com_jacobian(self):
        import mujoco
        w=World(TASKS["roulade"],"mujoco")
        try:
            w.reset(72000,randomize=False);m,d=w.mj.model,w.mj.data
            d.qvel[:]=np.random.default_rng(72000).normal(0,2,m.nv)
            mujoco.mj_forward(m,d)
            reports=w.mj.body_poses_mujoco()
            for b in reports:
                bid=w.meta[b["name"]];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacBodyCom(m,d,jp,jr,bid)
                np.testing.assert_allclose(b["linvel"],jp@d.qvel,atol=1e-10)
                np.testing.assert_allclose(b["angvel"],jr@d.qvel,atol=1e-10)
            w.state=w.mj._state();f=w.measure(reset=True)
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBodyCom(m,d,jp,jr,w.mj.base_body_id)
            np.testing.assert_allclose(f["vel"],jp@d.qvel,atol=1e-10)
        finally:w.close()

    def test_interactive_tapes_match_across_backends_and_survive_handoff(self):
        worlds=[World(TASKS["walking"],b) for b in ("mujoco","godot")]
        try:
            for w in worlds:w.reset(62001,"random_seq");w.enter_from_standing()
            for _ in range(35):
                np.testing.assert_array_equal(worlds[0].obs()[48:],worlds[1].obs()[48:])
                for w in worlds:w.step(np.zeros(14))
            for w in worlds:
                w.reset(62002,"walk_025")
                np.testing.assert_array_equal(w.obs()[48:],command(w.task,0,"walk_025"))
        finally:
            for w in worlds:w.close()

    def test_source_play_wheel_friction_is_explicit_and_reference_only(self):
        import mujoco
        a=World(TASKS["roller"],"mujoco");b=World(TASKS["roller"],"mujoco",reference_profile="source_play")
        try:
            for j in range(a.mj.model.njnt):
                name=mujoco.mj_id2name(a.mj.model,mujoco.mjtObj.mjOBJ_JOINT,j) or ""
                if name.startswith("passive_"):
                    adr=a.mj.model.jnt_dofadr[j]
                    self.assertEqual(a.mj.model.dof_frictionloss[adr],0.)
                    self.assertEqual(b.mj.model.dof_frictionloss[adr],.003)
            self.assertNotEqual(a.physics,b.physics)
        finally:a.close();b.close()

    def test_roll_start_preserves_physical_pose_and_action_history(self):
        from sim2sim.research.models import NativeAnchor
        source=World(TASKS["roulade"],"mujoco");target=World(TASKS["roulade"])
        try:
            obs=source.reset(40000);policy=NativeAnchor(TASKS["roulade"].source)
            obj=Objective(source)
            for _ in range(45):
                obs=source.step(policy(obs[None])[0]);obj.compute()
            target.reset(7)
            target.reset_from_roll_state(source.mj.data.qpos,source.mj.data.qvel,
                source.last,source.heading,[obj.net,obj.frontier,obj.pivot,obj.inverted])
            np.testing.assert_allclose(target.state.q,source.state.q,atol=2e-5)
            np.testing.assert_allclose(target.state.base_pos,source.state.base_pos,atol=2e-6)
            np.testing.assert_array_equal(target.obs()[34:48],source.last)
            for name,body in source.features["bodies"].items():
                np.testing.assert_allclose(target.features["bodies"][name]["linvel"],body["linvel"],atol=2e-6)
            self.assertEqual(Objective(target).net,obj.net)
            target.step(policy(target.obs()[None])[0])
            self.assertTrue(np.isfinite(target.state.q).all())
            target.reset(8)
            self.assertEqual(Objective(target).net,0.)
            self.assertIsNone(target.roll_start)
        finally:source.close();target.close()

    def test_real_handoff_preserves_history_and_counted_time(self):
        w=World(TASKS["kick_left"])
        try:
            w.reset(23);obs=w.enter_from_standing()
            self.assertEqual(w.t,0.)
            self.assertGreater(np.linalg.norm(w.last),.01)
            np.testing.assert_array_equal(obs[34:48],w.last)
            self.assertGreater(w.features["z"],.09)
            self.assertIsNotNone(w.pending_ball)
            from sim2sim.research.models import NativeAnchor
            w.step(NativeAnchor(TASKS["kick_left"].source)(obs[None])[0])
            self.assertAlmostEqual(w.t,.02)
            self.assertIsNone(w.pending_ball)
            self.assertLess(np.linalg.norm(w.features["ball_pos"][:2]-w.features["xy"]),.2)
        finally:w.close()

    def test_mouth_is_actual_site_and_ball_is_free(self):
        import mujoco
        w=World(TASKS["kick_left"],"mujoco")
        try:
            w.reset(11,randomize=False)
            self.assertEqual(w.mj.model.nu,14)
            self.assertIn("ball",w.features["bodies"])
            np.testing.assert_allclose(w.features["mouth_pos"],w.mj.data.site_xpos[w.site_id],atol=1e-12)
            bid=w.meta["ball"];jid=w.mj.model.body_jntadr[bid]
            self.assertEqual(w.mj.model.jnt_type[jid],mujoco.mjtJoint.mjJNT_FREE)
        finally:w.close()

    def test_godot_reports_distinct_ball_and_ground_contact(self):
        w=World(TASKS["kick_left"])
        try:
            w.reset(11,randomize=False)
            for _ in range(10):w.step(np.zeros(14))
            ball=w.features["bodies"]["ball"]
            self.assertTrue(ball["ground_contact"])
            self.assertTrue(all(c["ground"] for c in ball["contact_events"]))
            self.assertEqual(w.features["kick_contacts"],[])
            saved=w.state.extra["raw"].pop("body_states")
            with self.assertRaises(RuntimeError):w.measure()
            w.state.extra["raw"]["body_states"]=saved
        finally:w.close()

class Deployment(unittest.TestCase):
    def test_fast_command_hinge_is_exact_and_preserves_keyboard_range(self):
        from sim2sim.research.conditioning import adapt,parity as adapter_parity
        from sim2sim.research.models import NativeAnchor
        with tempfile.TemporaryDirectory() as d:
            source=TASKS['walking'].source;hinge=(.3,[2.,0.,6.])
            dest=adapt(source,Path(d)/'speed.onnx',forward_command_hinge=hinge)
            x=np.random.default_rng(991).normal(0,.2,(100,61)).astype(np.float32)
            x[:,48]=np.linspace(-.3,.3,100)
            np.testing.assert_array_equal(NativeAnchor(source)(x),NativeAnchor(dest)(x))
            self.assertTrue(adapter_parity(source,dest,np.eye(3),n=500,forward_command_hinge=hinge)['passed'])

    def test_nested_expert_retime_matches_rebuilding_the_same_blend(self):
        from sim2sim.research.roll_experts import build,retime
        from sim2sim.research.mirror import symmetrize
        from sim2sim.research.models import NativeAnchor
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=build(TASKS['roulade'].source,d/'early.onnx',1.8,2.)
            b=build(TASKS['roulade'].source,d/'late.onnx',2.2,2.4)
            expected=symmetrize(a,d/'expected.onnx');nested=symmetrize(b,d/'nested.onnx')
            actual=retime(nested,d/'actual.onnx',1.8,2.)
            x=np.random.default_rng(773).normal(0,.2,(250,61)).astype(np.float32);x[:,48]=np.linspace(0,1,len(x))
            np.testing.assert_array_equal(NativeAnchor(actual)(x),NativeAnchor(expected)(x))

    def test_fast_forward_heading_adapter_preserves_slower_commands(self):
        from sim2sim.research.conditioning import adapt,parity as adapter_parity
        from sim2sim.research.models import NativeAnchor
        with tempfile.TemporaryDirectory() as d:
            source=TASKS['walking'].source;dest=adapt(source,Path(d)/'hinge.onnx',forward_yaw_hinge=(.3,-3.))
            x=np.random.default_rng(90).normal(0,.2,(100,61)).astype(np.float32);x[:,48]=np.linspace(-.3,.3,100)
            np.testing.assert_array_equal(NativeAnchor(source)(x),NativeAnchor(dest)(x))
            self.assertTrue(adapter_parity(source,dest,np.eye(3),n=500,forward_yaw_hinge=(.3,-3.))['passed'])

    def test_heading_ready_actor_preserves_parent_and_scales_new_inputs(self):
        from sim2sim.research.time_input import prepare,add_heading
        from sim2sim.research.models import NativeAnchor,Critic
        with tempfile.TemporaryDirectory() as d:
            time_path=prepare(TASKS['roulade'].source,Path(d)/'time.onnx')
            path=add_heading(time_path,Path(d)/'heading.onnx')
            x=np.random.default_rng(551).normal(0,.2,(100,61)).astype(np.float32);expected=x.copy();expected[:,49:51]=0
            np.testing.assert_array_equal(NativeAnchor(path)(x),NativeAnchor(time_path)(expected))
            p=Policy(path,'residual',template=TASKS['roulade'].source,time_gate=(.2,.5));p.task_name='roulade'
            self.assertTrue(p.anchor.heading_input)
            np.testing.assert_array_equal(p.delta.denominator[48:51],1.)
            np.testing.assert_array_equal(Critic(TASKS['roulade'].source,EXTRA_DIM,5.,True).denominator[48:51],1.)
            with torch.no_grad():p.delta.net[-1].weight.add_(.001)
            self.assertTrue(parity(p,export_policy(p,Path(d)/'trained.onnx'),n=500)['passed'])

    def test_time_gate_preserves_launch_exactly_and_is_inside_export(self):
        from sim2sim.research.time_input import prepare
        with tempfile.TemporaryDirectory() as d:
            path=prepare(TASKS['roulade'].source,Path(d)/'teacher.onnx')
            p=Policy(path,'plain',template=TASKS['roulade'].source,time_gate=(1.8,2.1));p.task_name='roulade'
            with torch.no_grad():p.delta.net[-1].bias.add_(.1)
            x=np.zeros((1,61),np.float32);x[:,5]=-1.;x[:,48]=.2
            np.testing.assert_array_equal(p.predict(x),p.anchor(x))
            x[:,48]=.8;np.testing.assert_allclose(p.predict(x)-p.anchor(x),.1,atol=1e-6)
            exported=export_policy(p,Path(d)/'gated.onnx')
            self.assertTrue(parity(p,exported,n=500)['passed'])

    def test_time_ready_actor_keeps_teacher_and_exports_nonzero_adaptation(self):
        from sim2sim.research.time_input import prepare
        from sim2sim.research.models import NativeAnchor,Critic
        from sim2sim.policy import OnnxPolicy
        with tempfile.TemporaryDirectory() as d:
            path=prepare(TASKS["roulade"].source,Path(d)/"teacher.onnx")
            x=np.random.default_rng(42).normal(0,.2,(200,61)).astype(np.float32)
            original=x.copy();original[:,48:]=0
            np.testing.assert_array_equal(NativeAnchor(path)(x),NativeAnchor(TASKS["roulade"].source)(original))
            p=Policy(path,"plain",template=TASKS["roulade"].source);p.task_name="roulade"
            self.assertEqual(p.delta.denominator[48],1.)
            self.assertEqual(Critic(TASKS["roulade"].source,EXTRA_DIM,5.).denominator[48],1.)
            with torch.no_grad():
                p.delta.net[-1].weight.add_(torch.randn_like(p.delta.net[-1].weight)*1e-5)
            exported=export_policy(p,Path(d)/"candidate.onnx")
            self.assertTrue(parity(p,exported,n=500)["passed"])
            loaded=OnnxPolicy(exported);loaded.check_dims(14);self.assertEqual(loaded.time_input_s,5.)

    def test_adapted_native_anchor_can_be_adapted_again(self):
        import onnx
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            source=TASKS["walking"].source
            first=Policy(source,"plain");first.task_name="walking"
            intermediate=export_policy(first,Path(td)/"first.onnx")
            second=Policy(intermediate,"residual",template=source);second.task_name="walking"
            with torch.no_grad():second.delta.net[-1].bias.add_(.01)
            final=export_policy(second,Path(td)/"second.onnx")
            props=onnx.load(final).metadata_props
            self.assertEqual(len({p.key for p in props}),len(props))
            self.assertTrue(parity(second,final,n=1000)["passed"])

    def test_command_adapter_is_a_single_onnx_with_exact_parity(self):
        from sim2sim.research.conditioning import adapt,parity as conditioning_parity
        with tempfile.TemporaryDirectory() as td:
            matrix=np.array([[2,0,.5],[0,1,0],[0,0,1.5]],np.float32)
            source=TASKS["walking"].source;dest=adapt(source,Path(td)/"policy.onnx",matrix)
            self.assertTrue(conditioning_parity(source,dest,matrix,n=1000)["passed"])

    def test_nonzero_adaptation_exports_without_relaxed_tolerance(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("plain","anchor","residual"):
                torch.manual_seed(7);p=Policy(TASKS["kick_left"].source,variant)
                with torch.no_grad():
                    for v in p.delta.net.parameters():v.add_(torch.randn_like(v)*1e-5)
                export=export_policy(p,Path(tmp)/(variant+".onnx"))
                result=parity(p,export,n=1000)
                self.assertTrue(result["passed"],(variant,result))

if __name__=="__main__":unittest.main()
