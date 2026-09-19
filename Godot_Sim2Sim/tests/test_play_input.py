"""PlayBrain / hold-to-move mapping (no Godot, no ONNX)."""

from __future__ import annotations

import unittest

import numpy as np

from sim2sim.play_input import (
    PlayBrain,
    TwistLimits,
    TwistRamp,
    WalkGait,
    clamp_time_scale,
    held_twist,
    keys_to_held,
    keys_to_taps,
    relaunch_argv,
    wall_dt,
)


class TestHeldTwist(unittest.TestCase):
    def test_empty_is_idle(self) -> None:
        self.assertEqual(held_twist(set()), (0.0, 0.0, 0.0))

    def test_fwd_back_and_space(self) -> None:
        lim = TwistLimits()
        self.assertEqual(held_twist({"fwd"}, lim)[0], lim.vmax_x)
        self.assertEqual(held_twist({"back"}, lim)[0], lim.vmin_x)
        self.assertEqual(held_twist({"fwd", "idle"}, lim), (0.0, 0.0, 0.0))

    def test_yaw_and_strafe(self) -> None:
        lim = TwistLimits()
        self.assertEqual(held_twist({"left"}, lim)[2], lim.vmax_ang)
        self.assertEqual(held_twist({"right"}, lim)[2], -lim.vmax_ang)
        self.assertEqual(held_twist({"strafe_l"}, lim)[1], lim.vmax_y)
        self.assertEqual(held_twist({"strafe_r"}, lim)[1], lim.vmin_y)

    def test_arrow_aliases(self) -> None:
        self.assertEqual(keys_to_held({"W", "UP"}), {"fwd"})
        self.assertEqual(keys_to_held({"A", "LEFT"}), {"left"})
        self.assertIn("pick", keys_to_taps({"G", "KEY_1"}))
        self.assertEqual(keys_to_taps({"ESCAPE"}), ["quit"])
        self.assertEqual(keys_to_taps({"KEY_6"}), ["switch_robot"])


class TestTimeScale(unittest.TestCase):
    def test_default_is_realtime(self) -> None:
        self.assertAlmostEqual(wall_dt(0.02, 1.0), 0.02)

    def test_faster_and_slower(self) -> None:
        self.assertAlmostEqual(wall_dt(0.02, 2.0), 0.01)
        self.assertAlmostEqual(wall_dt(0.02, 0.25), 0.08)

    def test_clamp(self) -> None:
        self.assertAlmostEqual(clamp_time_scale(0.01), 0.25)
        self.assertAlmostEqual(clamp_time_scale(9.0), 3.0)
        self.assertAlmostEqual(wall_dt(0.02, 99.0), 0.02 / 3.0)


class TestPlayBrain(unittest.TestCase):
    def setUp(self) -> None:
        self.b = PlayBrain()

    def test_hold_fwd_switches_to_walking(self) -> None:
        # ramped: one tick no longer snaps to full speed; settle first.
        for _ in range(15):
            out = self.b.tick({"fwd"}, [], 0.02)
        self.assertEqual(out.policy, "walking")
        np.testing.assert_allclose(out.command[0:3], [0.3, 0.0, 0.0], atol=1e-6)

    def test_release_returns_standing(self) -> None:
        for _ in range(15):
            self.b.tick({"fwd"}, [], 0.02)
        for _ in range(10):
            out = self.b.tick(set(), [], 0.02)
        self.assertEqual(out.policy, "standing")
        np.testing.assert_allclose(out.command, np.zeros(13))

    def test_sit_toggle_and_blocks_kick(self) -> None:
        out = self.b.tick(set(), ["sit"], 0.02)
        self.assertEqual(out.policy, "sitstand")
        self.assertAlmostEqual(float(out.command[0]), 1.0)
        self.b.tick(set(), ["kick_left"], 0.02)
        self.assertEqual(self.b.policy, "sitstand")
        out = self.b.tick(set(), ["sit"], 0.02)
        self.assertFalse(self.b.sit)
        self.assertAlmostEqual(float(out.command[0]), 0.0)

    def test_rise_keeps_sitstand_policy_until_transition_finishes(self) -> None:
        self.b.tick(set(), ["sit"], .02)
        out = self.b.tick({"fwd"}, ["sit"], .02)
        self.assertEqual(out.policy, "sitstand")
        self.assertEqual(out.status, "rising")
        for _ in range(149):
            out = self.b.tick({"fwd"}, ["kick_left"], .02)
            self.assertEqual(out.policy, "sitstand")
            self.assertEqual(float(out.command[0]), 0.)
        out = self.b.tick({"fwd"}, [], .02)
        self.assertEqual(self.b.rise_t, 0.)
        self.assertNotEqual(out.policy, "sitstand")
        self.b.tick(set(), ["sit"], .02)
        self.b.tick(set(), ["sit"], .02)
        self.b.reset_motion()
        self.assertEqual(self.b.rise_t, 0.)

    def test_pick_cycle_returns(self) -> None:
        self.b.tick(set(), ["pick"], 0.02)
        self.assertEqual(self.b.policy, "ground_pick")
        # 4s period; 4.1s must finish
        for _ in range(205):
            self.b.tick(set(), [], 0.02)
        self.assertEqual(self.b.policy, "standing")

    def test_roulade_timer(self) -> None:
        self.b.tick(set(), ["roulade"], 0.02)
        self.assertEqual(self.b.policy, "roulade")
        np.testing.assert_allclose(self.b.command_13(), np.zeros(13))
        for _ in range(101):
            self.b.tick(set(), [], 0.02)
        self.assertEqual(self.b.policy, "roulade")
        for _ in range(149):
            self.b.tick(set(), [], 0.02)
        self.assertEqual(self.b.policy, "standing")

    def test_phase_skills_start_at_zero_and_run_full_period(self) -> None:
        for b, tap, policy, steps in [
            (PlayBrain(), "pick", "ground_pick", 200),
            (PlayBrain(has_standing=False, has_sitstand=False, has_roller_crouch=True),
             "sit", "roller_crouch", 250),
        ]:
            out = b.tick(set(), [tap], 0.02)
            self.assertEqual(out.started_skill, policy)
            np.testing.assert_allclose(out.command[:2], [1.0, 0.0], atol=1e-7)
            for _ in range(steps - 1):
                out = b.tick(set(), [], 0.02)
                self.assertEqual(out.policy, policy)
                self.assertIsNone(out.started_skill)
                self.assertAlmostEqual(float(np.linalg.norm(out.command[:2])), 1.0, places=6)
            out = b.tick(set(), [], 0.02)
            self.assertEqual(out.policy, "walking" if policy == "roller_crouch" else "standing")
            np.testing.assert_allclose(out.command, np.zeros(13))

    def test_kick_trigger_is_once_and_full_five_seconds(self) -> None:
        for skill in ("kick_left", "kick_right"):
            b = PlayBrain()
            self.assertEqual(b.tick(set(), [skill], 0.02).started_skill, skill)
            for _ in range(249):
                out = b.tick(set(), [skill], 0.02)
                self.assertEqual(out.policy, skill)
                self.assertIsNone(out.started_skill)
            self.assertEqual(b.tick(set(), [], 0.02).policy, "standing")

    def test_ball_trigger_rotates_with_heading(self) -> None:
        from types import SimpleNamespace
        from sim2sim.play import kick_ball_position
        st = SimpleNamespace(base_pos=np.array([2.0, 3.0, 0.12]),
                             base_quat_wxyz=np.array([2**-0.5, 0, 0, 2**-0.5]))
        np.testing.assert_allclose(kick_ball_position(st, "kick_left"), [1.958, 3.09, 0.035])
        np.testing.assert_allclose(kick_ball_position(st, "kick_right"), [2.042, 3.09, 0.035])

    def test_idle_tap_clears_walk(self) -> None:
        for _ in range(15):
            self.b.tick({"fwd"}, [], 0.02)
        out = self.b.tick({"fwd"}, ["idle"], 0.02)  # SPACE pressed while W held
        # idle is newest → target snaps to 0, then ramps down at decel.
        self.assertEqual(out.policy, "standing")
        self.assertLess(float(out.command[0]), 0.3)
        for _ in range(10):
            out = self.b.tick({"fwd"}, ["idle"], 0.02)
        self.assertAlmostEqual(float(out.command[0]), 0.0)
        # SPACE released, W still held -> duck moves again (no deadlock).
        for _ in range(15):
            out = self.b.tick({"fwd"}, [], 0.02)
        self.assertAlmostEqual(float(out.command[0]), 0.3, places=6)

    def test_reset_and_quit(self) -> None:
        self.b.tick({"fwd"}, [], 0.02)
        out = self.b.tick(set(), ["reset"], 0.02)
        self.assertTrue(out.reset)
        self.assertEqual(out.policy, "standing")
        out = self.b.tick(set(), ["quit"], 0.02)
        self.assertTrue(out.quit)

    def test_walking_only_stays_on_walking(self) -> None:
        b = PlayBrain(has_standing=False, has_sitstand=False, has_pick=False)
        out = b.tick(set(), [], 0.02)
        self.assertEqual(out.policy, "walking")

    def test_switch_robot_tap(self) -> None:
        out = self.b.tick(set(), ["switch_robot"], 0.02)
        self.assertTrue(out.switch_robot)
        self.assertEqual(out.policy, "standing")

    def test_roller_limits_block_strafe(self) -> None:
        lim = TwistLimits(vmax_x=0.6, vmin_x=-0.5, vmax_y=0.0, vmin_y=0.0, vmax_ang=1.0)
        b = PlayBrain(has_sitstand=False, has_pick=False, has_kick_left=False, has_kick_right=False, has_roulade=False, lim=lim)
        for _ in range(35):
            out = b.tick({"fwd"}, [], 0.02)
        np.testing.assert_allclose(out.command[0:3], [0.6, 0.0, 0.0], atol=1e-6)
        out = b.tick({"strafe_l"}, [], 0.02)  # roller: strafe target is 0
        self.assertAlmostEqual(float(out.command[1]), 0.0, places=6)
        b.tick(set(), ["kick_left"], 0.02)
        self.assertNotEqual(b.policy, "kick_left")


class TestInputShaping(unittest.TestCase):
    """New behaviour: ramp slew, opposing-key resolution, diagonal norm, hysteresis."""

    def test_ramp_is_monotone_and_capped(self) -> None:
        b = PlayBrain()
        prev = 0.0
        for _ in range(40):
            b.tick({"fwd"}, [], 0.02)
            v = float(b.vel[0])
            self.assertGreaterEqual(v, prev - 1e-6)
            self.assertLessEqual(v, 0.3 + 1e-6)
            prev = v
        self.assertAlmostEqual(prev, 0.3, places=6)

    def test_ramp_reaches_full_speed_quickly(self) -> None:
        b = PlayBrain()
        for i in range(1, 20):
            b.tick({"fwd"}, [], 0.02)
            if float(b.vel[0]) >= 0.3 - 1e-6:
                break
        self.assertLessEqual(i, 5)  # 0→0.3 m/s within ~0.1 s

    def test_first_tick_no_longer_snaps_to_full(self) -> None:
        b = PlayBrain()
        out = b.tick({"fwd"}, [], 0.02)
        self.assertLess(abs(float(out.command[0])), 0.3)

    def test_opposing_newest_wins(self) -> None:
        b = PlayBrain()
        for _ in range(10):
            b.tick({"fwd"}, [], 0.02)
        # S arrives after W → back wins (decelerate/reverse), not freeze at 0
        b.tick({"fwd", "back"}, [], 0.02)
        self.assertLess(float(b.vel[0]), 0.2)

    def test_held_twist_opposing_without_order_is_stable(self) -> None:
        v = held_twist({"fwd", "back"})
        self.assertNotEqual(v[0], 0.0)

    def test_diagonal_length_normalised(self) -> None:
        lim = TwistLimits()
        vx, vy, _ = held_twist({"fwd", "strafe_l"}, lim)
        self.assertLessEqual(np.hypot(vx, vy), np.hypot(lim.vmax_x, lim.vmax_y) + 1e-9)
        self.assertGreater(vx, 0.0)
        self.assertGreater(vy, 0.0)

    def test_local_release_then_repress_updates_newest_order(self) -> None:
        b = PlayBrain()
        for _ in range(4):
            b.tick({"fwd"}, [], 0.02)
        b.tick({"fwd", "back"}, [], 0.02)  # back pressed last
        self.assertEqual(b.press_order[-1], "back")
        b.tick({"back"}, [], 0.02)          # release fwd
        b.tick({"fwd", "back"}, [], 0.02)  # re-press fwd
        self.assertEqual(b.press_order[-1], "fwd")
        self.assertGreater(held_twist(
            {"fwd", "back"}, b.lim, press_order=b.press_order
        )[0], 0.0)

    def test_sitstand_only_initialises_and_resets(self) -> None:
        b = PlayBrain(
            has_walking=False,
            has_standing=False,
            has_sitstand=True,
        )
        self.assertEqual(b.policy, "sitstand")
        b.policy = "ground_pick"
        b.reset_motion()
        self.assertEqual(b.policy, "sitstand")

    def test_yaw_not_speed_bumped_by_diagonal(self) -> None:
        vx, vy, w = held_twist({"fwd", "left"}, TwistLimits())
        self.assertAlmostEqual(w, TwistLimits().vmax_ang)
        self.assertAlmostEqual(vx, TwistLimits().vmax_x)  # yaw not part of norm

    def test_gait_hysteresis_no_chatter(self) -> None:
        g = WalkGait(switch_on=0.10, switch_off=0.03)
        self.assertTrue(g.settled(0.12, 0.0, True))
        self.assertTrue(g.settled(0.05, 0.0, False))   # between off/on: stays walking
        self.assertFalse(g.settled(0.02, 0.0, False))
        self.assertFalse(g.settled(0.08, 0.0, False))  # between: stays standing
        self.assertTrue(g.settled(0.11, 0.0, True))

    def test_gait_yaw_engages_and_holds(self) -> None:
        g = WalkGait(switch_on=0.10, switch_off=0.03)
        self.assertTrue(g.settled(0.0, 1.5, True))    # fresh A press -> walking
        self.assertTrue(g.settled(0.0, 0.15, False))  # held yaw still decaying -> stays walking
        self.assertFalse(g.settled(0.0, 0.02, False)) # yaw decayed below off -> stand
        g2 = WalkGait(switch_on=0.10, switch_off=0.03)
        self.assertTrue(g2.settled(0.2, 1.5, True))   # forward+turn walks
        self.assertTrue(g2.settled(0.05, 1.5, False)) # held yaw keeps walking (no chatter)
        self.assertFalse(g2.settled(0.01, 0.01, False))  # both released -> stand

    def test_pure_yaw_hold_emits_yaw_command(self) -> None:
        b = PlayBrain()
        out = None
        for _ in range(20):
            out = b.tick({"left"}, [], 0.02)
        assert out is not None
        # A fresh yaw press engages walking (standing has no twist axis), so
        # the turn command actually reaches the policy.
        self.assertEqual(b.policy, "walking")
        np.testing.assert_allclose(out.command[0:3], [0.0, 0.0, 1.5], atol=1e-6)

    def test_idle_held_does_not_deadlock(self) -> None:
        b = PlayBrain()
        b.tick(set(), ["idle"], 0.02)        # SPACE tap -> idle stops the duck
        b.tick(set(), [], 0.02)              # SPACE released, standing
        out = None
        for _ in range(15):                  # walk normally
            out = b.tick({"fwd"}, [], 0.02)
        assert out is not None
        self.assertAlmostEqual(float(out.command[0]), 0.3, places=6)
        for _ in range(10):                  # SPACE pressed last -> stop
            out = b.tick({"fwd", "idle"}, [], 0.02)
        self.assertAlmostEqual(float(out.command[0]), 0.0)
        # SPACE never released, W re-pressed -> newest-wins is W, duck moves.
        for _ in range(15):
            out = b.tick({"fwd"}, [], 0.02)
        assert out is not None
        self.assertAlmostEqual(float(out.command[0]), 0.3, places=6)

    def test_external_press_order_newest_wins(self) -> None:
        # Godot echoes held_order; last entry = newest press wins.
        b = PlayBrain()
        for _ in range(10):
            b.tick({"fwd"}, [], 0.02, press_order=["fwd"])
        v0 = float(b.vel[0])
        self.assertGreater(v0, 0.25)
        # S pressed after W: order says back is newest -> decel/reverse.
        b.tick({"fwd", "back"}, [], 0.02, press_order=["fwd", "back"])
        self.assertLess(float(b.vel[0]), v0)
        # W re-pressed last: forward wins again.
        for _ in range(10):
            b.tick({"fwd", "back"}, [], 0.02, press_order=["back", "fwd"])
        self.assertGreater(float(b.vel[0]), 0.1)

    def test_external_press_order_idle_no_deadlock(self) -> None:
        # Exact real-loop trace. `held` is the RAW Godot echo (held_twist
        # newest-wins resolves idle downstream), held_order newest-last.
        b = PlayBrain()
        # SPACE pressed while walking: idle newest -> decel ramp to stop.
        for _ in range(15):
            b.tick({"fwd"}, [], 0.02, press_order=["fwd"])
        for i in range(4):
            out = b.tick({"fwd", "idle"}, [], 0.02, press_order=["fwd", "idle"])
            self.assertLess(float(out.command[0]), 0.3)  # ramping down
        for _ in range(10):
            out = b.tick({"fwd", "idle"}, [], 0.02, press_order=["fwd", "idle"])
        self.assertAlmostEqual(float(out.command[0]), 0.0)
        # W re-pressed (order says newest). Godot's next raw echo is
        # {"fwd"} again (the brain's stop consumed idle), order ["idle","fwd"].
        for _ in range(15):
            out = b.tick({"fwd"}, [], 0.02, press_order=["idle", "fwd"])
        self.assertAlmostEqual(float(out.command[0]), 0.3, places=6)
        # SPACE released: plain forward echo, still walking.
        for _ in range(5):
            out = b.tick({"fwd"}, [], 0.02, press_order=["fwd"])
        self.assertAlmostEqual(float(out.command[0]), 0.3, places=6)

    def test_brain_hysteresis_roundtrip(self) -> None:
        b = PlayBrain()
        b.tick({"fwd"}, [], 0.02)
        # ramp is below switch_on (0.10), but yaw alone counts toward the
        # walk/stand norm — pure yaw should not start walking.
        for _ in range(15):
            b.tick({"fwd"}, [], 0.02)
        self.assertEqual(b.policy, "walking")


class TestRelaunchArgv(unittest.TestCase):
    def test_toggle_roller_keeps_local_ppo(self) -> None:
        walk = ["/bin/sim2sim-play", "--local-ppo"]
        roller = relaunch_argv(walk, want_roller=True, executable="/py")
        self.assertEqual(roller, ["/py", "-u", "/bin/sim2sim-play", "--local-ppo", "--roller"])
        child = ["/bin/sim2sim-play", "--local-ppo", "--roller"]
        back = relaunch_argv(child, want_roller=False, executable="/py")
        self.assertEqual(back, ["/py", "-u", "/bin/sim2sim-play", "--local-ppo"])


if __name__ == "__main__":
    unittest.main()
