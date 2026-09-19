"""Keyboard / HUD mapping matching microduck_rl infer_policy hold-to-move.

Hold W/↑ forward, S/↓ back, A/← yaw left, D/→ yaw right, Q/E strafe.
Release = idle. One-shot taps: pick / sit / kick / roll / reset / quit / push.

Locomotion shaping (see docs/research_3c_camera.md):
  * opposing keys resolve newest-wins instead of cancelling to zero,
  * diagonal inputs are length-normalised (no corner over-speed),
  * every axis is rate-limited by an accel/decel ramp,
  * stand↔walk switches use hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# Godot keycode names (Input.is_physical_key_pressed) → hold bits
KEY_TO_HOLD: dict[str, str] = {
    "W": "fwd",
    "UP": "fwd",
    "S": "back",
    "DOWN": "back",
    "A": "left",
    "LEFT": "left",
    "D": "right",
    "RIGHT": "right",
    "Q": "strafe_l",
    "E": "strafe_r",
    "SPACE": "idle",
    "SHIFT_LEFT": "sprint",
}

# One-shot taps (keyboard or HUD). Multiple aliases collapse to one action.
KEY_TO_TAP: dict[str, str] = {
    "G": "pick",
    "KEY_1": "pick",
    "Y": "sit",
    "KEY_2": "sit",
    "K": "kick_left",
    "KEY_3": "kick_left",
    "L": "kick_right",
    "KEY_4": "kick_right",
    "R": "roulade",
    "KEY_5": "roulade",
    "KEY_6": "switch_robot",
    "KEY_7": "stand",
    "KEY_0": "reset",
    "BACKSPACE": "reset",
    "P": "push",
    "ESCAPE": "quit",
}

SKILL_TAPS = ("pick", "sit", "kick_left", "kick_right", "roulade", "stand")
LOCO_HOLDS = ("fwd", "back", "left", "right", "strafe_l", "strafe_r", "idle")

TIME_SCALE_MIN = 0.25
TIME_SCALE_MAX = 3.0
TIME_SCALE_DEFAULT = 1.0

# Opposing hold pairs: simultaneous press resolves newest-wins (B), not sum-to-zero.
OPPOSING_PAIRS: tuple[tuple[str, str], ...] = (
    ("fwd", "back"),
    ("strafe_l", "strafe_r"),
    ("left", "right"),
)


def clamp_time_scale(value: float) -> float:
    return min(TIME_SCALE_MAX, max(TIME_SCALE_MIN, float(value)))


def wall_dt(sim_dt: float, time_scale: float) -> float:
    """Wall-clock seconds to wait for this control tick. Physics dt is unchanged."""
    return float(sim_dt) / clamp_time_scale(time_scale)


@dataclass(frozen=True)
class TwistLimits:
    vmax_x: float = 0.3
    vmin_x: float = -0.3
    vmax_y: float = 0.2
    vmin_y: float = -0.2
    vmax_ang: float = 1.5
    switch_threshold: float = 0.05
    # stand↔walk hysteresis: start walking above on, stop below off.
    switch_on: float = 0.10
    switch_off: float = 0.03
    # ramp slew rates in command-units per second of wall sim time.
    accel: float = 12.0
    decel: float = 20.0
    sprint_vmax_x: float = 0.5
    sprint_vmax_ang: float = 0.8
    sprint_yaw_reversal_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sprint_yaw_reversal_s) or self.sprint_yaw_reversal_s < 0:
            raise ValueError('Sprint yaw reversal duration must be finite and nonnegative')


def keys_to_held(keys: set[str]) -> set[str]:
    held: set[str] = set()
    for k in keys:
        bit = KEY_TO_HOLD.get(k.upper() if len(k) == 1 else k)
        if bit:
            held.add(bit)
    return held


def relaunch_argv(argv: list[str], *, want_roller: bool, executable: str) -> list[str]:
    """Rebuild process argv to toggle walk ⇄ roller. Same trick as infer_policy os.execve."""
    rest = [a for a in argv[1:] if a != "--roller"]
    out = [executable, "-u", argv[0], *rest]
    if want_roller:
        out.append("--roller")
    return out


def keys_to_taps(keys: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        ku = k.upper() if len(k) == 1 else k
        action = KEY_TO_TAP.get(ku)
        if action and action not in seen:
            seen.add(action)
            out.append(action)
    return out


def resolve_held(held: set[str], press_order: list[str] | None = None) -> set[str]:
    """Collapse opposing pairs to newest-wins so W+S never sums to a zero
    command that confuses the standing/walking switch. Without press_order
    metadata the deterministic fallback keeps the alphabetically later bit
    of the pair, which is stable but not input-faithful; pass press_order
    from the key-capture layer when available."""
    out = set(held)
    if "idle" in out:
        return {"idle"}
    for a, b in OPPOSING_PAIRS:
        if a in out and b in out:
            keep = a
            if press_order:
                ia = press_order.index(a) if a in press_order else -1
                ib = press_order.index(b) if b in press_order else -1
                keep = a if ia >= ib else b
            else:
                keep = max(a, b)
            out.discard(a if keep == b else b)
    return out


def last_pressed_bit(held: set[str], press_order: list[str] | None) -> str | None:
    """Most recently pressed bit among `held` (press_order is oldest-first)."""
    if press_order:
        for bit in reversed(press_order):
            if bit in held:
                return bit
    return None


def diagonal_cap(lim: TwistLimits, vx: float, vy: float) -> float:
    """Length the (vx, vy) command vector is capped to.

    The corner budget is a circle of radius hypot(axis caps), NOT the
    larger single-axis limit: a pure single-axis press must still reach
    its full trained speed (hypot >= each axis cap), while a diagonal
    press only shrinks the vector (hypot < raw diagonal norm).
    """
    cap_x = lim.vmax_x if vx >= 0 else -lim.vmin_x
    cap_y = lim.vmax_y if vy >= 0 else -lim.vmin_y
    return float(np.hypot(cap_x, cap_y))


def held_twist(
    held: set[str],
    lim: TwistLimits | None = None,
    *,
    press_order: list[str] | None = None,
) -> tuple[float, float, float]:
    """Map held directions to a target twist. Empty set = idle.

    Diagonal (vx, vy) is length-normalised to hypot(axis caps) so corners
    don't command 1.4× the trained speed while single-axis presses stay
    at full speed.
    """
    lim = lim or TwistLimits()
    if "idle" in held:
        return 0.0, 0.0, 0.0
    h = resolve_held(held, press_order)
    vx = (lim.vmax_x if "fwd" in h else 0.0) + (lim.vmin_x if "back" in h else 0.0)
    vy = (lim.vmax_y if "strafe_l" in h else 0.0) + (lim.vmin_y if "strafe_r" in h else 0.0)
    n = float(np.hypot(vx, vy))
    caps = diagonal_cap(lim, vx, vy)
    if n > caps > 0.0:
        s = caps / n
        vx *= s
        vy *= s
    yaw = (lim.vmax_ang if "left" in h else 0.0) + (-lim.vmax_ang if "right" in h else 0.0)
    return float(vx), float(vy), float(yaw)


@dataclass
class TwistRamp:
    """One-sided accel/decel slew limiter per twist axis.

    rate-limit toward `target`: |dv| <= accel*dt while moving away from 0,
    <= decel*dt toward 0 or across zero. Keeps commands inside the policy's
    training envelope while removing the instant 0→full snap.
    """

    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    accel: float = 12.0
    decel: float = 20.0
    yaw_reversing: bool = field(default=False, init=False)

    def step(self, target: tuple[float, float, float], dt: float, *,
             sprint: bool = False, turn_limit: float = 0.8,
             reversal_seconds: float = 0.0) -> np.ndarray:
        tgt = np.asarray(target, dtype=np.float32)
        out = self.vel.copy()
        for i in range(3):
            cur = float(out[i])
            t = float(tgt[i])
            away = (cur == 0.0 and t != 0.0) or (
                np.sign(t) == np.sign(cur) and abs(t) > abs(cur)
            )
            rate = self.accel if away else self.decel
            max_d = rate * dt
            if i == 2:
                if not sprint or reversal_seconds <= 0 or abs(t) <= .05:
                    self.yaw_reversing = False
                elif cur * t < 0 and abs(cur) > .05:
                    self.yaw_reversing = True
                # Carry the reversal through zero rather than restarting acceleration there.
                if self.yaw_reversing:
                    max_d = 2. * turn_limit * dt / reversal_seconds
            d = t - cur
            if abs(d) <= max_d:
                out[i] = t
            else:
                out[i] = cur + np.sign(d) * max_d
            if i == 2 and abs(float(out[i]) - t) < 1e-7:
                self.yaw_reversing = False
        self.vel = out.astype(np.float32)
        return self.vel

    def reset(self) -> None:
        self.vel[:] = 0.0
        self.yaw_reversing = False


@dataclass
class WalkGait:
    """Hysteresis for the standing↔walking policy switch (no chatter at the
    old single 0.05 threshold when the ramp glides through it)."""

    walking: bool = False
    switch_on: float = 0.10
    switch_off: float = 0.03

    def settled(self, lin_mag: float, yaw_mag: float, yaw_new: bool) -> bool:
        """Hysteresis runs on planar speed; the yaw axis participates with a
        softer rule so in-place turns still emit a yaw command: a *fresh*
        yaw press engages walking, a releasing yaw never disengages by
        itself (a held yaw keeps walking), full release (lin and yaw both
        below switch_off) returns to standing."""
        if self.walking:
            if lin_mag <= self.switch_off and yaw_mag <= self.switch_off:
                self.walking = False
        elif lin_mag >= self.switch_on or (yaw_new and yaw_mag >= self.switch_off):
            self.walking = True
        return self.walking

    def reset(self) -> None:
        self.walking = False


@dataclass
class BrainOut:
    policy: str
    command: np.ndarray
    reset: bool = False
    quit: bool = False
    push: bool = False
    switch_robot: bool = False
    status: str = ""
    started_skill: str | None = None
    sprint: bool = False


@dataclass
class PlayBrain:
    """Policy / command state machine. No ONNX, no Godot — unit-testable."""

    has_walking: bool = True
    has_standing: bool = True
    has_sitstand: bool = True
    has_pick: bool = True
    has_kick_left: bool = True
    has_kick_right: bool = True
    has_roulade: bool = True
    has_roller_crouch: bool = False
    # Explicit standing access is independent of the walk model's idle partner.
    has_stand_hold: bool = False
    has_sprint: bool = False
    lim: TwistLimits = field(default_factory=TwistLimits)
    pick_period: float = 4.0
    kick_duration: float = 5.0
    roulade_duration: float = 5.0
    crouch_period: float = 5.0
    rise_duration: float = 3.0

    policy: str = "standing"
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    sit: bool = False
    stand_hold: bool = False
    sprinting: bool = False
    pick_phase: float = 0.0
    behavior_t: float = 0.0
    rise_t: float = 0.0
    ramp: TwistRamp = field(init=False)
    gait: WalkGait = field(init=False)
    press_order: list[str] = field(default_factory=list)
    _prev_held: set[str] = field(default_factory=set, init=False, repr=False)
    _ext_order_active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.ramp = TwistRamp(accel=self.lim.accel, decel=self.lim.decel)
        self.gait = WalkGait(switch_on=self.lim.switch_on, switch_off=self.lim.switch_off)
        self._init_policy()

    def _init_policy(self) -> None:
        # Default idle stance when the bank offers it; an isolated sitstand
        # bank (no walking/standing) initialises and resets to sitstand.
        if self.has_standing:
            self.policy = "standing"
        elif self.has_sitstand:
            self.policy = "sitstand"
        elif self.has_walking:
            self.policy = "walking"

    def _busy(self) -> bool:
        return self.rise_t > 0 or self.policy in ("ground_pick", "roller_crouch", "kick_left", "kick_right", "roulade")

    def reset_motion(self) -> None:
        self.vel[:] = 0.0
        self.sit = False
        self.stand_hold = False
        self.sprinting = False
        self.pick_phase = 0.0
        self.behavior_t = 0.0
        self.rise_t = 0.0
        self.ramp.reset()
        self.gait.reset()
        self.press_order.clear()
        self._prev_held.clear()
        self._init_policy()

    def update_press_order(self, held_now: set[str]) -> set[str]:
        """Track press edges and releases, returning newly pressed bits."""
        pressed_now = held_now - self._prev_held
        # A set has no chronology for truly simultaneous local events; sort
        # so the fallback is deterministic. Godot supplies exact order.
        for bit in sorted(pressed_now):
            if bit in self.press_order:
                self.press_order.remove(bit)
            self.press_order.append(bit)
        self._prev_held = set(held_now)
        self.press_order = [b for b in self.press_order if b in held_now]
        return pressed_now

    def _set_loco(self, held: set[str], dt: float) -> None:
        requested_sprint = 'sprint' in held
        held = held - {'sprint'}
        self.press_order = [bit for bit in self.press_order if bit != 'sprint']
        self.sprinting = False
        if self._busy() or self.sit:
            return
        if self.stand_hold:
            if not (held - {"idle"}):
                self.policy = "standing"
                return
            self.stand_hold = False
        if self._ext_order_active:
            # Godot-side sampler owns press order + edges (play.py echoes
            # held_order every tick): newest press == last entry of the
            # incoming order.
            new_bits = held - self._prev_held
            self._prev_held = set(held)
        else:
            new_bits = self.update_press_order(held)
        # SPACE/idle only stops when it is the NEWEST press; movement keys
        # re-pressed after SPACE keep walking (a held SPACE no longer
        # deadlocks the duck at zero forever).
        if "idle" in held and last_pressed_bit(held, self.press_order) == "idle":
            held = {"idle"}
        self.sprinting = self.has_sprint and requested_sprint and 'fwd' in resolve_held(held, self.press_order)
        limits = replace(self.lim, vmax_x=self.lim.sprint_vmax_x,
                         vmax_ang=self.lim.sprint_vmax_ang) if self.sprinting else self.lim
        target = held_twist(held, limits, press_order=self.press_order)
        # Local-mode idle stop (see the newest-wins block below): skip the
        # ramp step so a stale alphabetical fwd/back resolve can't re-drive
        # the duck after a tap stop.
        skip_ramp = False
        if not self._ext_order_active and self.press_order:
            # Local mode has no tap timestamps: an idle tap that just fired
            # (policy switch to busy/sit) is by construction the newest
            # press, even if it lost the alphabetical fallback below.
            last_bit = self.press_order[-1]
            if last_bit != "idle" and last_bit in held and ("idle" in held or self.sit or self._busy()):
                self.press_order = [b for b in self.press_order if b != "idle"] + ["idle"]
                skip_ramp = True
        if not skip_ramp:
            self.vel[:] = self.ramp.step(
                target, dt, sprint=self.sprinting,
                turn_limit=self.lim.sprint_vmax_ang,
                reversal_seconds=self.lim.sprint_yaw_reversal_s,
            )
        if self.has_walking and self.has_standing:
            self.policy = "walking" if self.gait.settled(
                float(np.hypot(self.vel[0], self.vel[1])),
                float(abs(self.vel[2])),
                bool({"left", "right"} & new_bits),
            ) else "standing"
        elif self.has_walking:
            self.policy = "walking"
        elif self.has_standing:
            self.policy = "standing"

    def _tap(self, action: str) -> None:
        if action == "stand":
            if not self.has_stand_hold or self._busy() or self.sit:
                return
            self.reset_motion()
            self.stand_hold = True
            self.policy = "standing"
            return
        if action in SKILL_TAPS and not self._busy():
            self.stand_hold = False
        if action == "sit":
            if self.has_roller_crouch and not self._busy():
                self.policy = "roller_crouch"
                self.pick_phase = 0.0
                self.vel[:] = 0.0
                self.ramp.reset()
                return
            if not self.has_sitstand:
                return
            if self._busy():
                return
            self.sit = not self.sit
            self.rise_t = 0.0 if self.sit else self.rise_duration
            self.vel[:] = 0.0
            self.ramp.reset()
            self.gait.reset()
            self.policy = "sitstand"
            return
        if action == "pick":
            if not self.has_pick or self._busy() or self.sit:
                return
            self.policy = "ground_pick"
            self.pick_phase = 0.0
            self.vel[:] = 0.0
            self.ramp.reset()
            return
        if action in ("kick_left", "kick_right", "roulade"):
            has = {
                "kick_left": self.has_kick_left,
                "kick_right": self.has_kick_right,
                "roulade": self.has_roulade,
            }[action]
            if not has or self._busy() or self.sit:
                return
            self.policy = action
            self.behavior_t = self.roulade_duration if action == "roulade" else self.kick_duration
            self.vel[:] = 0.0
            self.ramp.reset()

    def _advance(self, dt: float) -> None:
        if self.rise_t > 0:
            self.rise_t = max(0.0, self.rise_t - dt)
            if self.rise_t < 1e-9:
                self.rise_t = 0.0
            return
        if self.policy in ("ground_pick", "roller_crouch"):
            period = self.crouch_period if self.policy == "roller_crouch" else self.pick_period
            self.pick_phase += dt / period
            if self.pick_phase >= 1.0 - 1e-9:
                self.pick_phase = 0.0
                self.reset_motion()
            return
        if self.policy in ("kick_left", "kick_right", "roulade"):
            self.behavior_t -= dt
            if self.behavior_t <= 1e-9:
                self.reset_motion()

    def command_13(self) -> np.ndarray:
        cmd = np.zeros(13, dtype=np.float32)
        if self.policy in ("kick_left", "kick_right", "roulade"):
            return cmd
        if self.policy in ("ground_pick", "roller_crouch"):
            cmd[0] = np.cos(2 * np.pi * self.pick_phase)
            cmd[1] = np.sin(2 * np.pi * self.pick_phase)
            return cmd
        if self.policy == "sitstand":
            cmd[0] = 1.0 if self.sit else 0.0
            return cmd
        if self.policy == "walking":
            # Walking is the only policy with a twist command: held yaw
            # (in-place turn) engages walking via WalkGait even at zero
            # planar speed, so turning always reaches the policy.
            cmd[0:3] = self.vel
        return cmd

    def tick(
        self,
        held: set[str],
        taps: list[str],
        dt: float,
        press_order: list[str] | None = None,
    ) -> BrainOut:
        reset = "reset" in taps
        quit_ = "quit" in taps
        push = "push" in taps
        switch_robot = "switch_robot" in taps
        held_now = set(held)
        if "idle" in taps:
            held_now.add("idle")  # idle tap = held SPACE for resolve/stop
        self._ext_order_active = press_order is not None
        order_new: list[str] | None = None
        if press_order is not None:
            # Fresh order from the input layer (Godot _held_press_order):
            # its last entry is the newest press, so edge info comes with it.
            # Idle taps only stop when the *external* order still lists idle
            # as newest; local taps must not fabricate order chronology.
            order_new = [b for b in press_order if b in held]
            self.press_order = order_new
            if "idle" in taps and "idle" not in press_order:
                order_new = None
        elif "idle" not in taps:
            order_new = press_order
        if order_new is not None:
            last = last_pressed_bit(held_now, order_new)
            if last is not None and last in self.press_order:
                self.press_order.remove(last)
                self.press_order.append(last)
        started_skill = None
        if reset:
            self.reset_motion()
        else:
            # Advance the command that ran in the previous control interval.
            # A newly triggered phase policy must receive phase zero first.
            self._advance(dt)
            previous_policy = self.policy
            for action in taps:
                if action in SKILL_TAPS:
                    self._tap(action)
            if self._busy() and self.policy != previous_policy:
                started_skill = self.policy
            prev_busy, prev_sit = self._busy(), self.sit
            self._set_loco(held_now, dt)
            if not (prev_busy or prev_sit) and self._busy():
                # A tap fired *after* this tick (e.g. the SPACE idle stop):
                # the locomotion pass already consumed the order edge and
                # may have re-resolved a stale held target. Freeze the ramp
                # so the stop is honoured next tick, newest-wins.
                self.ramp.vel[:] = self.vel
        status = self.policy
        if self.policy == "sitstand":
            status = "sit" if self.sit else ("rising" if self.rise_t > 0 else "sitstand-stand")
        elif self.policy == "walking":
            status = f"{'sprint' if self.sprinting else 'walk'} vx={self.vel[0]:+.2f} vy={self.vel[1]:+.2f} w={self.vel[2]:+.2f}"
        return BrainOut(
            policy=self.policy,
            command=self.command_13(),
            reset=reset,
            quit=quit_,
            push=push,
            switch_robot=switch_robot,
            status=status,
            started_skill=started_skill,
            sprint=self.sprinting,
        )
