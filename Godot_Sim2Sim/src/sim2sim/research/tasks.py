"""Shared task contracts. Evaluation tolerances live separately in evaluate.py."""
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

from sim2sim.paths import sim2sim_root
from sim2sim.train.rewards import sit_target_q

SESSION = Path(os.environ.get("SIM2SIM_RESEARCH_DIR") or
               sim2sim_root() / "results/research_20260910").expanduser().resolve()
BASELINE = SESSION / "baseline"
DT = 0.02

@dataclass(frozen=True)
class Task:
    name: str
    factory: str
    previous: str
    mode: str
    seconds: float
    robot: str = "microduck"
    period: float = 4.0
    foot: int = 0

    @property
    def robot_path(self):
        return sim2sim_root() / "robots" / (self.robot + ".json")

    @property
    def source(self):
        return BASELINE / self.factory

TASKS = {t.name: t for t in [
    Task("standing", "alpha_stand.onnx", "Stand_Godot.onnx", "zeros", 10),
    Task("walking", "alpha_walking.onnx", "Walk_Godot.onnx", "twist", 10),
    Task("sitstand", "alpha_sitstand.onnx", "Sitstand_Godot.onnx", "sit", 12),
    Task("ground_pick", "alpha_ground_pick.onnx", "GroundPick_Godot.onnx", "phase", 4),
    Task("kick_left", "ball_kick_left.onnx", "KickLeft_Godot.onnx", "zeros", 5, "microduck_ball", foot=0),
    Task("kick_right", "ball_kick_right.onnx", "KickRight_Godot.onnx", "zeros", 5, "microduck_ball", foot=1),
    Task("roulade", "roulade.onnx", "Roulade_Godot.onnx", "zeros", 5),
    Task("roller", "roller.onnx", "Roller_Godot.onnx", "twist", 10, "microduck_roller"),
    Task("roller_crouch", "roller_crouch.onnx", "RollerCrouch_Godot.onnx", "phase", 5, "microduck_roller", period=5),
]}

# Source: microduck_roller_crouch_env_cfg.py, pinned source commit in session.
CROUCH_STAND = np.array([-.0476,-.0629,-.2869,.9618,1.1674,.6029,.543,-.069,-.0414,-.0337,-.0061,.1534,-.9725,-1.0646],np.float32)
CROUCH_DOWN = np.array([-.0184,.0307,1.4082,1.5248,-.0675,1.0937,1.2149,-.0184,-.0368,.0184,-.0169,-1.4757,-1.5907,.0568],np.float32)

def command(task, t, condition="default"):
    """A command schedule used in both rollout evaluation and training episodes."""
    out = np.zeros(13,np.float32)
    if task.mode == "phase":
        out[:2] = (np.cos(2*np.pi*t/task.period), np.sin(2*np.pi*t/task.period))
    elif task.mode == "sit":
        if condition == "sit_hold": flag = 1
        elif condition == "stand_hold": flag = 0
        elif condition == "rise": flag = 0 if t < 6 else 1
        else: flag = 1 if t < 6 else 0
        out[0] = flag
    elif task.mode == "twist":
        table = {"idle":(0,0,0),"walk_015":(.15,0,0),"walk_025":(.25,0,0),
                 "run_040":(.4,0,0),"back_020":(-.2,0,0),"strafe_l":(0,.2,0),
                 "strafe_r":(0,-.2,0),"turn_l":(0,0,.8),"turn_r":(0,0,-.8),
                 "walk_turn":(.2,0,.5),"walk_push":(.25,0,0),"glide":(.5,0,0)}
        if condition in ("game_seq","default"):
            table_value = [(0,0,0),(.25,0,0),(.2,0,.6),(0,0,0)][min(3,int(t/2.5))]
        else: table_value = table[condition]
        out[:3] = table_value
        if task.robot == "microduck_roller": out[1] = 0
    return out

def conditions(task):
    if task.name == "walking":
        return ["idle","walk_015","walk_025","run_040","back_020","strafe_l","strafe_r","turn_l","turn_r","walk_turn","walk_push","game_seq"]
    if task.name == "roller": return ["idle","glide","back_020","turn_l","turn_r","game_seq"]
    if task.name == "sitstand": return ["default","rise","sit_hold","stand_hold"]
    return ["default"]

def crouch_blend(phase):
    if phase < .1: return phase/.1
    if phase < .5: return 1.
    if phase < .6: return (.6-phase)/.1
    return 0.
