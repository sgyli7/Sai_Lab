"""Reproducible interactive command tapes, separate from fixed test cases."""
import numpy as np

KEYBOARD_TAPES = {
    'forward_long': [('idle', 1., set()), ('forward', 6., {'fwd'}), ('brake', 3., set())],
    'forward': [('idle', 1., set()), ('forward', 2., {'fwd'}), ('brake', 2., set())],
    'turn': [('idle', 1., set()), ('turn', 2., {'left'}), ('brake', 2., set())],
    'mixed': [('idle', 1., set()), ('forward', 2., {'fwd'}),
              ('walk_turn', 2., {'fwd', 'left'}), ('back', 1., {'back'}), ('brake', 2., set())],
}


def keyboard_commands(condition, dt=.02):
    """Actual default keyboard commands for supplementary training episodes."""
    from sim2sim.play_input import PlayBrain
    brain = PlayBrain(has_standing=False)
    commands = []
    for _, seconds, held in KEYBOARD_TAPES[condition]:
        for _ in range(round(seconds / dt)):
            commands.append(brain.tick(held, [], dt).command.copy())
    return np.stack(commands)


def roller_keyboard_commands(condition,dt=.02):
    """The same command transitions used by the standalone flat-ground replay.

    Training retains real physics from reset to push to braking; no fabricated
    moving pose or newly reset hidden action history substitutes for the prefix.
    """
    from sim2sim.standalone.cases import standard_cases
    from sim2sim.play import ROLLER_LIMITS
    from sim2sim.play_input import PlayBrain
    case=standard_cases()[condition]
    if case['skill']!='roller' or 'crouch' in condition:raise ValueError('Roller keyboard curriculum requires one locomotion actor')
    brain=PlayBrain(has_standing=False,has_sitstand=False,has_pick=False,has_kick_left=False,
                    has_kick_right=False,has_roulade=False,lim=ROLLER_LIMITS)
    commands=[]
    for i in range(round(case['seconds']/dt)):
        t=i*dt;segment=next(s for s in reversed(case['segments']) if s.get('at',0)<=t+1e-9)
        held=segment.get('held',[])
        commands.append(brain.tick(set(held),[],dt,press_order=segment.get('order',held)).command.copy())
    return np.stack(commands)


def random_schedule(task,seed):
    rng=np.random.default_rng(int(seed)+92317);t=0.;schedule=[]
    while t<task.seconds:
        kind=int(rng.choice(6,p=[.25,.25,.1,.1,.15,.15]));target=np.zeros(3,np.float32)
        roller=task.name=="roller"
        if kind==1:target[0]=rng.uniform(.08,.6 if roller else .4)
        elif kind==2:target[0]=rng.uniform(-.3,-.08)
        elif kind==3:
            if roller:target[2]=rng.uniform(-1.,1.)
            else:target[1]=rng.uniform(-.2,.2)
        elif kind==4:target[2]=rng.uniform(-1. if roller else -.8,1. if roller else .8)
        elif kind==5:
            target[0]=rng.uniform(.08,.4);target[2]=rng.uniform(-.6,.6)
        schedule.append((t,target));t+=rng.uniform(.5,2.5)
    return schedule


def scheduled_command(schedule,t):
    index=max(i for i,(start,_) in enumerate(schedule) if start<=t+1e-9)
    start,target=schedule[index];previous=np.zeros(3) if index==0 else schedule[index-1][1]
    blend=np.clip((t-start)/.1,0,1);out=np.zeros(13,np.float32)
    out[:3]=previous*(1-blend)+target*blend
    return out
