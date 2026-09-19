"""Explicit 68D roller contract: original 61D plus seven causal task states.

The extra state is computed before each 50 Hz action, once per simulation stamp.
It contains brake elapsed time, consecutive low-speed duration, 0.2 s mean
speed, mean forward/lateral velocity, and left/right wheel-ground contact.
No future targets, score labels, or physics modifications enter inference.
"""
from collections import deque
from pathlib import Path
import hashlib

import numpy as np

TASK_STATE_KEY = 'sim2sim_roller_task_input'
BRAKE_TASK_V1 = 'brake_markov_68_v1'
TASK_DIM = 7


def task_input(metadata):
    mode = metadata.get(TASK_STATE_KEY, '')
    if mode not in ('', BRAKE_TASK_V1):
        raise ValueError('Unknown roller task input: '+str(mode))
    return mode


class BrakeTaskState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stamp = None
        self.brake_start = None
        self.velocities = deque(maxlen=10)
        self.low_steps = 0
        self.values = np.zeros(TASK_DIM, np.float32)

    def observe(self, obs, contacts, stamp):
        x = np.asarray(obs, np.float32)
        contact = np.asarray(contacts, np.float32)
        if x.shape != (61,) or contact.shape != (2,):
            raise ValueError('Task state needs a 61D base observation and two contacts')
        if not np.isfinite(x).all() or not np.isfinite(contact).all() or not np.isfinite(stamp):
            raise ValueError('Nonfinite task observation')
        if self.stamp is not None and stamp < self.stamp-1e-9:
            raise ValueError('Reset task state before resetting simulation time')
        if self.stamp is None or stamp > self.stamp+1e-9:
            braking = x[48] < -.01
            if not braking:
                self.brake_start = None
                self.velocities.clear()
                self.low_steps = 0
            elif self.brake_start is None:
                self.brake_start = float(stamp)
                self.velocities.clear()
                self.low_steps = 0
            velocity = x[58:60].astype(np.float64)
            self.velocities.append(velocity)
            samples = np.array(self.velocities)
            mean_speed = float(np.linalg.norm(samples, axis=1).mean())
            mean_velocity = samples.mean(axis=0)
            if braking and len(samples)==10 and mean_speed < .05:
                self.low_steps += 1
            else:
                self.low_steps = 0
            self.values = np.array([
                0. if self.brake_start is None else min(6., stamp-self.brake_start),
                min(2., self.low_steps*.02), mean_speed, *mean_velocity,
                *np.clip(contact, 0., 1.)], np.float32)
            self.stamp = float(stamp)
        return np.concatenate([x, self.values])


def contacts_from_raw(raw, support_groups):
    bodies = {b['name']: b for b in raw.get('body_states', raw.get('dump', []))}
    if len(support_groups)!=2 or not all(support_groups):
        raise ValueError('Two declared wheel support groups are required')
    missing = set(sum(support_groups, []))-set(bodies)
    if missing:
        raise ValueError('Missing wheel contact telemetry: '+', '.join(sorted(missing)))
    return np.array([any(bodies[n].get('ground_contact', False) for n in group)
                     for group in support_groups], np.float32)


def wrap_anchor(source, destination):
    """Expose a new input dimension while leaving the old ONNX graph intact."""
    import onnx
    from onnx import helper, numpy_helper
    from sim2sim.policy_state import BRAKE_STATE_V1, state_input
    model = onnx.load(source)
    metadata = {p.key:p.value for p in model.metadata_props}
    if task_input(metadata) or state_input(metadata)!=BRAKE_STATE_V1:
        raise ValueError('Task wrapper requires an existing 61D velocity-state roller anchor')
    tensor = model.graph.input[0]
    if [d.dim_value for d in tensor.type.tensor_type.shape.dim] != [1,61]:
        raise ValueError('Expected a fixed 61D anchor')
    prefix='roller_task/'
    names={n for node in model.graph.node for n in [*node.input,*node.output]}
    if any(n.startswith(prefix) for n in names):
        raise ValueError('Task wrapper name collision')
    old=tensor.name; tensor.name=prefix+'obs'
    tensor.type.tensor_type.shape.dim[1].dim_value=61+TASK_DIM
    indices=prefix+'indices'
    model.graph.initializer.append(numpy_helper.from_array(np.arange(61,dtype=np.int64),indices))
    model.graph.node.insert(0,helper.make_node('Gather',[tensor.name,indices],[old],axis=1))
    metadata[TASK_STATE_KEY]=BRAKE_TASK_V1
    metadata['sim2sim_task_anchor_sha256']=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    helper.set_model_props(model,metadata)
    onnx.checker.check_model(model)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,destination)
    return destination
