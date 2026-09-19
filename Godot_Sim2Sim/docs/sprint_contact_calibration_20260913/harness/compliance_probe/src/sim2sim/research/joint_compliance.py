"""Experimental translation compliance in the GPU training asset only.

Jolt's constrained rigid bodies need not lie exactly on the ideal articulated
manifold. Three passive, equality-constrained translations per hinge represent
that possibility without changing body mass, rotor tensors or action channels.
This is a hypothesis to calibrate, not an identified Jolt solver model.
"""
from pathlib import Path
import json
import math


def add_joint_compliance(spec, godot_spec, timeconst):
    import mujoco
    timeconst=float(timeconst)
    if not math.isfinite(timeconst) or not .01<=timeconst<=.05:
        raise ValueError('Compliance time must be 0.01–0.05 s at 200 Hz')
    source=json.loads(Path(godot_spec).read_text())
    names=[]
    for joint in source['joints']:
        if joint['type']!='hinge':continue
        body=spec.body(joint['body'])
        for axis in range(3):
            name=f"proxy_compliance_{joint['name']}_{axis}"
            vector=[0.,0.,0.];vector[axis]=1.
            body.add_joint(name=name,type=mujoco.mjtJoint.mjJNT_SLIDE,axis=vector,
                           pos=[0.,0.,0.],limited=False,range=[0.,0.],
                           damping=0.,frictionloss=0.,armature=0.,stiffness=0.,ref=0.,springref=0.)
            eq=spec.add_equality(name=name+'_zero',type=mujoco.mjtEq.mjEQ_JOINT,
                                 objtype=mujoco.mjtObj.mjOBJ_JOINT,name1=name,active=True,
                                 solref=[timeconst,1.],solimp=[.99,.99,.001,.5,2.])
            eq.data[:]=0.
            names.append(name)
    if len(names)!=42:raise ValueError('Expected three passive translations for 14 hinges')
    return dict(mode='passive_translation_equalities_v1',timeconst=timeconst,
                solimp=[.99,.99,.001,.5,2.],joint_names=names,
                scope='Training proxy only; extra passive coordinates, no added mass or actuator')
