"""G1 Homie mobile policy contract: measured state32, effective action32.

The native GR00T decoder additionally emits waist3. HomieV2Config.enable_waist
is False: the original action term passes only the 28 arm/hand targets to WBC.
Legs and waist remain outputs of the frozen Homie balance/walking networks.
No object pose, simulator map or route waypoint enters either VLA observation.
"""
from __future__ import annotations
import numpy as np
from experiments.pi05_g1.contract import (
    GROUPS, UPPER_NAMES, WAIST_NAMES, STATE_NAMES, ACTION_NAMES, DATASET_NAMES,
    encode_state, encode_action, encode_row, decode_to_source, normalize, denormalize,
)
SCHEMA = 'pi05_g1_mobile_effective32_v1'
PROMPT = ('Pick up the brown box from the shelf, and place it into the blue bin '
          'on the table located at the right of the shelf.')
CONTROL_HZ = 50
PHYSICS_HZ = 200
ACTION_HORIZON = 50


def native_outputs_to_actions(outputs):
    """Keep native waist output as evidence while projecting the inactive group."""
    widths = {**{k:len(v) for k,v in GROUPS.items()}, 'waist':3,
              'base_height_command':1, 'navigate_command':3}
    if set(outputs) != set(widths):
        raise ValueError('Native mobile output groups differ from the pinned contract')
    arrays = {}
    for key,width in widths.items():
        x = np.asarray(outputs[key])
        if x.shape != (1,50,width) or not np.isfinite(x).all():
            raise ValueError('Malformed native action group: '+key)
        arrays[key] = x.astype(np.float32,copy=False)
    actions = np.concatenate([arrays[key][0] for key in (
        'left_arm','right_arm','left_hand','right_hand','base_height_command','navigate_command')],axis=1)
    return actions,arrays['waist'][0]


def observation_groups(q43,names43):
    q = np.asarray(q43,dtype=np.float32)
    # Shared encoder validates the complete named43 set, not simulator index order.
    encode_state(q,names43)
    return {key:q[[names43.index(name) for name in group]].reshape(1,1,-1)
            for key,group in {**GROUPS,'waist':WAIST_NAMES}.items()}
