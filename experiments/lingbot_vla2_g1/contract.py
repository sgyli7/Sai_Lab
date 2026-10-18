"""Explicit local 55-slot embedding of the unchanged mobile effective32 API.

This is a local G1 schema, not an upstream-provided G1 embodiment mapping.
All joint positions remain absolute radians; height m; navigation m/s, m/s,rad/s.
Hands use slots39:53, extending the upstream hand12 into two reserved pad slots.
"""
import numpy as np
from experiments.pi05_g1_mobile.contract import (
    ACTION_NAMES,STATE_NAMES,UPPER_NAMES,WAIST_NAMES,PROMPT,CONTROL_HZ,ACTION_HORIZON,
    encode_state,encode_action,encode_row,decode_to_source,
)
SCHEMA='lingbot_vla2_g1_homie_effective32_in_55_v1'
ACTION_SLOTS=np.array([*range(14),*range(39,53),30,36,37,38])
STATE_SLOTS=np.array([*range(14),*range(39,53),31,32,33])
ACTION_MASK=np.zeros(55,dtype=bool);ACTION_MASK[ACTION_SLOTS]=True
STATE_MASK=np.zeros(55,dtype=bool);STATE_MASK[STATE_SLOTS]=True


def pack_actions(values):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=32 or not np.isfinite(values).all():raise ValueError('Finite effective32 required')
    result=np.zeros((*values.shape[:-1],55),dtype=np.float32);result[...,ACTION_SLOTS]=values;return result


def unpack_actions(values):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=55 or not np.isfinite(values).all():raise ValueError('Finite model55 required')
    return values[...,ACTION_SLOTS].copy()


def pack_state(values):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=32 or not np.isfinite(values).all() or np.any(values[...,-1]!=0):raise ValueError('State31 plus zero padding required')
    result=np.zeros((*values.shape[:-1],55),dtype=np.float32);result[...,STATE_SLOTS]=values[...,:31];return result


def unpack_state(values):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=55 or not np.isfinite(values).all():raise ValueError('Finite model55 required')
    result=np.zeros((*values.shape[:-1],32),dtype=np.float32);result[...,:31]=values[...,STATE_SLOTS];return result


def manifest():
    return {'schema':SCHEMA,'external_action_names':list(ACTION_NAMES),'external_state_names':list(STATE_NAMES),'action_slots':ACTION_SLOTS.tolist(),'state_slots':STATE_SLOTS.tolist(),'action_valid_dimensions':int(ACTION_MASK.sum()),'state_valid_dimensions':int(STATE_MASK.sum()),'waist_slots30_to33':['action_height_m','state_waist_yaw_rad','state_waist_roll_rad','state_waist_pitch_rad'],'base_slots36_to38':'action velocity vx_m_s,vy_m_s,yaw_rate_rad_s; state unobserved/masked','hand_slots39_to52':'left7 then right7 absolute positions; local extension into two upstream padding slots','unused':'zero state/target, zero action-loss mask','relative_actions':False,'input':'head RGB, self joint state, fixed prompt; no actor pose or route hints','prompt':PROMPT,'control_hz':CONTROL_HZ,'action_horizon':ACTION_HORIZON}
