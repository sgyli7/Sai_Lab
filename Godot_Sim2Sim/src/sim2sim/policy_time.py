"""Explicit, model-declared elapsed-time input for finite maneuvers."""
import math
import numpy as np


def time_input_seconds(metadata):
    value=float(metadata.get("sim2sim_time_input_s",0.))
    if not math.isfinite(value) or value<0:raise ValueError("Invalid policy time-input duration")
    return value


def has_heading_input(metadata):
    mode=metadata.get("sim2sim_heading_input","")
    if mode not in ("","lateral_axis_sin_cos"):raise ValueError("Unknown relative heading input")
    if mode and not time_input_seconds(metadata):raise ValueError("Heading input requires a declared timed maneuver")
    return bool(mode)


def time_command(elapsed,seconds,rotation=None,heading=None):
    if seconds<=0:raise ValueError("A time-input policy requires a positive duration")
    out=np.zeros(13,np.float32);out[0]=np.clip(elapsed/seconds,0.,1.)
    if rotation is not None:
        target=np.array([-heading[1],heading[0]])
        lateral=np.asarray(rotation)[:2,1];lateral=lateral/max(float(np.linalg.norm(lateral)),1e-6)
        out[1]=target[0]*lateral[1]-target[1]*lateral[0]
        out[2]=target@lateral
    return out
