"""Explicit, opt-in motion control experiments shared by play and offline replay.

These change policy commands, not physics or model weights. Their configuration
and version must accompany every trace and deployment comparison.
"""
import math

import numpy as np

from sim2sim.coords import quat_wxyz_to_mat


class MotionControl:
    def __init__(self,settings=None):
        self.settings=settings or {};self.reset()

    def reset(self):
        self.started=False;self.target_yaw=0.;self.brake_elapsed=0.;self.was_braking=False
        self.brake_speed=0.
        self.brake_target_yaw=0.
        self.walk_has_moved=False
        self.walk_was_moving=False
        self.walk_idle_elapsed=0.
        self.walk_path_started=False
        self.walk_path_origin=np.zeros(2)
        self.walk_path_yaw=0.
        self.walk_path_speed=0.

    def command(self,command,state,skill,dt=.02):
        out=np.asarray(command,np.float32).copy()
        rotation=quat_wxyz_to_mat(state.base_quat_wxyz)
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        ordinary=skill=='walking'
        if skill=='sprint':skill='walking'
        if not self.started or skill not in ('walking','roller'):
            self.target_yaw=yaw;self.started=True
        if skill not in ('walking','roller'):
            self.brake_elapsed=0.;self.was_braking=False;self.walk_has_moved=False;self.walk_path_started=False;self.walk_was_moving=False;return out
        if skill=='roller' and self.settings.get('heading_hold',False):
            self.target_yaw+=float(out[2])*dt
            difference=self.target_yaw-yaw
            out[2]=np.clip(math.atan2(math.sin(difference),math.cos(difference)),-1.,1.)
        elif skill=='walking' and self.settings.get('walk_heading_gain',0.)>0:
            idle_only=self.settings.get('walk_heading_scope','all')=='idle_after_motion'
            moving=math.sqrt(float(out[0])**2+float(out[1])**2)>.01 or abs(float(out[2]))>.05
            if moving and not self.walk_was_moving and self.settings.get('walk_reanchor_on_start',False):
                self.target_yaw=yaw
            self.walk_was_moving=moving
            if moving:self.walk_has_moved=True;self.walk_idle_elapsed=0.
            if abs(float(out[2]))>.05 or (idle_only and moving):
                self.target_yaw=yaw # Requested turns retain their original velocity command.
            elif not idle_only or self.walk_has_moved:
                if idle_only and self.walk_idle_elapsed<float(self.settings.get('walk_idle_delay_s',0.))-1e-9:
                    self.target_yaw=yaw
                else:
                    difference=self.target_yaw-yaw
                    error=math.atan2(math.sin(difference),math.cos(difference))
                    limit=float(self.settings.get('walk_heading_limit',.3))
                    out[2]=np.clip(float(self.settings['walk_heading_gain'])*error,-limit,limit)
                self.walk_idle_elapsed+=dt
        if skill=='walking':
            path_gain=float(self.settings.get('walk_ordinary_path_gain',self.settings.get('walk_path_gain',0.)) if ordinary else self.settings.get('walk_path_gain',0.))
            straight=float(command[0])>.01 and abs(float(command[1]))<.01 and abs(float(command[2]))<.05
            if path_gain>0. and straight:
                if not self.walk_path_started:
                    self.walk_path_origin=np.asarray(state.base_pos[:2],float).copy()
                    self.walk_path_yaw=yaw;self.walk_path_started=True
                    self.walk_path_speed=0.
                delta=np.asarray(state.base_pos[:2])-self.walk_path_origin
                error=-math.sin(self.walk_path_yaw)*float(delta[0])+math.cos(self.walk_path_yaw)*float(delta[1])
                damping=float(self.settings.get('walk_path_damping',0.))
                if damping>0.:
                    speed=-math.sin(self.walk_path_yaw)*float(state.base_linvel[0])+math.cos(self.walk_path_yaw)*float(state.base_linvel[1])
                    self.walk_path_speed=.8*self.walk_path_speed+.2*speed
                path_limit=float(self.settings.get('walk_path_limit',.15))
                out[1]=np.clip(-path_gain*error-damping*self.walk_path_speed,-path_limit,path_limit)
                lookahead=float(self.settings.get('walk_path_lookahead',0.))
                if lookahead>0. and self.settings.get('walk_heading_gain',0.)>0.:
                    # Face a point ahead on the original straight path. This
                    # couples cross-track correction to turn authority instead
                    # of relying entirely on learned sideways stepping.
                    self.target_yaw=self.walk_path_yaw-math.atan2(error,lookahead)
                    difference=self.target_yaw-yaw
                    heading_error=math.atan2(math.sin(difference),math.cos(difference))
                    limit=float(self.settings.get('walk_heading_limit',.3))
                    out[2]=np.clip(float(self.settings['walk_heading_gain'])*heading_error,-limit,limit)
            else:self.walk_path_started=False
            scale=float(self.settings.get('walk_translation_scale',1.))
            # Calibrate the actor's translational command; scoring retains the
            # user's requested velocity. Match GDScript double multiplication.
            out[0]=float(out[0])*scale
            out[1]=float(out[1])*scale
        braking=skill=='roller' and out[0]<0
        if braking:
            if not self.was_braking:self.brake_elapsed=0.
            heading_gain=float(self.settings.get('brake_heading_gain',0.))
            if heading_gain>0.:
                if not self.was_braking or abs(float(out[2]))>.05:self.brake_target_yaw=yaw
                if abs(float(out[2]))<=.05:
                    difference=self.brake_target_yaw-yaw
                    out[2]=np.clip(heading_gain*math.atan2(math.sin(difference),math.cos(difference)),-1.,1.)
            gain=float(self.settings.get('brake_velocity_gain',0.))
            if gain>0.:
                speed=math.cos(yaw)*float(state.base_linvel[0])+math.sin(yaw)*float(state.base_linvel[1])
                self.brake_speed=speed if not self.was_braking else .8*self.brake_speed+.2*speed
                out[0]=np.clip(-gain*self.brake_speed,-.5,.2)
            pulse=self.settings.get('brake_pulse_s')
            if pulse is not None and self.brake_elapsed>=float(pulse)-1e-9:out[0]=0.
            self.brake_elapsed+=dt
        else:self.brake_elapsed=0.
        self.was_braking=braking
        return out
