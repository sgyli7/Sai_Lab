"""Contract tests for measured arrival; dynamics are checked by the native coupons."""
import copy
import numpy as np
import pytest
from sim2sim.support_arrival import SupportArrival


def snapshot(t,p=(0.,0.,.224),speed=0.,contacts=4):
    return dict(time=t,q=[0.]*25,v=[0.]*25,
        support_route=dict(serial=0,points_front_xy_m=[[1.,0.]],initial_park_s=3.),
        support_frame=dict(body_id='front',base_position=list(p),base_rotation_columns=np.eye(3).tolist(),
            base_linear_world=[speed,0.,0.],base_angular_world=[0.,0.,0.],projected_up_body=[0.,0.,1.],
            terrain_heights=[0.]*24,wheel_contacts=[i<contacts for i in range(4)]))


def enter_brake():
    c=SupportArrival();c.update(snapshot(3.))
    for i in range(1,43):c.update(snapshot(3.+.02*i,p=(.4 if i<42 else .98,0.,.224)))
    assert c.phase=='brake_settle'
    return c,3.84


def test_radius_is_not_arrival_and_requires_real_stability():
    c,t=enter_brake()
    for i in range(1,61):
        state=snapshot(t+.02*i,p=(.98,0.,.224),speed=.1 if i<30 else 0.,contacts=2)
        _,r=c.update(state)
        assert r['phase']=='brake_settle' and not r['done']
    t+=1.2
    for i in range(1,27):_,r=c.update(snapshot(t+.02*i,p=(.98,0.,.224)))
    assert r['phase']=='parked' and not r['done']
    assert r['stops'][0]['brake_started_s']==pytest.approx(3.84)


def test_original_brake_origin_survives_settling_and_hold():
    c,t=enter_brake()
    for i in range(1,130):
        _,r=c.update(snapshot(t+.02*i,p=(.995,0.,.224)))
    assert r['done']
    assert r['stops'][0]['maximum_displacement_from_brake_m']==pytest.approx(.015)
    assert r['stops'][0]['stable_max_displacement_m']==0.
    assert r['stops'][0]['stable_declared_s']>r['stops'][0]['brake_started_s']


def test_lost_contact_revokes_stable_park_without_discarding_brake_origin():
    c,t=enter_brake()
    for i in range(1,30):c.update(snapshot(t+.02*i,p=(.98,0.,.224)))
    assert c.phase=='parked'
    _,r=c.update(snapshot(t+.6,p=(.98,0.,.224),contacts=2))
    assert r['phase']=='brake_settle' and r['parked_at_s'] is None
    assert r['stops'][0]['brake_started_s']==pytest.approx(3.84)
    assert r['events'][-1]['type']=='stability_lost'


def test_no_precision_arrival_on_uneven_scans_and_input_is_untouched():
    c=SupportArrival();s=snapshot(3.,p=(.98,0.,.224));s['support_frame']['terrain_heights'][0]=.02
    saved=copy.deepcopy(s);out,r=c.update(s)
    assert s==saved and r['phase']=='cruise' and not out['support_precision']
    missing=snapshot(3.02);del missing['support_frame']['wheel_contacts']
    with pytest.raises(ValueError,match='contact'):c.update(missing)


def test_fixed_path_lookahead_does_not_flip_into_in_place_turn_at_goal_plane():
    c=SupportArrival();c.update(snapshot(3.))
    before,_=c.update(snapshot(3.02,p=(.99,.04,.224)))
    after,r=c.update(snapshot(3.04,p=(1.01,.04,.224)))
    assert r['phase']=='approach' and not r['done']
    assert r['distance_m']>.025
    assert before['command'][1]<=0 and after['command'][1]<=0
    assert abs(after['command'][1])<=after['command'][0]*c.CONFIG['maximum_curvature_per_m']
    assert r['steering_curvature_per_m']<0
