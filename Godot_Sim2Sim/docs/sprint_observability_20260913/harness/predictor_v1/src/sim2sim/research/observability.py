"""Causal, pre-action features for offline native dynamics diagnostics.

These are predictors, not deployable policies or acceptance scores. Whole reset
seeds must be separated by the caller, including all policy/noise repetitions.
"""
import numpy as np


TAPS = (1, 2, 4, 8, 16)
WIDTH = 61 + 14 + 3 + 4 + 61 * len(TAPS)
ARMS = ('immediate', 'state', 'contacts', 'history', 'duplicate_current')


def episode_features(rows, stride=4):
    """Predict 20 ms innovations; never use response state in current inputs.

    Velocity differences use the CURRENT yaw frame. Angular target is the
    change in world-vertical angular velocity, not Euler yaw differentiation.
    Contacts are sampled at the same pre-action point as observations.
    """
    if stride < 1:
        raise ValueError('Positive sampling stride required')
    if len(rows) <= max(TAPS) + 1:
        raise ValueError('Episode shorter than context and one response')
    times = np.array([r['t'] for r in rows], dtype=np.float64)
    if not np.allclose(np.diff(times), .02, rtol=0, atol=1e-7):
        raise ValueError('Expected contiguous 50 Hz decisions')
    if any(r.get('fell', False) for r in rows):
        raise ValueError('Diagnostic corpus protocol excludes fall episodes')
    obs = np.asarray([r['obs'] for r in rows], dtype=np.float64)
    action = np.asarray([r['action'] for r in rows], dtype=np.float64)
    if obs.shape != (len(rows), 61) or action.shape != (len(rows), 14):
        raise ValueError('Unexpected policy contract')
    quat = np.asarray([r['body']['base_quat'] for r in rows], dtype=np.float64)
    w, x, y, z = quat.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    vel = np.asarray([r['body']['base_linvel'] for r in rows], dtype=np.float64)
    height = np.asarray([r['body']['base_pos'][2] for r in rows])
    contact = []
    for r in rows:
        feet = {f['name']: f for f in r['raw']['feet']}
        values = []
        for name in ('ankle_left', 'ankle_right'):
            foot = feet[name]
            values.extend((float(foot['ground_contact']), np.log1p(max(0., foot['impulse']) / .001)))
        contact.append(values)
    contact = np.asarray(contact)
    ix = np.arange(max(TAPS), len(rows)-1, stride)
    cs, sn = np.cos(yaw[ix]), np.sin(yaw[ix])

    def planar(v):
        return np.column_stack((cs*v[:, 0]+sn*v[:, 1], -sn*v[:, 0]+cs*v[:, 1]))

    state = np.column_stack((planar(vel[ix]), height[ix]))
    features = np.concatenate((obs[ix], action[ix], state, contact[ix],
                               *(obs[ix-tap] for tap in TAPS)), axis=1)
    # obs angular velocity and projected gravity share the body frame.
    vertical_angular = np.sum(obs[:, :3] * -obs[:, 3:6], axis=1)
    target = np.column_stack((planar(vel[ix+1]-vel[ix]),
                              vertical_angular[ix+1]-vertical_angular[ix]))
    last_change = times[0]
    previous = frozenset(rows[0]['held'])
    transition = np.zeros(len(rows), dtype=bool)
    moving = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        held = frozenset(r['held'])
        if held != previous:
            last_change = times[i]
        transition[i] = times[i]-last_change < 1.-1e-7 and last_change > times[0]
        moving[i] = 'fwd' in held
        previous = held
    if features.shape[1] != WIDTH or not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError('Invalid diagnostic features')
    return dict(x=features.astype(np.float32), y=target.astype(np.float32),
                t=times[ix], moving=moving[ix], transition=transition[ix], index=ix)


def arm_features(standardized, arm):
    """Apply after train-only normalization; all networks have identical width."""
    if arm not in ARMS:
        raise ValueError(arm)
    x = standardized.copy()
    x[:, 75:] = 0
    if arm == 'state':
        x[:, 75:78] = standardized[:, 75:78]
    elif arm == 'contacts':
        x[:, 78:82] = standardized[:, 78:82]
    elif arm == 'history':
        x[:, 82:] = standardized[:, 82:]
    elif arm == 'duplicate_current':
        for i in range(len(TAPS)):
            x[:, 82+61*i:82+61*(i+1)] = standardized[:, :61]
    return x
