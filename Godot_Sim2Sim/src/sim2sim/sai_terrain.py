"""Geometric edge detection in the wheel corridor, independent of gait phase."""
import numpy as np

EDGE_X = np.arange(46) * .02 - .30
EDGE_Y = np.array([-.146, 0., .146])


def step_in_path(state, crouch=False):
    dense = state.get('terrain_edge_heights')
    if dense is not None:
        edge = _signed_edges(dense, crouch)
        return bool(np.any(np.abs(edge) > .008))
    path = state.get('terrain_path_heights')
    if path is None:
        return True
    h = np.asarray(path, dtype=float)
    if h.shape != (15,) or not np.isfinite(h).all():
        raise ValueError('Invalid wheel-path terrain scan')
    h = h.reshape(5,3)[:4 if crouch else 5]
    # Compatibility only: coarse samples can reject an exact plane, but cannot
    # distinguish all curved surfaces from steps. New game clients send dense rays.
    delta = np.diff(h, axis=0)
    return bool(np.max(np.abs(delta-np.median(delta,axis=0))) > .008)


def _signed_edges(dense, crouch=False):
    h = np.asarray(dense, dtype=float)
    if h.shape != (138,) or not np.isfinite(h).all():
        raise ValueError('Invalid dense wheel-path terrain scan')
    delta = np.diff(h.reshape(46, 3), axis=0)
    padded = np.pad(delta, ((2, 2), (0, 0)), mode='edge')
    background = np.median(np.stack([padded[i:i+45] for i in (0,1,3,4)]), axis=0)
    near = (EDGE_X[:-1] >= -.20) & (EDGE_X[:-1] < (.36 if crouch else .54))
    return (delta-background)[near]


def descending_in_path(state):
    """Only unambiguous downward edges use contact-following descent."""
    dense = state.get('terrain_edge_heights')
    if dense is None:
        return False
    edge = _signed_edges(dense)
    return bool(np.any(edge < -.008) and not np.any(edge > .008) and np.min(edge) >= -.045)
