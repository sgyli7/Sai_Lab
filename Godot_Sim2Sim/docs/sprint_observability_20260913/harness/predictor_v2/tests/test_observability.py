"""Guard causal timing and world-frame targets in the offline research path."""
import copy
import unittest
import numpy as np
from sim2sim.research.observability import episode_features, arm_features, TAPS


def trace():
    rows = []
    for i in range(40):
        obs = np.zeros(61)
        obs[2] = i * .1
        obs[5] = -1
        obs[6] = i
        rows.append(dict(t=i*.02, obs=obs.tolist(), action=[i*.01]*14,
                         body=dict(base_quat=[2**-.5, 0, 0, 2**-.5],
                                   base_linvel=[i*.02, i*.04, 0], base_pos=[0, 0, .12]),
                         raw=dict(feet=[dict(name=name, ground_contact=True, impulse=.002)
                                        for name in ('ankle_left', 'ankle_right')]),
                         held=[] if i < 20 else ['sprint', 'fwd']))
    return rows


class ObservabilityTests(unittest.TestCase):
    def test_response_is_target_only_and_future_is_not_used(self):
        rows = trace()
        base = episode_features(rows)
        changed = copy.deepcopy(rows)
        changed[17]['body']['base_linvel'][1] += .2
        changed[17]['obs'][2] += 1
        response = episode_features(changed)
        np.testing.assert_array_equal(base['x'][0], response['x'][0])
        np.testing.assert_allclose(response['y'][0]-base['y'][0], [.2, 0, 1], atol=1e-7)
        changed[18]['obs'] = [999]*61
        changed[18]['body']['base_linvel'] = [999]*3
        future = episode_features(changed)
        np.testing.assert_array_equal(response['x'][0], future['x'][0])
        np.testing.assert_array_equal(response['y'][0], future['y'][0])


    def test_history_is_strictly_past_and_targets_use_current_yaw(self):
        result = episode_features(trace())
        for i, tap in enumerate(TAPS):
            assert result['x'][0, 82+61*i+6] == 16-tap
        np.testing.assert_allclose(result['y'][0], [.04, -.02, .1], atol=1e-7)
        assert not result['transition'][0]
        assert result['transition'][1] and result['moving'][1]


    def test_masked_and_duplicate_controls_cannot_see_extra_information(self):
        x = episode_features(trace())['x']
        altered = x.copy()
        altered[:, 75:] += 100
        for arm in ('immediate', 'duplicate_current'):
            np.testing.assert_array_equal(arm_features(x, arm), arm_features(altered, arm))
        visible = arm_features(x, 'history')
        np.testing.assert_array_equal(visible[:, 82:], x[:, 82:])
        assert not np.any(visible[:, 75:82])
