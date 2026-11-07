import numpy as np
import pytest

from src.world_model.m3w_prefix_velocity import PRIOR, candidates, prefix_weights


def test_constant_velocity_and_stationary_path_are_preserved():
    h = np.zeros((2, 8, 2))
    h[0, :, 0] = np.arange(8)
    weights = prefix_weights(h)
    np.testing.assert_allclose(weights, np.tile(PRIOR, (2, 1)))
    out = candidates(h, np.array([4, 4]))
    for p in out.values():
        np.testing.assert_allclose(p[0, :, 0], np.arange(1, 13))
        np.testing.assert_array_equal(p[1], 0)


def test_prefix_regret_changes_weights_without_using_current_query_labels():
    h = np.zeros((3, 8, 2))
    h[0, :, 0] = np.arange(8)**2
    h[1, :, 0] = [0., 1., 2., 3., 2., 1., 0., -1.]
    w = prefix_weights(h)
    assert w[0, 0] > PRIOR[0]
    assert not np.allclose(w[0], w[1])
    p = candidates(h, np.array([0, 0, 1]))
    shifted = candidates(h + [127., -39.], np.array([0, 0, 1]))
    for k in p:
        np.testing.assert_allclose(p[k], shifted[k])
    np.testing.assert_allclose(p['agent_prefix'][2], p['query_prefix'][2])
    assert not np.allclose(p['agent_prefix'][0], p['query_prefix'][0])
    assert not np.allclose(p['agent_prefix'][0], p['fixed_prefix'][0])


def test_prefix_input_contract_rejects_future_or_nonfinite_positions():
    with pytest.raises(ValueError):
        prefix_weights(np.zeros((2, 20, 2)))
    h = np.zeros((2, 8, 2))
    h[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        prefix_weights(h)
