import numpy as np
import pytest
from src.world_model.m3w_incumbent_relative import features, targets, choices, replay


def test_cost_roles_invert_only_where_incumbent_is_neural():
    cv = np.array([4., 4., 0., np.nan])
    f = np.array([4., 4., 0., np.nan]); n = np.array([1., 1., 2., np.nan])
    old = np.array([False, True, False, False])
    u, r = targets(cv, f, n, old, arm='incumbent_reference', event='all', easy_cut=5)
    np.testing.assert_array_equal(u[:3], [[3, 0], [0, 3], [0, 2]])
    np.testing.assert_array_equal(r[:3], [[4, 0], [1, 3], [0, 2]])
    assert np.isnan(u[3]).all() and np.isnan(r[3]).all()
    _, easy = targets(cv, f, n, old, arm='incumbent_reference', event='easy', easy_cut=5)
    np.testing.assert_array_equal(easy[2], [0, 0])


@pytest.mark.parametrize('direction', ['both', 'add', 'remove'])
def test_policy_falls_back_to_incumbent_and_scalar_replay(direction):
    old = np.array([False, True, True, False])
    move = np.array([True, True, True, False])
    u = np.array([[3., 0.], [3, 0], [0, 3], [3, 0]])
    r = np.array([[2., .01]]*4)
    out = choices(u, r, move, old, arm='incumbent_reference', direction=direction)
    np.testing.assert_array_equal(out, replay(u, r, move, old, arm='incumbent_reference', direction=direction))
    assert out[2] and not out[3]
    if direction == 'add': assert np.all(out[old])
    if direction == 'remove': assert not np.any(out[~old])


def test_safety_rejects_invalid_scores_and_stopping_violation():
    with pytest.raises(ValueError): choices([[1, 0]], [[1, 0]], [False], np.array([True]), arm='incumbent_reference')
    with pytest.raises(ValueError): choices([[np.nan, 0]], [[1, 0]], [True], np.array([False]), arm='incumbent_reference')
    out = choices([[1, 0]], [[1, .021]], [True], np.array([True]), arm='incumbent_reference')
    assert out[0]


def test_identical_causal_inputs_for_both_target_arms():
    x = np.zeros((2, 380)); old = np.array([True, False])
    out = features(x, old)
    assert out.shape == (2, 381) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, -1], old)
    with pytest.raises(ValueError): features(x, [1, 0])


def test_budget_and_closed_roles():
    import json
    from pathlib import Path
    c = json.loads(Path('configs/m3w_european_incumbent_relative_v1.json').read_text())
    assert c['groups'] == 36 and c['views'] == 288 and c['new_neural_heads'] == 144
    assert c['neural_updates'] == 144*2000 and c['ridge_fits'] == 72
    assert c['head_training']['width'] == 64 and c['risk_budget'] == .02
    assert c['bootstrap_resamples'] == 3000
    assert not any(c[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
        'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
