import numpy as np
import pytest

from src.world_model import m3w_false_safe_diagnostic as api


def test_component_identity_and_signed_contributions():
    p = np.array([[1, .01, 1, .5, .001], [1, .02, 2, .8, .001]])
    y = np.array([[1, .05, 1, .1, .05], [1, .001, 2, 1., .001]])
    r = api.components(p, y)
    assert r['harm_underprediction_mass'] == pytest.approx(.049)
    assert r['reference_overprediction_budget_mass'] == pytest.approx(.004)
    assert r['realized_excess_mass'] == pytest.approx(.029)
    assert r['row_false_safe'] == 1


def test_unknown_kept_in_action_and_query_groups_not_independent_rows():
    p = np.tile([1, .01, 1, .5, .001], (3, 1))
    y = np.array([[1, .05, 1, .1, .05], [1, 0, 1, .5, 0], [np.nan]*5])
    r = api.diagnose(p, y, np.ones(3, bool), np.ones(3, bool), ['a']*3, [1, 1, 2], 1)
    assert r['selected_unknown'] == 1
    assert r['selected'] == 3
    assert r['query']['total_groups'] == 2
    assert r['query']['positive_excess_groups'] == 1
    assert r['recording']['total_groups'] == 1


def test_unknown_mask_and_empty_selected():
    p = np.zeros((2, 5)); y = np.ones((2, 5))
    r = api.diagnose(p, y, np.ones(2, bool), np.ones(2, bool), ['a', 'a'], [1, 2], 1)
    assert r['known_selected']['known_positive_easy_risk'] is None
    y[0, 1] = np.nan
    with pytest.raises(ValueError, match='Whole-row'):
        api.diagnose(p, y, np.ones(2, bool), np.ones(2, bool), ['a', 'a'], [1, 2], 1)


def test_moving_uses_cv_not_floor_and_never_labels():
    x = np.zeros((3, 380), np.float32)
    x[0, 355] = 1; x[1, 300] = 5; x[2, 378] = -1e-20
    np.testing.assert_array_equal(api.moving_from_features(x), [True, False, True])
    with pytest.raises(ValueError):
        api.moving_from_features(x[:, :-1])


def test_scale_invariance_and_support_fallback():
    p = np.tile([1, .01, 1, .5, .001], (2, 1)); y = p*2
    args = (np.ones(2, bool), np.array([True, False]), ['a', 'b'], [1, 2])
    a = api.diagnose(p, y, *args, 1)
    b = api.diagnose(p*100, y*100, *args, 100)
    assert a == b
    assert a['selected'] == 1
