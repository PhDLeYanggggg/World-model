import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from src.world_model.m3w_zero_atom import zero_event, source_weights, fit_tree, predict, allowed, matched_with_incumbent


def fixture():
    x = np.arange(20, dtype=np.float32)[:, None]
    event = np.arange(20) < 4
    weights = np.arange(1, 21).astype(float)
    forest = ExtraTreesRegressor(n_estimators=3, max_depth=2, min_samples_leaf=2, random_state=17).fit(x, event, sample_weight=weights)
    readouts = [fit_tree(t, x, event, weights) for t in forest.estimators_]
    return x, event, weights, forest, readouts


def test_zero_is_in_easy_labels_but_unknown_is_not():
    q = np.array([[0, 1, 1, 0, 0, 1], [.5, 0, 0, .5, .2, 1], [0, 0, 0, 0, 0, 0]], float)
    np.testing.assert_array_equal(zero_event(q, np.array([True, True, False])), [True, False, False])


def test_source_unknown_and_zero_distance_exclusion():
    np.testing.assert_array_equal(source_weights(np.array([True, True, False]), np.array([3, 4, 0]), np.array([2., 0, 1])), [3, 0, 0])
    with pytest.raises(ValueError): source_weights(np.array([True, False]), np.array([2, 1]), np.ones(2))


def test_leaf_readout_reconstructs_binary_regression():
    x, _, _, forest, r = fixture()
    p = predict(forest, r, x)
    np.testing.assert_allclose(p['probability'], forest.predict(x), rtol=1e-15)
    assert p['supported'].all() and (p['minimum_unique_rows'] >= 2).all()


def test_zero_weight_rows_cannot_change_fit():
    x, y, w, forest, r = fixture()
    xx = np.concatenate((x, [[999.]])); yy = np.r_[y, True]; ww = np.r_[w, 0]
    for tree, old in zip(forest.estimators_, r):
        new = fit_tree(tree, xx, yy, ww)
        for k in old: np.testing.assert_array_equal(old[k], new[k])


def test_changed_source_weights_rejected():
    x, y, w, forest, _ = fixture()
    with pytest.raises(AssertionError): fit_tree(forest.estimators_[0], x, y, w*2)


def test_missing_tree_readout_rejected():
    x, _, _, f, r = fixture()
    with pytest.raises(ValueError): predict(f, r[:1], x)


def test_no_small_probability_tolerance_or_unsupported_admission():
    np.testing.assert_array_equal(allowed(np.array([0., 1e-100, .2, 0.]), np.array([True, True, True, False])), [True, False, False, False])


@pytest.mark.parametrize('probability', [-.1, 1.1, np.nan])
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError): allowed(np.array([probability]), np.array([True]))


def test_count_control_retains_incumbent_on_failed_solver():
    def failed(*args, **kwargs): return np.zeros(2, bool), {'status':'failed','optimal':False}
    bits, report = matched_with_incumbent(np.array([1., 2]), np.array([0., 1]), np.ones(2,bool), 0., np.array([True,False]), failed)
    np.testing.assert_array_equal(bits, [True, False])
    assert report['status']=='feasible_incumbent_retained' and not report['optimal']


def test_count_control_accepts_better_feasible_proposal():
    def solved(*args, **kwargs): return np.array([False,True]), {'status':'optimal','optimal':True}
    bits, _ = matched_with_incumbent(np.array([1.,2]), np.zeros(2), np.ones(2,bool), 0., np.array([True,False]), solved)
    np.testing.assert_array_equal(bits, [False,True])


def test_infeasible_incumbent_rejected():
    with pytest.raises(ValueError): matched_with_incumbent(np.ones(2), np.ones(2), np.ones(2,bool), 0., np.array([True,False]), None)
