import numpy as np
import pytest
from src.evaluation.m3w_cost_head_transfer import choices
from src.world_model.m3w_bounded_cost_head import ARMS


def fixture():
    past = np.zeros((6, 8, 2)); past[1:, -1, 0] = 1
    d = np.array([1., 1., 0., 1., 1., 1.])
    s = np.array([[100., 0.], [10., 1.], [100., 0.], [10., 1.01], [1., 2.], [0., 0.]])
    return {a:s.copy() for a in ARMS}, past, d, np.arange(6)


def test_exact_frozen_guard_and_zero_gain_rejection():
    s, p, d, ids = fixture(); r = choices(s, p, d, ids)
    for arm in ARMS:
        np.testing.assert_array_equal(r[arm+'_strict_stop'], [False, True, False, False, False, False])
        np.testing.assert_array_equal(r[arm+'_net_stop'], [False, True, False, True, False, False])
        assert r[arm+'_matched_count'].sum() == 1
    assert not r['floor'].any() and r['uncontrolled'].all()


def test_identical_count_controls_are_not_threshold_search():
    s, p, d, ids = fixture()
    s['direct_native'][3] = [30, 2]
    r = choices(s, p, d, ids)
    assert r['direct_native_strict_stop'].sum() == 2
    assert r['direct_native_matched_count'].sum() == 1
    assert r['direct_native_matched_count'][3]


def test_tied_rank_is_stable_by_query_identity():
    s, p, d, ids = fixture(); s['direct_native'][3] = s['direct_native'][1]
    original = choices(s, p, d, ids)
    order = np.array([5, 3, 1, 0, 2, 4])
    again = choices({a:x[order] for a,x in s.items()}, p[order], d[order], ids[order])
    for key in original:
        np.testing.assert_array_equal(np.sort(ids[original[key]]), np.sort(ids[order][again[key]]))


def test_empty_reference_capacity_stays_empty():
    s, p, d, ids = fixture(); s['bounded_fraction'][:] = [0, 2]
    r = choices(s, p, d, ids)
    assert all(not r[a+'_matched_count'].any() for a in ARMS)


@pytest.mark.parametrize('bad', ['nan_score', 'negative_distance', 'duplicate_ids', 'nan_past'])
def test_invalid_causal_contract_rejected(bad):
    s, p, d, ids = fixture()
    if bad == 'nan_score': s['direct_native'][0, 0] = np.nan
    if bad == 'negative_distance': d[0] = -1
    if bad == 'duplicate_ids': ids[0] = ids[1]
    if bad == 'nan_past': p[0, 0, 0] = np.nan
    with pytest.raises(ValueError): choices(s, p, d, ids)


def test_future_labels_are_not_accepted_by_decision_api():
    s, p, d, ids = fixture()
    with pytest.raises(TypeError): choices(s, p, d, ids, future_endpoint=np.zeros((6, 2)))
