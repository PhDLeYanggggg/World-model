import copy
import numpy as np
import pytest

from src.world_model.m3w_source_separated_critics import partition_jobs, choices, matched_self


def test_nested_routing_uses_all_and_only_original_C_parts_without_mutating_roles():
    outer = dict(critic_sites=['a', 'b', 'c'], producer_sites=['P'], evaluation_sites=['H'])
    original = copy.deepcopy(outer)
    sites = np.array(['P', 'a', 'b', 'c', 'a', 'H'])
    parts = partition_jobs(sites, outer, dict(a=0, b=1, c=2))
    assert outer == original
    assert [r['names'] for r in parts] == [['a'], ['b'], ['c']]
    assert [r['routing_job']['evaluation_scope']['rows'] for r in parts] == [2, 1, 1]
    assert all(set(r['routing_job']['evaluation_sites']).isdisjoint(['P', 'H']) for r in parts)
    with pytest.raises(ValueError):
        partition_jobs(sites, outer, dict(a=0, b=1, H=2))


def test_count_and_amplitude_histogram_match_not_merely_total_count():
    own = np.array([1, 6, 1, 6, 2, 2])
    utility = np.array([9., 8., 7., 6., 5., 4.])
    veto = np.array([-1, -1, 1, 6, -1, 2])
    groups = [np.arange(4), np.array([4, 5])]
    actual = matched_self(own, utility, veto, groups, np.arange(6))
    np.testing.assert_array_equal(actual, [1, 6, -1, -1, 2, -1])
    for group in groups:
        np.testing.assert_array_equal(np.bincount(actual[group]+1, minlength=8), np.bincount(veto[group]+1, minlength=8))


def test_vetoes_keep_original_action_and_do_not_accept_future_argument():
    p = np.tile([4., .01, 10., 2., .001], (4, 7, 1))
    p[:, 6, 0] = [9., 8., 7., 6.]
    same, other = p.copy(), p.copy()
    same[0, 6, 1] = 1.
    other[1, 6, 4] = 1.
    mask = np.ones((4, 7), bool); mask[:, 0] = False
    args = (p, same, other, mask, [np.arange(4)], np.arange(4))
    original = [x.copy() for x in args[:4]]
    out = choices(*args)
    np.testing.assert_array_equal(out['same_data_veto'], [-1, 6, 6, 6])
    np.testing.assert_array_equal(out['disjoint_data_veto'], [6, -1, 6, 6])
    for a, b in zip(args[:4], original): np.testing.assert_array_equal(a, b)
    with pytest.raises(TypeError): choices(*args, future_endpoint=np.zeros((4, 2)))


def test_zero_reference_is_not_risk_certification_and_groups_cannot_overlap():
    p = np.tile([4., .01, 10., 2., .001], (2, 7, 1))
    q = p.copy(); q[:, :, 3] = 0
    mask = np.ones((2, 7), bool); mask[:, 0] = False
    out = choices(p, q, q, mask, [np.arange(2)], np.arange(2))
    assert np.all(out['disjoint_data_veto'] == -1)
    assert 'safety_pass' not in out
    with pytest.raises(ValueError): choices(p, q, q, mask, [np.array([0, 1]), np.array([1])], np.arange(2))
