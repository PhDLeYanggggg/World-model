import numpy as np
import pytest
from src.world_model.m3w_selection_exchange import account, check_queries, partition


def test_benefit_lost_despite_less_harm():
    f = np.array([10., 10., 10., 10.])
    n = np.array([2., 14., 9., 11.])
    old = np.array([True, True, False, False])
    new = ~old
    r = account(f, n, new, old, np.ones(4, bool), np.ones(4), 2.)
    assert r['metric']['benefit_change_percent'] == pytest.approx(-700/36)
    assert r['metric']['harm_change_percent'] == pytest.approx(-300/36)
    assert r['metric']['net_ADE_gain_percent'] == pytest.approx(-400/36)
    assert sum(x['rows'] for x in r['sums'].values()) == 4


def test_unknown_rows_kept_but_not_assigned_zero_cost():
    r = account(np.array([1., np.nan]), np.array([2., np.nan]), np.array([False, True]),
        np.array([True, False]), np.ones(2, bool), np.ones(2), 1.)
    assert r['sums']['new_only']['rows'] == 1
    assert r['sums']['new_only']['unknown'] == 1
    assert r['sums']['new_only']['known'] == 0
    assert r['metric']['net_ADE_gain_percent'] == 50


def test_zero_reference_is_undefined_not_zero():
    r = account(np.zeros(2), np.zeros(2), np.ones(2, bool), np.ones(2, bool),
        np.ones(2, bool), np.zeros(2), 1.)
    assert r['metric']['net_ADE_gain_percent'] is None


def test_query_matching_includes_unknown_and_rejects_future_slot_borrowing():
    new = np.array([True, False, False, True])
    old = ~new
    x = check_queries(new, old, np.ones(4, bool), np.array(['v']*4), np.array([1,1,2,2]), np.arange(4))
    assert x == dict(queries=2, changed_queries=2)
    with pytest.raises(ValueError, match='current-query'):
        check_queries(new, np.array([True, True, False, False]), np.ones(4, bool),
            np.array(['v']*4), np.array([1,1,2,2]), np.arange(4))


def test_partition_rejects_ineligible_switch():
    with pytest.raises(ValueError, match='eligible'):
        partition(np.ones(2, bool), np.zeros(2, bool), np.zeros(2, bool))


def test_future_label_support_mismatch_rejected():
    with pytest.raises(ValueError, match='labels'):
        account(np.array([np.nan]), np.zeros(1), np.zeros(1, bool), np.zeros(1, bool),
            np.ones(1, bool), np.zeros(1), 1.)
