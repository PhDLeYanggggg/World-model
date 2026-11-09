import numpy as np
import pytest

from scripts import run_m3w_opendd_relative_trust as run


def test_cap_filters_before_retrieval_and_does_not_pad():
    feasibility = np.array([.1, .8, .6, .5, .9, .3])
    gain = np.array([0., .5, .4, -.1, .6, .2])
    bound = np.array([0., 1., .49, .5, 1.2, .51])
    shortlist, eligible = run.capped_shortlist(feasibility, gain, bound, original=1)
    np.testing.assert_array_equal(eligible, [0, 2, 3])
    np.testing.assert_array_equal(shortlist, [2, 3, 0])
    assert run.pick(gain[shortlist], shortlist) == 2
    assert run.pick(-np.abs(gain[shortlist]), shortlist) == shortlist[0]


def test_zero_cap_has_legal_anchor_and_rejects_malformed_menu():
    feasibility, gain = np.array([.4, .9]), np.array([0., .1])
    short, eligible = run.capped_shortlist(feasibility, gain, np.array([0., .3]), original=0)
    np.testing.assert_array_equal(short, [0])
    np.testing.assert_array_equal(eligible, [0])
    with pytest.raises(ValueError, match="legal additive"):
        run.capped_shortlist(feasibility, gain, np.array([.01, .3]), original=0)


def test_common_anchor_bound_change_is_not_direct_pair_bound():
    a = np.zeros((2, 12, 2))
    p = a.copy()
    p[1, 1:, 0] = 2
    v = np.zeros((2, 12), dtype=bool)
    v[:, 0] = True
    q = np.zeros(2, dtype=int)
    parent = run.debit.components(p, a, a, v, q)
    trust = run.debit.components(p*.5, a, a, v, q)
    direct = run.debit.components(p*.5, p, a, v, q)
    assert trust[1] == pytest.approx(parent[1]*.5)
    assert trust[-1]-parent[-1] > 0
    assert direct[-1] < 0
