import numpy as np

from scripts.run_m3w_recipient_shortfall import capacity, recipient_rules


def test_shortfall_prefers_supported_jointly_feasible_pair_over_gain():
    gain = np.array([10., 9., 7., 1.])
    a = np.array([5., 3., -3., -4.])
    e = np.array([3., 4., -4., -2.])
    ref = np.array([0., 2., 1., 0.])
    g, s, risk = recipient_rules(gain, a, e, ref, np.ones(4, bool), 2)
    np.testing.assert_array_equal(g, [0, 1])
    np.testing.assert_array_equal(s, [1, 0])
    np.testing.assert_array_equal(risk, [1, 2])


def test_count_bound_can_rule_out_original_nomination_but_not_full_bank():
    cv = np.zeros((2, 2, 2))
    y = np.ones_like(cv)
    bank = np.stack((cv-1., y), axis=1)
    z = capacity(cv, bank, np.zeros(2, int), np.array([True, False]), [np.arange(2)],
                 y, np.ones((2, 2), bool), 5.)
    assert z[0]['provably_infeasible']
    assert not z[1]['provably_infeasible']
