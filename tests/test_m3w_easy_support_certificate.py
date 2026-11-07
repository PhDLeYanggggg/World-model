import numpy as np

from scripts.audit_m3w_easy_support_certificate import assess


def test_one_distinct_easy_action_is_necessary_and_sufficient_for_the_free_subset():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 2., 10.])[:, None]
    p = cv.copy()
    p[0, :, 0] = -.03
    p[2] = y[2]
    valid = np.ones((3, 12), bool)
    out = assess(cv, p[:, None], y, valid, np.ones(3, bool), 2.)
    assert out['distinct_easy_paths'] == 1 and not out['feasible']
    np.testing.assert_allclose(out['minimum_risk_ratio'], .03)
    p[1] = y[1]
    assert assess(cv, p[:, None], y, valid, np.ones(3, bool), 2.)['feasible']
    assert not assess(cv, p[:, None], y, valid, np.array([True, False, True]), 2.)['feasible']
