import numpy as np
import pytest

from scripts.audit_m3w_opendd_full_grid_exchange import certificate, summarize_view


def test_certified_headroom_does_not_count_keep_on_the_forced_mask():
    z = [(True, True, True), (True, False, False), (False, False, False)]
    r = summarize_view(z, 1)
    np.testing.assert_array_equal(r['incumbent'], [1, 1, 1])
    np.testing.assert_array_equal(r['mask_oracle'], [1, 0, 0])
    np.testing.assert_array_equal(r['uniform'], [.5, 0, 0])
    np.testing.assert_array_equal(r['menu_or_keep'], [1, 1, 1])
    r = summarize_view(z, None)
    np.testing.assert_array_equal(r['aligned'], r['mask_oracle'])
    with pytest.raises(ValueError, match='Invalid frozen'):
        summarize_view(z, 0)


def test_missing_easy_membership_can_disqualify_an_original_pass():
    cv = np.zeros((2, 12, 2))
    p, y = cv+.01, cv+.1
    valid = np.ones((2, 12), bool)
    valid[:, -1] = False
    score = dict(dual_positive=True, edits=2, vs_cv=.3, vs_ols4=.1)
    assert certificate(cv, p, y, valid, .2, score) == (False, ('easy_support',))
    valid[0, -1] = True
    assert certificate(cv, p, y, valid, .2, score) == (True, ())
    score['dual_positive'] = False
    assert certificate(cv, p, y, valid, .2, score) == (False, ())
