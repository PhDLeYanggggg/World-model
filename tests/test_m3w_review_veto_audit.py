import numpy as np
import pytest

from scripts.audit_m3w_review_veto import rank_auc, region_stats


@pytest.mark.parametrize("score,label,expected", [
    ([0, 1], [False, True], 1), ([1, 0], [False, True], 0),
    ([1, 1], [False, True], .5), ([1, 1, 2], [False, True, True], .75),
    ([], [], None), ([1], [True], None),
])
def test_rank_auc(score, label, expected):
    assert rank_auc(np.asarray(score), np.asarray(label, bool)) == expected


def test_auc_matches_explicit_pairs():
    rng = np.random.default_rng(38201)
    for _ in range(10):
        score = rng.integers(0, 4, size=31)
        label = rng.random(31) > .5
        pos, neg = score[label, None], score[~label][None, :]
        expected = np.mean((pos > neg) + .5 * (pos == neg))
        assert rank_auc(score, label) == pytest.approx(expected)


def test_unknown_is_not_safe_and_partial_is_not_complete():
    r = region_stats([1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0],
                     [3, -1, 100, np.nan], {"review": [1, 2, 3, 4]})
    assert (r["rows"], r["complete"], r["incomplete"], r["unknown"]) == (4, 2, 2, 1)
    assert r["costs"]["net_sum"] == 2
    assert r["costs"]["benefit_mean"] == 1.5
    assert r["costs"]["harm_mean"] == .5
    assert r["costs"]["beneficial"] == r["costs"]["harmful"] == 1


def test_empty_region_has_no_cost():
    r = region_stats([False], [True], [True], [2], {"review": [1]})
    assert r["costs"] is None and r["complete"] == 0


def test_invalid_diagnostic_inputs():
    with pytest.raises(ValueError):
        rank_auc([np.nan, 1], np.array([True, False]))
    with pytest.raises(ValueError):
        region_stats([1], [1], [0], [1], {})
    with pytest.raises(ValueError):
        region_stats([1], [1], [1], [np.nan], {})
