import numpy as np

from src.world_model import m3w_full_query_retrieval as m


def test_pair_opportunity_needs_valid_easy_reference_but_can_credit_hard_partner():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 10., 4., 100.])[:, None]
    valid = np.ones((4, 12), bool)
    valid[3] = False
    gain = m.opportunity(cv, y[:, None], y, valid, np.ones(4), [np.arange(4)], 1.5, [2])
    assert gain[0, 1] == gain[1, 1] == 5.5
    assert gain[2, 1] == 2.5
    assert gain[3, 1] == 0.


def test_retrieval_is_budget_matched_and_uses_distinct_legal_recipients():
    score = np.column_stack((np.arange(12), -np.arange(12)))
    legal = np.ones((12, 2), bool)
    legal[0] = False
    a, b = m.lists(score, legal)
    assert len(a) == len(b) == len(set(b)) == m.SHORTLIST
    assert 0 not in b and 1 in b and 1 not in a
