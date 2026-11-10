import numpy as np

from scripts.audit_m3w_opendd_exchange_stability import parent, retained_queries, stress


def test_deletion_always_protects_exchange_endpoints():
    masks = retained_queries(9, 1, 7)
    assert len(masks) == 3
    assert all(m[1] and m[7] for m in masks)
    assert [np.flatnonzero(~m).tolist() for m in masks] == [[0, 2], [3, 4, 5], [6, 8]]
    assert len(retained_queries(3, 0, 2)) == 1


def test_losing_unchanged_easy_support_is_not_a_pass():
    query = np.arange(9)
    y = np.zeros((len(query), 12, 2))
    cv = np.full_like(y, 4.)
    cv[1] = .5
    ols = cv+1.
    anchor, swap = cv.copy(), cv.copy()
    anchor[[0, 1]] *= .5
    swap[[1, 8]] *= .5
    d = dict(y=y, valid=np.ones(y.shape[:2], dtype=bool), rec=np.repeat('recording', len(y)))
    cut = [1., 4.]
    full = parent.score(d, cv, ols, swap, cut)
    outcome, causes = stress(d, cv, ols, anchor, swap, cut, full, query, len(query), 0, 8)
    assert outcome['full_rescue'] and outcome['three_blocks_supported']
    assert not outcome['stable_risk'] and not outcome['stable_dual']
    assert causes == {'unknown_risk': 1, 'dual_pass': 2}


def test_full_witness_with_protected_easy_reference_survives():
    query = np.arange(9)
    y = np.zeros((len(query), 12, 2))
    cv = np.full_like(y, .5)
    ols = cv+1.
    anchor, swap = cv.copy(), cv.copy()
    anchor[0] *= .5
    swap[8] *= .5
    d = dict(y=y, valid=np.ones(y.shape[:2], dtype=bool), rec=np.repeat('recording', len(y)))
    cut = [1., 4.]
    full = parent.score(d, cv, ols, swap, cut)
    outcome, causes = stress(d, cv, ols, anchor, swap, cut, full, query, len(query), 0, 8)
    assert outcome['stable_dual'] and outcome['stable_material_and_slack']
    assert not outcome['two_easy_support_queries']
    assert causes == {'dual_pass': 3}
