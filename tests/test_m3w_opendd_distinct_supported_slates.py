import itertools

import numpy as np

from scripts import audit_m3w_opendd_distinct_supported_slates as audit


def test_original_scoring_terms_reproduce_gain_and_risk():
    y = np.zeros((3, 12, 2))
    cv = y.copy()
    cv[..., 0] = np.array([1., 2., 3.])[:, None]
    p = cv.copy()
    p[:2] *= .5
    valid = np.ones(y.shape[:2], bool)
    valid[1, -1] = False
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()/len(cv)
    m = audit.terms(p[None], cv, cv, cv, y, valid, 1.5, 1./3., mass, 1)[0]
    r = audit.study.policy(p, cv, cv, y, valid, ["r"]*len(y), easy_cut=1.5, hard_cut=2.5)["recordings"]["r"]
    assert r["supported"]
    assert m[-1] == 1
    assert m[4] < 0 and m[5] < 0 and m[6] < 0
    expected = 100.*audit.lower(cv, p, y, valid).sum()/len(cv)/mass
    np.testing.assert_allclose(m[1:4], expected)


def test_integer_witness_and_distinct_second_match_brute_force():
    values = np.array([
        [8., 1., 1., 0., 1., 0., 0., 1.],
        [7., 1., 1., 0., -1., 0., 0., 0.],
        [5., 1., 1., 0., 0., 0., 0., 1.],
        [3., 1., 1., 0., -1., 0., 0., 0.],
        [2., 1., 1., 0., 0., 0., 0., 0.],
    ])
    d = dict(values=values, fixed=np.zeros(8), cap=.01, locals=[dict(count=3), dict(count=2)])
    ranks = sorted([(values[list(pair), 0].sum(), pair) for pair in itertools.product(range(3), [3, 4])
                    if values[list(pair), 4].sum() <= 0. and values[list(pair), -1].sum() >= 1.], reverse=True)
    first, r = audit.solve(d)
    assert r["status"] == "optimal"
    assert tuple(first) == ranks[0][1]
    equivalent = np.isin(np.arange(len(values)), first)
    second, r = audit.solve(d, forbidden=equivalent)
    assert r["status"] == "optimal"
    assert tuple(second) == ranks[1][1]
    _, r = audit.solve(d, forbidden=np.ones(len(values), dtype=bool))
    assert r["status"] == "infeasible"


def test_exclusion_marks_duplicate_candidates_and_reversed_equal_slots():
    cv = np.zeros((3, 12, 2))
    bank = np.ones((3, 2, 12, 2))
    loc = audit.local_choices(cv, cv, bank, np.ones(2))
    loc["ids"] = np.arange(len(cv))
    chosen = audit.render(loc, np.array([0]))[0]
    mask = audit.equivalent(dict(locals=[loc]), chosen)
    assert mask.sum() == 8
    assert (~mask).any()
    np.testing.assert_array_equal(audit.render(loc, np.flatnonzero(mask)),
                                  np.broadcast_to(chosen, (mask.sum(), *chosen.shape)))
