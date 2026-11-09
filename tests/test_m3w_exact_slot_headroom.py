import itertools

import numpy as np

from src.evaluation.m3w_exact_slot_headroom import make_design, solve
from src.evaluation.m3w_selection_forecast_readout import policy


def setup():
    cv = np.zeros((3, 2, 2))
    bank = np.repeat(cv[:, None], 2, axis=1)
    bank[:, 0, :, 0], bank[:, 1, :, 0] = .5, -.5
    y = cv.copy()
    y[:, :, 0] = np.array([.6, -.7, 2.])[:, None]
    return cv, bank, y, np.ones((3, 2), bool), np.zeros(3, dtype=int)


def test_optimum_reuses_exact_slots_and_passes_the_unmodified_risk_scorer():
    cv, bank, y, v, q = setup()
    control, orig = cv.copy(), cv.copy()
    control[0], orig[1] = bank[0, 1], bank[1, 1]
    d, none = make_design(cv, cv, bank, control, orig, y, v, q, easy_cut=1.)
    assert none is None
    take, log = solve(d, 2.)
    assert log['status'] == 'optimal'
    assert log['upper_pp'] > 0 and take.sum() == 1
    p = cv.copy()
    p[d['row'][take]] = d['path'][take]
    assert policy(p, cv, cv, y, v, ['r']*len(y), easy_cut=1., hard_cut=1.5)['recordings']['r']['supported']


def test_relaxed_bound_matches_brute_force_with_two_distinct_recipients():
    cv, bank, y, v, q = setup()
    v[2] = False
    control = cv.copy()
    control[0], control[1] = bank[0, 1]*.4, bank[1, 0]
    d, _ = make_design(cv, cv, bank, control, control, y, v, q, easy_cut=1.)
    candidates = [(a, b) for a, b in itertools.product(*(np.flatnonzero(d['slot'] == i) for i in range(2)))
                  if d['row'][a] != d['row'][b]]
    brute = np.array([d['gain'][list(ab)].sum(0)+d['offset'] for ab in candidates]).max(0)
    np.testing.assert_allclose(d['upper'], brute)
    take, log = solve(d, 2.)
    assert log['status'] == 'optimal'
    assert len(set(d['row'][take])) == 2
    assert (d['row'][take] != 2).all()


def test_zero_edits_and_no_easy_reference_are_not_passes():
    cv, bank, y, v, q = setup()
    assert make_design(cv, cv, bank, cv, cv, y, v, q, easy_cut=1.) == (None, 'no_edits')
    edited = cv.copy()
    edited[0] = bank[0, 0]
    d, _ = make_design(cv, cv, bank, edited, edited, y, v, q, easy_cut=.1)
    take, log = solve(d, 2.)
    assert take is None and log['status'] == 'no_easy_support'


def test_fixing_singletons_cannot_credit_pair_score_for_a_unary_fix():
    cv, bank, y, v, q = setup()
    control = cv.copy()
    control[0] = bank[0, 1]
    all_edits, _ = make_design(cv, cv, bank, control, control, y, v, q, easy_cut=1.)
    assert solve(all_edits, 2.)[1]['status'] == 'optimal'
    restricted, _ = make_design(cv, cv, bank, control, control, y, v, q, easy_cut=1., double_only=True)
    assert restricted['enabled'].sum() == 1
    np.testing.assert_allclose(restricted['upper'][2], 0., atol=1e-12)
    assert solve(restricted, 2.)[1]['status'] == 'nonpositive_reference_bound'
