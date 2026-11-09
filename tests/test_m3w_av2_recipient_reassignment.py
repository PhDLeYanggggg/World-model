import numpy as np

from scripts.run_m3w_av2_recipient_reassignment import eq, prev
from src.world_model.m3w_av2_recipient_reassignment import assignments, bank, eligibility, render, select


def test_reassign_preserves_profiles_and_does_not_edit_third_actor():
    cv = np.zeros((3, 12, 2))
    a = cv.copy()
    a[0, :, 0] = np.linspace(.1, 1.2, 12)
    a[1, :, 1] = np.linspace(.8, 2.1, 12)
    h = np.zeros((3, 8, 2))
    h[0, :, 0] = h[1, :, 1] = np.arange(8)
    h[2, :, 0] = -np.arange(8)
    p, slots = bank(cv, h, a)
    np.testing.assert_array_equal(render(cv, p, slots), a)
    out = render(cv, p, [2, 0])
    np.testing.assert_allclose(np.linalg.norm(out[[2, 0]], axis=-1), np.linalg.norm(a[slots], axis=-1))
    np.testing.assert_array_equal(out[1], cv[1])
    opts = assignments(3)
    choices, _ = select(h, cv, p, slots, opts, 1.)
    np.testing.assert_array_equal(np.sort(opts[choices[0]]), slots)
    assert (opts[choices, 0] != opts[choices, 1]).all()


def test_vector_feasibility_matches_original_scorer_with_missing_labels():
    cv = np.zeros((3, 12, 2))
    a, ols = cv.copy(), cv.copy()
    a[0, :, 0], a[1, :, 1] = .5, .8
    ols[..., 0] = -1.
    y = a.copy()
    y[2, :, 1] = 1.2
    v = np.ones((3, 12), dtype=bool)
    v[1, -3:] = False
    p, _ = bank(cv, np.zeros((3, 8, 2)), a)
    opts = assignments(3)
    mask, _ = eligibility(cv, ols, p, y, v, 1., opts)
    score = [eq.ceiling.eligible_additive(prev.metrics(render(cv, p, a), dict(cv=cv, ols4=ols), y, v, [1., 2.])) for a in opts]
    np.testing.assert_array_equal(mask, score)
    assert mask[0] and not mask.all()
