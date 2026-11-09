import numpy as np

from scripts.audit_m3w_opendd_self_menu import ARMS, recording, scaled_to_path


def test_scaled_family_match_does_not_credit_fallback_or_paths_larger_than_source():
    cv = np.zeros((4, 2, 2))
    candidate = cv.copy()
    candidate[:, :, 0] = 1.
    path = candidate*np.array([.5, 0., 1.1, -.5])[:, None, None]
    np.testing.assert_array_equal(scaled_to_path(cv, candidate, path), [True, False, False, False])


def test_scoring_uses_exact_same_budget_and_conservative_pair_effect():
    cv = np.zeros((3, 2, 2))
    y = cv.copy()
    y[:, :, 0] = 1.
    y[0, :, 0] = .3
    valid = np.array([[True, False], [True, True], [True, True]])
    known = cv.copy()
    known[1, :, 0] = .5
    partial = cv.copy()
    partial[0, :, 0] = .5
    paths = dict(cv=cv, ols4=cv, **{k: known if 'expanded' in k else partial for k in ARMS})
    counts, contrasts = recording(paths, y, valid, np.zeros(3, dtype=int), (1., 2.))
    assert all(r['edits'] == 1 and r['risk'] for r in counts.values())
    np.testing.assert_allclose(contrasts['expanded_joint_vs_base_joint'], 100.*.4/4.3)
    assert contrasts['expanded_joint_vs_expanded_additive'] == 0.
