import numpy as np

from scripts.audit_m3w_opendd_exchange_phase import ORACLES, closure
from src.world_model.m3w_exchange_phase import phase_paths, prefix_choices


def test_phase_reorders_deviation_without_changing_endpoint_or_radius():
    cv = np.repeat(np.arange(12., dtype=float)[None, :, None], 2, axis=2)
    candidate = cv.copy()
    candidate[0, :, 0] += np.arange(1., 13.)
    p = phase_paths(cv, candidate)
    assert p['early'][0, 3, 0] > p['original'][0, 3, 0] > p['late'][0, 3, 0]
    for v in p.values():
        np.testing.assert_array_equal(v[:, -1], candidate[:, -1])
        np.testing.assert_allclose(np.linalg.norm(v-cv, axis=-1).max(1), [12.])
    np.testing.assert_array_equal(prefix_choices(p, p['early']), [1])


def test_phase_keeps_interior_peak_and_leaves_zero_residual_unchanged():
    rng = np.random.default_rng(27)
    cv = rng.normal(size=(2, 12, 2))
    y = cv.copy()
    y[0, 2] += [5., 1.]
    y[0, -1] -= [1., 2.]
    paths = phase_paths(cv, y)
    for x in paths.values():
        np.testing.assert_array_equal(x[:, 2], y[:, 2])
        np.testing.assert_array_equal(x[:, -1], y[:, -1])
        np.testing.assert_array_equal(x[1], cv[1])
        np.testing.assert_allclose(np.linalg.norm(x-cv, axis=-1).max(1), [np.sqrt(26), 0.])
    assert prefix_choices(paths, cv)[1] == 0


def test_expanded_oracle_adds_only_witnesses_that_pass_risk():
    fail = dict(vs_cv=8., vs_ols4=9., risk_pass=0, dual_positive=0)
    small = dict(vs_cv=.08, vs_ols4=.2, risk_pass=1, dual_positive=1)
    large = dict(vs_cv=.17, vs_ols4=.11, risk_pass=1, dual_positive=1)
    view = {k: set() for k in ORACLES}
    closure(view, dict(original=small, early=large, late=fail, prefix_ols=fail), fail,
            dict(original=False, early=True, late=False, prefix_ols=False))
    assert view['original'] == {'positive'}
    assert view['expanded'] == {'positive', 'gain_gt_0_10pp', 'stable_rescue'}
    assert not view['late']
