import numpy as np
import pytest
from src.world_model import m3w_cost_mass as m


def fixture():
    p = np.tile([4., 1., 2., .3], (9, 1))
    y = p.copy(); y[:, 1] = 2.; y[:, 3] = 1.
    return p, y, np.full(9, 5.), np.array(['a']*3+['b']*6)


def test_projected_mass_exact_and_denominators_unchanged():
    p, y, env, sites = fixture()
    p[0, 1] = 4.; env[0] = 4.
    model = m.fit(p, y, env, sites, 'c')
    got = m.predict(model, p, env)
    w = m.fitting_weights(p, y, env, sites, 'c')
    np.testing.assert_allclose(w @ got[:, [1, 3]], w @ y[:, [1, 3]], atol=1e-12)
    np.testing.assert_array_equal(got[:, [0, 2]], p[:, [0, 2]])
    assert np.all(got[:, 3] <= got[:, 1]) and np.all(got[:, 1] <= env)
    assert model['mass_preserved']


def test_equal_locality_not_row_mean_and_replication_invariant():
    p, y, env, sites = fixture(); y[:3, 1] = 3.; y[3:, 1] = 1.
    model = m.fit(p, y, env, sites, 'c')
    assert model['components'][0]['target_mass'] == 2.
    ids = np.r_[np.arange(9), np.tile(np.arange(3, 9), 5)]
    again = m.fit(p[ids], y[ids], env[ids], sites[ids], 'c')
    np.testing.assert_allclose(model['slopes'], again['slopes'], atol=1e-12)


def test_unknown_and_zero_envelope_rows_do_not_fit():
    p, y, env, sites = fixture(); y[-1] = np.nan; env[-2] = 0.
    a = m.fit(p, y, env, sites, 'c'); p[-2:] = 1e12
    b = m.fit(p, y, env, sites, 'c')
    assert a == b
    assert m.fitting_weights(p, y, env, sites, 'c')[-2:].sum() == 0


def test_zero_target_and_infeasible_bound_are_not_silently_dropped():
    p, y, env, sites = fixture(); y[:, 3] = 0.
    a = m.fit(p, y, env, sites, 'c')
    assert a['slopes'][1] == 0. and a['components'][1]['status'] == 'zero_target'
    p[:, 1] = .01; p[:, 3] = .001; y[:, 3] = 1.
    b = m.fit(p, y, env, sites, 'c')
    assert b['slopes'] == [8., 8.] and not b['mass_preserved']
    assert all(x['status'] == 'infeasible_at_bound' for x in b['components'])


def test_nested_cap_is_used_while_fitting_easy_mass():
    p, y, env, sites = fixture(); p[:3, 1] = .1; p[:3, 3] = .09
    p[3:, 3] = .01; y[:, 1] = .5; y[:, 3] = .4
    a = m.fit(p, y, env, sites, 'c')
    w = m.fitting_weights(p, y, env, sites, 'c')
    got = m.predict(a, p, env)
    assert a['components'][1]['maximum_mass'] < float(w @ np.minimum(env, 8*p[:, 3]))
    assert np.isclose(w @ got[:, 3], a['components'][1]['achieved_mass'])


@pytest.mark.parametrize('bad', ['outer', 'partial_nan', 'infinity', 'missing_site', 'negative'])
def test_bad_or_leaking_inputs_rejected(bad):
    p, y, env, sites = fixture(); outer = 'c'
    if bad == 'outer': outer = 'a'
    elif bad == 'partial_nan': y[0, 0] = np.nan
    elif bad == 'infinity': y[0] = np.inf
    elif bad == 'missing_site': env[sites == 'b'] = 0.
    elif bad == 'negative': p[0, 1] = -1.
    with pytest.raises(ValueError): m.fit(p, y, env, sites, outer)


def test_decomposition_sums_and_detects_projection_mass_loss():
    p, y, env, sites = fixture(); y[:3, 3] = 0.
    p[:3, 3] = 1.; w = m.fitting_weights(p, y, env, sites, 'c')
    score = m.predict(dict(slopes=[.1, 2.], max_slope=8.), p, env)
    d = m.decompose(p, y, score, w, sites, dict(slopes=[.1, 2.], max_slope=8.))
    h = d['H_easy']
    assert np.isclose(h['MSE'], h['zero_target_SSE']+h['positive_target_SSE'])
    assert np.isclose(h['MSE'], sum(x['MSE_contribution'] for x in h['localities'].values()))
    assert h['projection_mass_loss'] > 0
    assert h['LS_denominator_zero_share'] > 0


def test_no_target_argument_at_inference_and_held_perturbation_has_no_fit_effect():
    p, y, env, sites = fixture(); held = np.ones((4, 4))*2.
    model = m.fit(p, y, env, sites, 'c'); before = m.predict(model, held, np.ones(4)*3)
    held_target = np.full((4, 4), 1e12); held_target[:] = -1e12
    np.testing.assert_array_equal(before, m.predict(model, held, np.ones(4)*3))
    assert m.fit(p, y, env, sites, 'c') == model
