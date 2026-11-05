import numpy as np
import pytest
from src.evaluation import m3w_cost_partition_probe as m


def fixture():
    rng = np.random.default_rng(17); x = rng.normal(size=(240, 4))
    y = np.column_stack((np.where(x[:, 0] > 0, 3., -.5), np.where(x[:, 0] > 0, .5, -.1)))
    train = np.arange(len(x)) < 180; w = np.ones(len(x))/len(x)
    return x, y, train, w


def test_hurdle_identity_is_not_a_new_prediction():
    y = np.array([[0., -1.], [3., 2.], [-2., 0.], [0., 0.]])
    leaves = np.array([0, 0, 1, 2]); w = np.array([1., 3., 2., 1.])
    a = m.leaf_moments(leaves, y, w, 4)
    np.testing.assert_allclose(a['mean'], a['positive_probability']*a['positive_severity']-
        a['negative_probability']*a['negative_severity'])
    assert np.isfinite(a['positive_severity']).all() and a['mass'][3] == 0
    np.testing.assert_allclose(a['mean'][0], [2.25, 1.25])


@pytest.mark.parametrize('arm', m.ARMS[1:])
def test_only_fitting_source_controls_every_fitted_parameter(arm):
    x, y, train, w = fixture(); state = m.fit(x, y, train, w, arm)
    xx, yy = x.copy(), y.copy(); xx[~train] += 800; yy[~train] -= 500
    other = m.fit(xx, yy, train, w, arm)
    for k, v in m.fingerprint(state).items():
        np.testing.assert_array_equal(v, m.fingerprint(other)[k])
    audit = m.decomposition(state, x, y, train, w)
    assert audit['same_partition_factorization_max_abs_difference'] < 1e-12
    assert audit['direct_vs_factorized_joint_sign_changes'] == 0
    assert state['original_mean_readout'] and not state['probability_used_for_acceptance']
    assert np.isfinite(m.predict(state, x[~train])).all()


def test_readout_uses_original_measure_not_cost_weighted_measure():
    x, y, train, w = fixture(); state = m.fit(x, y, train, w, 'cost_partition')
    z = m.normalize(state, x)
    for tree, table in zip(state['model'].estimators_, state['tables']):
        leaves = tree.apply(z[train])
        for leaf in np.unique(leaves):
            use = leaves == leaf
            np.testing.assert_allclose(table['mean'][leaf], np.average(y[train][use], weights=w[train][use], axis=0), atol=1e-12)


@pytest.mark.parametrize('arm', m.ARMS[1:])
def test_separable_synthetic_signal_produces_both_decisions(arm):
    x, y, train, w = fixture(); x[:, 0] = np.sign(x[:, 0])
    state = m.fit(x, y, train, w, arm)
    score = m.predict(state, x[~train]).max(-1)
    assert (score <= 0).any() and (score > 0).any()
    assert np.array_equal(score > 0, y[~train].max(-1) > 0)


def test_no_positive_certificate_for_zero_cost_or_harm_tradeoff():
    def folds(a, b, c):
        return [dict(arms={arm:dict(metrics=dict(normalized_MSE=1., total_joint_cost=v,
            false_safe_cost=0., false_reject_cost=v)) for arm, v in zip(m.ARMS, (a,b,c))}) for _ in range(2)]
    assert not m.summary(folds(0, 0, 0))['passes_internal_screen']
    f = folds(1, .9, .8); assert m.summary(f)['passes_internal_screen']
    f[0]['arms']['cost_partition']['metrics']['false_safe_cost'] = .01
    assert not m.summary(f)['passes_internal_screen']


def test_invalid_supervision_is_rejected():
    x, y, train, w = fixture()
    with pytest.raises(ValueError):
        m.fit(x, y, train, w, 'unknown')
    with pytest.raises(ValueError):
        m.fit(x, y*0, train, w, 'cost_partition')
