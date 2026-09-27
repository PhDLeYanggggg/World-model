import inspect
import numpy as np
import pytest
from src.world_model import m3w_cap_attribution as m
from src.world_model import m3w_temporal_support as temporal


def sample():
    p = np.tile([1., .5, .4, .1], (6, 1)); e = np.ones(6)*2
    q = np.array([-1., .2, .6, 1.5, 4., .8])
    y = p.copy(); y[:, 1] = [0, 1, 1, 1.5, 2, 1]; y[:, 3] = [0, 0, .3, 1.5, 2, 0]
    return q,p,e,y


def test_label_free_projection_and_nested_joint():
    q,p,e,_ = sample(); outputs = m.project(q,p,e)
    np.testing.assert_array_equal(outputs['frozen_cap'][:, 3], np.clip(q, 0, p[:, 1]))
    joint = outputs['envelope_coupled']
    assert np.all(joint[:, 3] <= joint[:, 1]) and np.all(joint[:, 1] <= e)
    np.testing.assert_array_equal(joint[:, [0, 2]], p[:, [0, 2]])
    assert 'target' not in inspect.signature(m.project).parameters
    assert 'target' not in inspect.signature(m.unprojected).parameters


def test_frozen_coefficients_reconstruct_exact_projection():
    q,p,e,y = sample(); x = np.arange(12).reshape(6, 2)/12
    sites = np.array(['a']*3+['b']*3)
    model = temporal.fit(x,p,y,e,sites,{'outer','inner'})
    score = m.unprojected(model,x,p,e)
    np.testing.assert_array_equal(m.project(score,p,e)['frozen_cap'], temporal.predict(model,x,p,e))


def test_decomposition_and_causal_envelope_dominance():
    q,p,e,y = sample(); d = m.diagnose(q,p,e,y)
    assert d['lower_projection_MSE_benefit'] > 0 and d['envelope_projection_MSE_benefit'] > 0
    assert d['cap_relaxation_improves_rows'] > 0 and d['cap_relaxation_worsens_rows'] > 0
    assert d['cap_relaxation_MSE_benefit'] == pytest.approx(d['cap_relaxation_helping_mass']-d['cap_relaxation_harming_mass'])
    assert d['cap_relaxation_MSE_benefit'] == pytest.approx(d['event_cap_relaxation_contribution']+d['zero_event_cap_relaxation_contribution'])
    assert d['uncoupled_order_violation_fraction'] > 0


def test_missing_analysis_labels_do_not_change_inference():
    q,p,e,y = sample(); before = m.project(q,p,e)
    y[0] = np.nan; assert m.diagnose(q,p,e,y)['rows'] == 5
    after = m.project(q,p,e)
    for key in before: np.testing.assert_array_equal(before[key], after[key])
    y[1, 0] = np.nan
    with pytest.raises(ValueError): m.diagnose(q,p,e,y)


@pytest.mark.parametrize('bad', ['negative', 'order', 'envelope'])
def test_invalid_targets_are_hard_errors(bad):
    q,p,e,y = sample()
    if bad == 'negative': y[0, 3] = -1
    elif bad == 'order': y[0, 3] = 1
    else: y[0, 1] = 3
    with pytest.raises(ValueError): m.diagnose(q,p,e,y)


def test_pointwise_floor_is_not_mean_bias():
    y = np.r_[10., np.zeros(19)]; p = np.full(20, .5)
    assert y.mean() == p.mean()
    assert np.mean(np.maximum(y-p, 0)**2)/np.mean((y-p)**2) == pytest.approx(.95)


def test_zero_support_is_not_success():
    q,p,e,y = sample(); y[:] = np.nan
    assert m.diagnose(q,p,e,y)['status'] == 'not_estimable'
    assert all(v['status'] == 'not_estimable' for v in m.evaluate(q,p,e,y).values())
