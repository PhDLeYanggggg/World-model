import numpy as np
import pytest
import torch

from src.world_model.m3w_native_gain_harm import (
    cost_features, preprocess, cost_loss, fit_neural, predict_neural, fit_ridge,
)
from src.evaluation.m3w_native_cost_readout import fixed_interventions, conditional_report


def example(n=24):
    g = np.zeros((n, 476), np.float32)
    g[:, 16:24] = np.arange(-7, 1)/12
    g[:, :16] = np.tile(np.column_stack((np.arange(-7, 1)/12, np.zeros(8))).ravel(), (n, 1))
    g[:, 332:356] = np.tile(np.column_stack((np.arange(1, 13)/12, np.zeros(12))).ravel(), (n, 1))
    p = g[:, 332:356].reshape(n, 12, 2).copy()*1.1
    p[0] = g[0, 332:356].reshape(12, 2)
    x, same = cost_features(g, p, np.ones(n))
    y = np.column_stack((np.arange(n)%3, np.arange(n)%5)).astype(float)
    y[0] = 0; y[1] = np.nan
    cv = np.ones(n)*3; cv[1] = np.nan
    sites = np.repeat(['a', 'b', 'c'], n//3)
    return x, y, cv, sites, same, g, p


def test_features_only_use_causal_payload_and_append_native_scale():
    x, _, _, _, same, g, p = example()
    assert x.shape == (24, 355) and same.sum() == 1
    # Unused geometry columns cannot smuggle metadata into the head.
    g[:, 356:] = 1e9
    other, _ = cost_features(g, p, np.ones(len(g)))
    np.testing.assert_array_equal(x, other)
    scaled, _ = cost_features(g, p, np.full(len(g), 2.))
    np.testing.assert_array_equal(x[:, :-1], scaled[:, :-1])
    np.testing.assert_allclose(scaled[:, -1], np.log(2))
    g[:, 16] = 1
    with pytest.raises(ValueError):
        cost_features(g, p, np.ones(len(g)))


def test_outer_exposure_and_unknown_supervision_rejected_or_excluded():
    x, y, cv, sites, _, _, _ = example()
    with pytest.raises(ValueError):
        preprocess(x, y, cv, sites, 'a')
    pr = preprocess(x, y, cv, sites, 'held')
    assert pr['weights'][1] == 0 and pr['known'].sum() == 23
    assert np.isnan(y[1]).all()
    for s in set(sites):
        assert pr['weights'][sites == s].sum() == pytest.approx(1/3)
    altered = x.copy(); altered[1] = 1e9
    second = preprocess(altered, y, cv, sites, 'held')
    np.testing.assert_array_equal(pr['mean'], second['mean'])
    bad = y.copy(); bad[1, 0] = 0
    with pytest.raises(ValueError):
        preprocess(x, bad, cv, sites, 'held')


def test_asymmetric_loss_penalizes_under_harm_not_benefit():
    prediction = torch.tensor([[1., 1.]], requires_grad=True)
    target = torch.tensor([[2., 2.]])
    ordinary = cost_loss(prediction, target, 'mse')
    ordinary.backward(); first = prediction.grad.clone(); prediction.grad.zero_()
    cost_loss(prediction, target, 'underharm4').backward()
    assert prediction.grad[0, 0] == first[0, 0]
    assert prediction.grad[0, 1] == first[0, 1]*4
    assert cost_loss(torch.tensor([[3., 3.]]), target, 'mse') == cost_loss(torch.tensor([[3., 3.]]), target, 'underharm4')


def test_exact_neural_resume_and_structural_zero(tmp_path):
    torch.set_num_threads(2)
    x, y, cv, sites, same, _, _ = example()
    pr = preprocess(x, y, cv, sites, 'held')
    settings = dict(steps=6, batch_size=8, learning_rate=.001, width=8,
                    checkpoint_every=2, heartbeat_every=1, gradient_clip=5.)
    def run(path, **kw):
        return fit_neural(x, y, sites, same, pr, seed=17, arm='mse', settings=settings,
            identity={'temporary':True}, directory=path, heartbeat=lambda **k:None, **kw)
    full, _ = run(tmp_path/'full')
    run(tmp_path/'split', stop_at=3)
    resumed, result = run(tmp_path/'split', resume=True)
    for k in full.state_dict():
        torch.testing.assert_close(full.state_dict()[k], resumed.state_dict()[k], atol=0, rtol=0)
    assert result['step'] == 6 and result['new_updates'] == 3 and result['unknown_rows_sampled'] == 0
    out = predict_neural(resumed, x, same, pr)
    assert not out[0].any() and np.isfinite(out).all()
    with pytest.raises(ValueError):
        fit_neural(x, y, sites, same, pr, seed=17, arm='mse', settings=settings,
            identity={'changed':True}, directory=tmp_path/'split', resume=True, heartbeat=lambda **k:None)


def test_ridge_keeps_signed_cost_predictions_available():
    x, y, cv, sites, _, _, _ = example()
    pr = preprocess(x, y, cv, sites, 'held')
    head = fit_ridge(x, y, pr, alpha=.01)
    assert head['coef'].shape == (356, 2)
    assert np.isfinite(head['coef']).all()


def test_unknown_selected_outcome_stays_unknown_in_conditional_readout():
    p = np.array([[2., .1], [2., .1], [0., 0.]])
    same = np.array([False, False, True])
    use = fixed_interventions(p, same)['harm_fraction_0p1']
    np.testing.assert_array_equal(use, [True, True, False])
    r = conditional_report(p, np.array([0., np.nan, 0.]), np.array([3., np.nan, 0.]), same)
    g = r['groups']['harm_fraction_0p1']
    assert g['unknown_rows'] == 1 and g['realized_harm'] == 3.
    assert g['realized_gain'] == -3. and not r['calibrated_probability']


def test_negative_ridge_harm_is_reported_not_silently_zeroed():
    p = np.array([[0., -1.], [2., .1]])
    same = np.zeros(2, bool)
    raw = conditional_report(p, np.zeros(2), np.ones(2), same)
    clipped = conditional_report(np.maximum(p, 0), np.zeros(2), np.ones(2), same)
    assert raw['negative_harm_predictions'] == 1
    assert raw['groups']['positive_gain']['supported_rows'] == 2
    assert clipped['groups']['positive_gain']['supported_rows'] == 1


def test_unknown_cost_does_not_become_zero_in_empty_readout():
    p = np.ones((3, 2)); y = np.full(3, np.nan)
    r = conditional_report(p, y, y, np.zeros(3, bool))
    assert r['harm_mse'] is None and r['unknown_rows'] == 3
    with pytest.raises(ValueError):
        conditional_report(p, y, np.zeros(3), np.zeros(3, bool))
