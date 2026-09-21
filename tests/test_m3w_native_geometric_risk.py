import numpy as np
import pytest
import torch
from src.world_model.m3w_native_geometric_risk import extend_features, zero_targets, build_head, objective, fit, predict
from src.world_model.m3w_native_gain_harm import preprocess


def test_history_features_and_control_have_same_shape():
    g = np.zeros((2, 476)); g[:, 16:24] = np.arange(-7, 1)/12
    g[:, :16] = np.tile(np.column_stack((np.arange(8), np.zeros(8))).ravel(), (2, 1))
    g[1, 2] += .3
    a = extend_features(np.zeros((2, 357)), g, 'kinematic_event')
    b = extend_features(np.zeros((2, 357)), g, 'base_event')
    assert a.shape == b.shape == (2, 361)
    np.testing.assert_allclose(a[0, -4:], 0, atol=1e-12)
    assert np.all(a[1, -4:] > 0) and np.all(b[:, -4:] == 0)
    g[:, 356:] = 999
    np.testing.assert_array_equal(a, extend_features(np.zeros((2, 357)), g, 'kinematic_event'))


def test_zero_targets_include_candidate_agreement_and_keep_unknown():
    np.testing.assert_equal(zero_targets(np.array([0., 1e-15, 0.]), np.array([True, True, False])), [1., 0., np.nan])


def test_geometric_loss_respects_known_distance():
    z, y, d = torch.zeros(3), torch.tensor([0., 1., 1.]), torch.tensor([0., 2., 4.])
    a, bce, mse = objective(z, y, d, 2., 'base_event')
    b, _, _ = objective(z, y, d, 2., 'base_geometric')
    torch.testing.assert_close(a, bce); torch.testing.assert_close(b, bce+mse)
    with pytest.raises(ValueError):
        objective(z, y, d, 2., 'unregistered')


def test_exact_resume_and_matched_draws(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 361)).astype(np.float32)
    sites = np.repeat(['a', 'b'], 15); d = np.ones(30)
    y = (np.arange(30) % 3 == 0).astype(float); y[4] = np.nan
    cv = np.where(np.isfinite(y), 1-y, np.nan)
    pr = preprocess(x, np.column_stack((y, y*d)), cv, sites, 'held')
    settings = dict(steps=6, batch_size=8, learning_rate=.001, checkpoint_every=2, heartbeat_every=2)
    args = dict(arm='base_geometric', seed=17, settings=settings, identity={'fixed':1}, heartbeat=lambda **_:None)
    fit(x, y, d, sites, pr, directory=tmp_path/'full', **args)
    fit(x, y, d, sites, pr, directory=tmp_path/'resume', stop_at=2, **args)
    r = fit(x, y, d, sites, pr, directory=tmp_path/'resume', resume=True, **args)
    predictions, draws = [], []
    for name in ('full', 'resume'):
        cp = torch.load(tmp_path/name/'checkpoint.pt', weights_only=False)
        model = build_head(17); model.load_state_dict(cp['model'])
        predictions.append(predict(model, x, pr)); draws.append(cp['draws'])
    np.testing.assert_array_equal(*predictions); np.testing.assert_array_equal(*draws)
    assert r['unknown_rows_sampled'] == 0 and r['total_draws'] == 48
