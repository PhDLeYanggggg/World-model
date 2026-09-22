import joblib
import numpy as np
import pytest

from src.world_model.m3w_forest_cost_head import fit_targets, fit, predict, make_model, selections


def toy():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(120, 5)).astype(np.float32)
    d = np.abs(x[:, 0]) + .1
    y = np.column_stack((np.where(x[:, 1] > 0, .7*d, 0), np.where(x[:, 1] <= 0, .2*d, 0)))
    known = np.ones(120, bool); known[-2:] = False; y[~known] = np.nan
    pr = dict(mean=x[known].mean(0), std=x[known].std(0), known=known, cost_scale=2.)
    draws = known.astype(np.int64)*3; weights = known.astype(float)
    settings = dict(trees=8, checkpoint_every=2, max_depth=4, min_samples_leaf=3,
                    max_features=1/3, fit_threads=2)
    return x, y, d, pr, draws, weights, settings


def test_weight_matches_realized_exposures_not_held_labels():
    x, y, d, pr, draws, w, _ = toy()
    q, weight = fit_targets(y, d, pr['known'], draws, w, pr['cost_scale'])
    expected = draws*w*d/pr['cost_scale']; expected /= expected[expected > 0].mean()
    np.testing.assert_array_equal(weight, expected)
    np.testing.assert_allclose(q[pr['known']]*d[pr['known'], None], y[pr['known']])
    assert not weight[~pr['known']].any() and np.isfinite(q).all()


def test_unknown_draws_and_bound_violation_rejected():
    _, y, d, pr, draws, w, _ = toy()
    draws[-1] = 1
    with pytest.raises(ValueError): fit_targets(y, d, pr['known'], draws, w, 2.)
    draws[-1] = 0; y[0] = d[0]
    with pytest.raises(ValueError): fit_targets(y, d, pr['known'], draws, w, 2.)


def test_zero_distance_not_imputed_harm():
    _, y, d, pr, draws, w, _ = toy()
    d[0] = 0
    with pytest.raises(ValueError): fit_targets(y, d, pr['known'], draws, w, 2.)
    y[0] = 0
    q, weight = fit_targets(y, d, pr['known'], draws, w, 2.)
    assert weight[0] == 0 and q[0].sum() == 0


def test_real_forest_resume_exact_and_simplex(tmp_path):
    x, y, d, pr, draws, w, settings = toy()
    kw = dict(settings=settings, seed=17, identity={'test': 1}, heartbeat=lambda **_: None)
    fit(x, y, d, pr, draws, w, directory=tmp_path/'resume', stop_at=2, **kw)
    resumed, r = fit(x, y, d, pr, draws, w, directory=tmp_path/'resume', resume=True, **kw)
    direct, direct_r = fit(x, y, d, pr, draws, w, directory=tmp_path/'direct', **kw)
    a, b = predict(resumed, x, d, pr), predict(direct, x, d, pr)
    np.testing.assert_array_equal(a, b)
    assert r['complete'] and direct_r['complete'] and (a.sum(1) <= d+1e-12).all()
    cp = joblib.load(tmp_path/'resume'/'checkpoint.joblib')
    assert cp['model'].n_jobs == 1
    with pytest.raises(ValueError):
        fit(x, y, d, pr, draws, w, directory=tmp_path/'resume', **kw)
    with pytest.raises(ValueError):
        fit(x, y, d, pr, draws, w, directory=tmp_path/'resume', resume=True,
            **dict(kw, identity={'test': 2}))


def test_same_count_stable_ties_and_past_stop_guard():
    past = np.zeros((4, 8, 2)); past[:3, -1, 0] = 1
    f = np.array([[10, .1], [1, 3], [2, .1], [20, 0.]])
    n = np.array([[4., 0], [4, 0], [4, 0], [4, 0]])
    result = selections(f, n, past, np.ones(4), np.array([3, 2, 1, 0]))
    np.testing.assert_array_equal(result['forest'], [True, False, True, False])
    np.testing.assert_array_equal(result['neural_matched'], [False, True, True, False])
    assert result['forest'].sum() == result['neural_matched'].sum()


def test_zero_causal_disagreement_prediction_stays_zero(tmp_path):
    x, y, d, pr, draws, w, settings = toy()
    m, _ = fit(x, y, d, pr, draws, w, settings=settings, seed=17, identity={},
               directory=tmp_path, heartbeat=lambda **_: None)
    out = predict(m, x, np.zeros(len(d)), pr)
    np.testing.assert_array_equal(out, np.zeros_like(out))
