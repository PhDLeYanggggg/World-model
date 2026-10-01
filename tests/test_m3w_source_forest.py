import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_source_forest as api


def sample():
    rng = np.random.default_rng(51); n = 80
    x = rng.normal(size=(n, 4)); env = rng.uniform(.5, 2, n)
    b = env * rng.uniform(0, .3, n); h = env * rng.uniform(0, .4, n)
    r = b + rng.uniform(.5, 2, n); easy = rng.random(n) > .5
    y = np.column_stack((b, h, r, r * easy, h * easy)); y[-3:] = np.nan
    sites = np.repeat('dev', n); rec = np.repeat('record', n); frame = np.arange(n) // 4
    pr = api.core.preprocess(x, env, y, sites, rec, frame, training_site='dev')
    return x, env, y, sites, rec, frame, pr


def test_objective_equivalence():
    _, _, y, _, _, _, pr = sample(); y = y[:-3]; p = y * .73
    expected = api.core.losses(torch.tensor(p / pr['scale']), torch.tensor(y / pr['scale']),
        torch.arange(len(y)), len(y), torch.tensor(pr['rms']))['total'].item()
    actual = ((api.transformed_targets(p, pr) - api.transformed_targets(y, pr))**2).mean()
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_causal_projection():
    p = np.array([[4., 3., 2., 6., 5.], [0., 0., 0., 0., 0.]])
    r = api.project_moments(p, np.array([1., 0.]))
    assert (r >= 0).all() and (r[:, :2].sum(1) <= [1., 0.]).all()
    assert (r[:, 0] <= r[:, 2]).all() and (r[:, 3] <= r[:, 2]).all()
    assert (r[:, 4] <= r[:, 1]).all()
    np.testing.assert_array_equal(p[0], [4, 3, 2, 6, 5])


def test_resume_and_exact_determinism(tmp_path):
    x, env, y, sites, rec, frames, pr = sample()
    settings = dict(trees=8, checkpoint_every=2, max_depth=4, min_samples_leaf=3,
                    max_features=1/3, fit_threads=2)
    kw = dict(settings=settings, seed=17, identity={'source': 'dev'}, heartbeat=lambda **k: None)
    full = api.fit(x, env, y, sites, rec, frames, pr, path=tmp_path/'full.joblib', **kw)
    api.fit(x, env, y, sites, rec, frames, pr, path=tmp_path/'resumed.joblib', stop_at=4, **kw)
    resumed = api.fit(x, env, y, sites, rec, frames, pr, path=tmp_path/'resumed.joblib', resume=True, **kw)
    api.exact_forest(full, resumed)
    p, support = api.predict(full, x, env)
    np.testing.assert_array_equal(p, api.predict(resumed, x, env)[0])
    assert p.shape == (len(x), 5) and support.dtype == bool
    assert full['unknown_training_rows'] == 3
    with pytest.raises(ValueError, match='resume'):
        api.fit(x, env, y, sites, rec, frames, pr, path=tmp_path/'full.joblib', **kw)
    changed = x.copy(); changed[0, 0] += 1
    with pytest.raises((ValueError, AssertionError)):
        api.fit(changed, env, y, sites, rec, frames, pr, path=tmp_path/'resumed.joblib', resume=True, **kw)


def test_unknown_labels_not_fitting_targets():
    _, _, y, _, _, _, pr = sample()
    with pytest.raises(ValueError, match='known'):
        api.transformed_targets(y, pr)


def test_train_only_preprocessing_rejected_when_changed(tmp_path):
    x, env, y, sites, rec, frames, pr = sample(); pr = copy.deepcopy(pr); pr['mean'][0] += 10
    with pytest.raises(AssertionError):
        api.fit(x, env, y, sites, rec, frames, pr, settings={}, seed=1,
                identity={}, path=tmp_path/'no.joblib', heartbeat=lambda **k: None)


def test_inference_does_not_accept_outcomes():
    import inspect
    assert list(inspect.signature(api.predict).parameters) == ['state', 'x', 'envelope']
    x, env, _, _, _, _, pr = sample()
    z, support = api.causal_inputs(x, env, pr)
    assert z.shape[1] == x.shape[1]+1
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='past-only'):
        api.causal_inputs(x, env, pr)
