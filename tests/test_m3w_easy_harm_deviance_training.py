import numpy as np
import pytest
import torch

from src.world_model import m3w_easy_harm_deviance_training as api


def data():
    rng = np.random.default_rng(41); x = rng.normal(size=(24, 4)); env = np.full(24, 3.)
    y = np.column_stack((.4+.1*np.maximum(x[:, 0], 0), .3+.1*np.maximum(x[:, 1], 0),
                         1.+.1*np.abs(x[:, 2]), np.full(24, .4), np.full(24, .1)))
    series = np.stack([np.tile([r[1]-r[0], r[2]], (12, 1)) for r in y]); y[-1] = np.nan; series[-1] = np.nan
    sites = np.repeat('site', 24); rec = np.repeat([1, 2, 3], 8); frames = np.tile(np.repeat([0, 1], 4), 3)
    pr = api.core.preprocess(x, env, y, sites, rec, frames, training_site='site')
    return x, env, y, series, sites, rec, frames, pr


def settings():
    return dict(width=6, steps=8, learning_rate=.001, query_batch_size=2,
                gradient_clip=5., checkpoint_every=4, heartbeat_every=4, auxiliary_weight=.1)


def fit(tmp_path, arm, name, **kw):
    return api.fit(*data(), arm=arm, settings=settings(), seed=17, identity={'test': True},
        experiment_sha256='test', path=tmp_path/(name+'.pt.gz'), heartbeat=lambda **_: None, **kw)


def test_original_control_replayed_exactly(tmp_path):
    old = api.parent.fit(*data(), arm='none', settings=settings(), seed=17, identity={'test': True},
        path=tmp_path/'old.pt.gz', heartbeat=lambda **_: None)
    new = fit(tmp_path, 'quadratic', 'new'); api.assert_original_control(new, old)


@pytest.mark.parametrize('arm', api.ARMS)
def test_resume_is_exact(tmp_path, arm):
    full = fit(tmp_path, arm, 'full'); fit(tmp_path, arm, 'resume', stop_at=3)
    resumed = fit(tmp_path, arm, 'resume', resume=True)
    for key in full:
        if key != 'seconds': api.core.exact(full[key], resumed[key])


def test_shared_initialization_draws_and_changed_model(tmp_path):
    q = fit(tmp_path, 'quadratic', 'q'); d = fit(tmp_path, 'easy_deviance', 'd')
    api.assert_matched(q, d)
    assert not torch.equal(q['model']['primary.output.weight'], d['model']['primary.output.weight'])
    assert d['unknown_rows_sampled'] == 0 and not d['primary_objective_unchanged']
    probe = api.gradient_probe(d, data()); assert probe['positive_easy_harm_rows'] > 0
    assert probe['deviance_logit_gradient_norm'] > 0


def test_resume_rejects_changed_objective(tmp_path):
    fit(tmp_path, 'quadratic', 'a', stop_at=3)
    with pytest.raises(AssertionError):
        fit(tmp_path, 'easy_deviance', 'a', resume=True)


def test_storage_guard_precedes_forward(tmp_path):
    def block(): raise OSError('reserve')
    with pytest.raises(OSError): fit(tmp_path, 'easy_deviance', 'a', checkpoint_guard=block)
    assert not (tmp_path/'a.pt.gz').exists()
