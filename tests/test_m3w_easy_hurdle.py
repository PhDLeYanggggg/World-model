import numpy as np
import pytest
import torch
from src.world_model import m3w_easy_hurdle as api


def example():
    rng = np.random.default_rng(42); n = 32
    x = rng.normal(size=(n, 4)).astype(np.float32); u = rng.normal(size=(n, 6)).astype(np.float32)
    cv = np.tile([0., .1, .3, np.nan], n//4)
    y = api.targets(cv, cv, cv+.01, .2, 1.)
    known = np.isfinite(cv); w = known/known.sum()
    pr = dict(known=known, weights=w, mean=np.zeros(4), std=np.ones(4), clip=5., cost_scale=1., training_sites=['a', 'b'])
    return x, u, y, np.repeat(['a', 'b'], n//2), np.repeat('r', n), np.arange(n)//2, np.ones(n), pr


def test_label_mask_and_structural_zero():
    y = api.targets([0, .1, 2, np.nan], [0, .2, 3, np.nan], [0, .3, 2, np.nan], .5, 2)
    np.testing.assert_array_equal(y[:3, 0], [0, 1, 0])
    assert np.isnan(y[-1]).all()
    assert y[1, 2] == pytest.approx(.05)
    with pytest.raises(ValueError):
        api.targets([1, np.nan], [1, 2], [1, np.nan], .5, 1)


def test_factorization_and_bounds():
    model = api.initialize(10, 8, 17)
    p = model(torch.zeros(4, 10), torch.tensor([0., 1., 2., 3.]))
    assert ((p[:, 0] > 0) & (p[:, 0] < 1)).all()
    assert (p[:, 2] <= torch.tensor([0., 1., 2., 3.])).all()
    torch.testing.assert_close(p[:, 3], p[:, 0]*(p[:, 2]-.02*p[:, 1]))


def test_objective_has_explicit_probability_and_conditional_supervision():
    p = api.initialize(10, 8, 17)(torch.zeros(4, 10), torch.ones(4))
    y = torch.tensor([[0., 1., .3], [1., .2, .1], [0., 3., .2], [1., .4, .3]])
    v = api.losses(p, y, torch.tensor([0, 0, 1, 1]), 2)
    torch.testing.assert_close(v['supervised'], v['marginal']+v['occurrence']+v['conditional'])
    torch.testing.assert_close(v['occurrence'], torch.nn.functional.binary_cross_entropy(p[:, 0], y[:, 0]))
    assert all(torch.isfinite(x) for x in v.values())
    y[0] = float('nan')
    with pytest.raises(ValueError):
        api.losses(p, y, torch.tensor([0, 0, 1, 1]), 2)


def test_zero_easy_batch_has_no_fake_conditional_cost():
    p = api.initialize(10, 8, 17)(torch.zeros(4, 10), torch.ones(4))
    y = torch.tensor([[0., 1., .3]]*4)
    assert api.losses(p, y, torch.tensor([0, 0, 1, 1]), 2)['conditional'] == 0


def test_matched_resume_exact_and_unknown_exclusion(tmp_path):
    torch.set_num_threads(1)
    settings = dict(width=8, steps=6, query_batch_size=4, learning_rate=.001,
                    gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    args = example(); kw = dict(seed=17, settings=settings, identity={'fit': 'synthetic'}, heartbeat=lambda **kw: None)
    full = api.fit(*args, arm='supervised', path=tmp_path/'full.pt.gz', **kw)
    api.fit(*args, arm='supervised', path=tmp_path/'resumed.pt.gz', stop_at=2, **kw)
    resumed = api.fit(*args, arm='supervised', path=tmp_path/'resumed.pt.gz', resume=True, **kw)
    for k in full:
        if k != 'seconds':
            api.sampling.exact(full[k], resumed[k])
    marginal = api.fit(*args, arm='marginal', path=tmp_path/'marginal.pt.gz', **kw)
    api.assert_matched(marginal, full)
    assert full['unknown_rows_sampled'] == 0
    assert full['step'] == 6 and full['query_draws'] == 24
    p = api.predict(full, args[0], args[1], args[6])
    assert np.isfinite(p).all() and p.shape == (32, 5)
    with pytest.raises(ValueError):
        api.predict(full, args[0]*np.nan, args[1], args[6])


def test_no_easy_training_fails(tmp_path):
    args = list(example()); args[2][np.isfinite(args[2]).all(1), 0] = 0
    with pytest.raises(ValueError, match='No fitting easy'):
        api.fit(*args, arm='supervised', seed=17, settings={}, identity={}, path=tmp_path/'a', heartbeat=lambda **kw: None)


def test_common_anchor_feasible_and_counts_match():
    eligible = np.ones(4, bool)
    risks = dict(raw=np.array([[-1., -1.], [-1., -1.], [-1., 2.], [-1., 2.]]),
                 marginal=np.array([[-1., -1.], [-1., 1.], [-1., -1.], [-1., 2.]]),
                 supervised=np.array([[-1., -1.], [-1., 2.], [-1., 2.], [-1., -1.]]))
    out, _ = api.decisions(np.arange(4.)+1, risks, eligible, np.repeat('r', 4), np.ones(4), np.arange(4))
    np.testing.assert_array_equal(out['common_anchor'], [True, False, False, False])
    for arm in risks:
        assert out[arm+'_matched'].sum() == 1
        assert (risks[arm][out[arm+'_matched']].sum(0) <= 1e-10).all()


def test_quality_known_rows_only():
    y = np.array([[1., 2., .1], [0., 2., .2], [np.nan]*3])
    p = np.array([[.8, 2., .1, .04], [.2, 2., .2, .03], [0., 0., 0., 0.]])
    q = api.quality(p, y, ['r']*3, [0, 0, 1])
    assert q['Brier'] == pytest.approx(.04)
    assert q['conditional_reference_MSE'] == 0
    assert q['evaluable_queries'] == 1


def test_registration_identity_serializes_without_tuple_drift():
    import json
    from scripts.run_m3w_easy_hurdle import CONTRASTS, POLICIES
    v = dict(contrasts=CONTRASTS, policies=POLICIES)
    assert json.loads(json.dumps(v)) == v
