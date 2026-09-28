import numpy as np
import pytest
import torch
from src.world_model import m3w_easy_risk_priority as repair
from tests.test_m3w_easy_hurdle import example


def test_cap_preserves_half_risk_projection_under_conflict():
    r = (torch.tensor([1., 0.]),)
    a = (torch.tensor([-100., 0.]),)
    g, stats = repair.capped_gradient(r, a)
    torch.testing.assert_close(g[0], torch.tensor([.5, 0.]))
    assert stats['alpha'] == .005 and stats['risk_projection'] == .5
    assert g[0].grad_fn is None


def test_cap_zero_and_small_gradients():
    z = (torch.zeros(3),)
    g, s = repair.capped_gradient(z, (torch.ones(3),))
    assert torch.equal(g[0], z[0]) and s['alpha'] == 0 and s['risk_projection'] is None
    g, s = repair.capped_gradient((torch.ones(3),), z)
    assert torch.equal(g[0], torch.ones(3))
    g, s = repair.capped_gradient((torch.ones(3),), (torch.full((3,), .1),))
    assert s['alpha'] == 1 and not s['capped']
    with pytest.raises(FloatingPointError):
        repair.capped_gradient(z, (torch.full((3,), float('nan')),))


def test_random_projection_bound_and_norm_cap():
    rng = torch.Generator().manual_seed(37)
    for _ in range(100):
        r = (torch.randn(30, generator=rng), torch.randn(2, 4, generator=rng))
        a = tuple(100*torch.randn(v.shape, generator=rng) for v in r)
        g, s = repair.capped_gradient(r, a)
        assert s['risk_projection'] >= .5-1e-6
        aux_norm = torch.cat([(gv-rv).flatten() for gv, rv in zip(g, r)]).norm()
        assert float(aux_norm) <= .5*s['risk_gradient_norm']+1e-5


def test_actual_fit_matched_sampler_exact_resume_and_control(tmp_path):
    torch.set_num_threads(1)
    settings = dict(width=8, steps=6, query_batch_size=4, learning_rate=.001,
                    gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    args = example(); kw = dict(seed=17, settings=settings, identity={'fit': 'synthetic'}, heartbeat=lambda **kw: None)
    full = repair.fit(*args, arm='risk_priority', path=tmp_path/'full.gz', **kw)
    repair.fit(*args, arm='risk_priority', path=tmp_path/'resume.gz', stop_at=2, **kw)
    resumed = repair.fit(*args, arm='risk_priority', path=tmp_path/'resume.gz', resume=True, **kw)
    for k in full:
        if k != 'seconds':
            repair.api.sampling.exact(full[k], resumed[k])
    control = repair.fit(*args, arm='uncapped', path=tmp_path/'control.gz', **kw)
    original = repair.api.fit(*args, arm='supervised', path=tmp_path/'original.gz', **kw)
    repair.assert_matched(control, full)
    repair.assert_parent_control(original, control)
    assert full['unknown_rows_sampled'] == 0 and full['min_risk_projection'] >= .5-1e-6
    assert np.isfinite(repair.api.predict(full, args[0], args[1], args[6])).all()


def test_no_fitting_easy_does_not_train(tmp_path):
    args = list(example()); args[2][np.isfinite(args[2]).all(1), 0] = 0
    with pytest.raises(ValueError, match='No fitting easy'):
        repair.fit(*args, arm='risk_priority', seed=1, settings={}, identity={},
                   path=tmp_path/'no.gz', heartbeat=lambda **kw: None)


def test_common_count_is_preserved():
    eligible = np.ones(4, bool)
    risk = np.array([[-1., -1.], [-1., -1.], [-1., 2.], [-1., 3.]])
    risks = {arm: risk.copy() for arm in ('raw', *repair.ARMS)}
    risks['risk_priority'][1, 1] = 1
    out, _ = repair.decisions(np.arange(4.)+1, risks, eligible, np.repeat('r', 4), np.ones(4), np.arange(4))
    for arm in risks:
        assert out[arm+'_matched'].sum() == 1
        assert (risks[arm][out[arm+'_matched']].sum(0) <= 1e-10).all()
