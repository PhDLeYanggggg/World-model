import numpy as np
import pytest
import torch

from src.world_model import m3w_cross_moment_rank as new
from src.world_model import m3w_supported_rank_pairs as old
from src.world_model.m3w_hurdle_risk import predict
from tests.test_m3w_supported_rank_pairs import fixture


def test_cross_moment_weight_prefers_conditional_order_in_counterexample():
    y = torch.tensor([[1., 1.], [1., .03], [100., 0.], [1., .03]], dtype=torch.float64)
    true = torch.tensor([[50.5, .5], [1., .03], [50.5, .5], [1., .03]], dtype=torch.float64)
    biased = true.clone(); biased[[1, 3], 1] = .005
    sites = np.array(['a'] * 4)
    before, _ = old.ranking_loss(true, y, sites, epsilon=1e-6)
    wrong, _ = old.ranking_loss(biased, y, sites, epsilon=1e-6)
    after, _ = new.ranking_loss(true, y, sites, epsilon=1e-6)
    worse, _ = new.ranking_loss(biased, y, sites, epsilon=1e-6)
    assert wrong < before and after < worse


def test_pairing_preserves_locality_and_undefined_rows_do_not_rank():
    y = torch.tensor([[1., 0.], [0., 0.], [1., .5], [0., 1.]])
    sites = np.array(['a', 'a', 'a', 'b'])
    i, j, d = new.cross_pairs(y, sites)
    np.testing.assert_array_equal(sites[i], sites[j])
    assert set(i) == {0, 2} and len(d) == 2


def test_gradient_direction_and_fixed_normalizer():
    y = torch.tensor([[1., 0.], [1., .5]])
    p = torch.tensor([[1., .7], [1., .1]], requires_grad=True)
    a, info = new.ranking_loss(p, y, np.array(['a', 'a']), epsilon=1e-6)
    b, _ = new.ranking_loss(p, y, np.array(['a', 'a']), epsilon=1e-6, fixed_denominator=2.)
    torch.testing.assert_close(b, a/2)
    a.backward()
    assert p.grad[0, 1] > 0 and p.grad[1, 1] < 0
    assert info['rank_weight_sum'] == 1 and info['rank_effective_pairs'] == 2


def test_target_unit_scaling_and_matching_fixed_scale_cancel():
    y = torch.tensor([[2., .2], [1., .5], [3., 0.]], dtype=torch.float64)
    p = torch.ones_like(y); sites = np.array(['a'] * 3)
    for fixed in (None, 3.):
        a, _ = new.ranking_loss(p, y, sites, epsilon=1e-6, fixed_denominator=fixed)
        b, _ = new.ranking_loss(p, 100*y, sites, epsilon=1e-6,
                               fixed_denominator=None if fixed is None else 10000*fixed)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_large_float32_cost_products_do_not_overflow():
    y = torch.tensor([[1e30, 1e29], [1e30, 2e29]], dtype=torch.float32)
    p = torch.ones(2, 2, requires_grad=True)
    loss, info = new.ranking_loss(p, y, np.array(['a', 'a']), epsilon=1e-6)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert np.isfinite(info['rank_weight_sum'])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_invalid_training_targets_are_rejected(bad):
    y = torch.tensor([[1., .1], [bad, 0.]])
    with pytest.raises(ValueError):
        new.ranking_loss(torch.ones_like(y), y, np.array(['a', 'a']), epsilon=1e-6)


def test_zero_mass_and_equal_ratio_targets_give_zero_gradient():
    y = torch.tensor([[0., 0.], [1., .5], [2., 1.]])
    p = torch.ones_like(y, requires_grad=True)
    loss, info = new.ranking_loss(p, y, np.array(['a'] * 3), epsilon=1e-6)
    loss.backward()
    assert info['rank_pairs'] == 0
    torch.testing.assert_close(p.grad, torch.zeros_like(p))
    with pytest.raises(ValueError):
        new.ranking_loss(p, y, np.array(['a'] * 3), epsilon=1e-6, fixed_denominator=0.)


def test_zero_auxiliary_fit_preserves_control_exactly(tmp_path):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture()
    kw = dict(seed=17, settings=settings, identity={'test': 'same'}, heartbeat=lambda **_: None,
              rank_weight=0., epsilon=1e-6)
    a, _ = old.fit(x, y, sites, d, pr, directory=tmp_path/'old', **kw)
    b, r = new.fit(x, y, sites, d, pr, directory=tmp_path/'new', fixed_denominator=1., **kw)
    np.testing.assert_array_equal(predict(a, x, d, pr), predict(b, x, d, pr))
    assert r['unknown_rows_sampled'] == 0


@pytest.mark.parametrize('denominator', [None, 1.2])
def test_both_objectives_resume_exactly_and_bind_scale(tmp_path, denominator):
    torch.set_num_threads(1)
    x, y, sites, d, pr, settings = fixture()
    kw = dict(seed=29, settings=settings, identity={'test': 'resume'}, heartbeat=lambda **_: None,
              rank_weight=1., epsilon=1e-6, fixed_denominator=denominator)
    a, full = new.fit(x, y, sites, d, pr, directory=tmp_path/'full', **kw)
    new.fit(x, y, sites, d, pr, directory=tmp_path/'part', stop_at=4, **kw)
    b, r = new.fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True, **kw)
    np.testing.assert_array_equal(predict(a, x, d, pr), predict(b, x, d, pr))
    assert r['fixed_trace'] == full['fixed_trace']
    with pytest.raises(ValueError):
        new.fit(x, y, sites, d, pr, directory=tmp_path/'part', resume=True,
                **dict(kw, fixed_denominator=3.))
