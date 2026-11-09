import numpy as np
import torch

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model import m3w_av2_forecast_cost as cost


def test_cost_uses_full_grid_and_keeps_missing_mask_out_of_describe():
    rng = np.random.default_rng(5)
    h = rng.normal(size=(3, 8, 2)).cumsum(1)
    cv = rng.normal(size=(3, 12, 2))
    a, p, y = cv+.2, np.stack((cv+.3, cv-.1, cv+.2)), cv+.1
    v = rng.random(size=y.shape[:2]) > .3
    features, bounds, scale = cost.describe(h, cv, cv, p, a, .25)
    t = cost.targets(p, a, cv, np.where(v[..., None], y, np.nan), v, .25, scale)
    np.testing.assert_allclose(t, cost.targets(p, a, cv, y, v, .25, scale), atol=1e-7)
    np.testing.assert_allclose(t[:, 0]-t[:, 1],
        np.where(v, np.linalg.norm(a-y, axis=-1)-np.linalg.norm(p-y, axis=-1),
                 -np.linalg.norm(p-a, axis=-1)).mean((1, 2))/scale, atol=1e-7)
    np.testing.assert_allclose(t[:, 2]-t[:, 3], pair_reduction(p, a, y, v)[:, 1]/scale, atol=1e-7)
    assert (np.abs(t) <= bounds+1e-6).all()
    np.testing.assert_array_equal(t[-1, :4], 0.)
    assert features.shape == (3, cost.relative.WIDTH+6)


def test_signed_easy_term_preserves_original_cap_and_is_not_positive_part():
    cv = np.ones((2, 12, 2))
    cv[..., 1] = 0.
    y = cv.copy()
    y[0, :, 0] -= .1
    y[1, :, 0] -= .8
    p = cv[None].copy()
    p[0, 0, :, 0] -= .05
    t = cost.targets(p, cv, cv, y, np.ones((2, 12), dtype=bool), .2, 1.)
    np.testing.assert_allclose(t[0, -1], (-.05-.02*.1)/2)


def test_cost_scoring_ablation_debits_and_easy_harm_without_changing_prior():
    prior = torch.ones(3)
    r = torch.tensor([[.3, .1, .4, .1, .3], [.3, .5, .4, .6, -.3], [.3, .1, .4, .1, -.3]])
    bounds = torch.ones_like(r)
    s = cost.scores(prior, r, bounds)
    torch.testing.assert_close(s["history"], prior)
    assert s["known"].unique().numel() == 1
    assert s["debit"][0] > s["debit"][1]
    assert s["debit_easy"][2] > s["debit_easy"][0]
    bounds[:, :4] = 0
    z = cost.scores(prior, r, bounds)
    torch.testing.assert_close(z["debit"], torch.full((3,), .25))
