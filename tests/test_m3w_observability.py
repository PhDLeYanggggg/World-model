import numpy as np
import torch

from src.world_model.m3w_observability import box_features, components, estimate, loss


def test_box_quality_uses_exactly_eight_frames_and_partial_peers():
    box = np.zeros((3, 8, 4))
    box[..., 2:] = [10., 12.]
    box[..., 1] = 100.
    box[..., 0] = np.arange(8) + np.array([0., 5., 500.])[:, None]
    seen = np.ones((3, 8), bool)
    seen[1, :4] = False
    out = box_features(box, seen, np.array([0]), (640, 480))
    assert out.shape == (1, 12)
    assert out[0, -1] == .5
    assert out[0, 9] > .3
    assert out[0, 7] > out[0, 6]


def test_future_box_changes_cannot_change_prefix_features():
    from scripts.run_m3w_observability import feature_at

    box = np.ones((2, 16, 4))
    seen = np.ones((2, 16), bool)
    _, original = feature_at(box, seen, 7, (100, 100))
    box[:, 8:] = 1000.
    seen[:, 8:] = False
    _, changed = feature_at(box, seen, 7, (100, 100))
    np.testing.assert_array_equal(original, changed)


def test_factorization_uses_debit_when_step_is_missing():
    out = torch.zeros((1, 25), requires_grad=True)
    d = torch.ones((1, 12))
    score, p = estimate(out, d, "factor_quality")
    torch.testing.assert_close(score, torch.tensor([-.5]))
    torch.testing.assert_close(p, .5*d)
    value = loss(out, d, d, torch.zeros_like(d, dtype=torch.bool), torch.zeros(1), "factor_quality")
    value.backward()
    assert torch.isfinite(out.grad).all()
    assert (out.grad[:, 13:] == 0).all()


def test_components_match_full_grid_gain_with_masked_nan():
    floor = np.zeros((1, 12, 2))
    delta = np.ones_like(floor)
    y = delta.copy()
    observed = np.ones((1, 12), bool)
    observed[:, :3] = False
    y[~observed] = np.nan
    debit, fraction, direct = components(floor, delta, y, observed, np.ones(1))
    np.testing.assert_allclose(direct, .5*debit[:, 0])
    np.testing.assert_allclose(fraction[:, 3:], 1.)
