import numpy as np
import pytest
import torch

from src.world_model import m3w_motion_choice_cost as model


def example():
    rng = np.random.default_rng(100)
    h = rng.normal(size=(36, 8, 2)).cumsum(1)
    h[-1] = 3
    return h, model.candidates(h)


def test_same_rows_and_exact_peak_for_different_actions():
    h, (ols, _, bank, legal, radius) = example()
    score = np.broadcast_to(np.arange(legal.shape[1]), legal.shape)
    picks = [model.choose(score, legal), model.choose(-score, legal)]
    assert (picks[0] != picks[1]).any() and all(p[-1] == -1 for p in picks)
    assert np.array_equal(picks[0] >= 0, picks[1] >= 0)
    for p in picks:
        path = model.apply(ols, bank, legal, radius, p)
        np.testing.assert_allclose(np.linalg.norm(path - ols, axis=-1).max(1), np.where(p >= 0, radius, 0.))
    with pytest.raises(ValueError, match='fixed recipients'):
        model.apply(ols, bank, legal, radius, np.full(len(h), -1))


def test_causal_inputs_are_translation_invariant():
    h, (ols, _, b, _, r) = example()
    moved = h + [100., -20.]
    other, _, other_b, _, other_r = model.candidates(moved)
    np.testing.assert_allclose(model.inputs(h, ols, b, r), model.inputs(moved, other, other_b, other_r), atol=1e-6)


def test_future_mask_and_bounded_costs():
    _, (ols, _, b, legal, r) = example()
    valid = np.ones(ols.shape[:2], bool)
    valid[:, 4:] = False
    safe = ols.copy()
    missing = safe.copy()
    missing[~valid] = np.nan
    x = model.targets(ols, b, r, safe, valid)
    np.testing.assert_array_equal(x, model.targets(ols, b, r, missing, valid))
    assert np.all(x[:, :, 4:] == 0.) and np.abs(x[legal]).max() <= 1.


def test_all_objectives_backpropagate_and_use_the_same_head():
    h, (ols, _, b, legal, r) = example()
    use = legal.sum(1) >= 2
    x = torch.from_numpy(model.inputs(h, ols, b, r)[use])
    mask = np.ones(ols.shape[:2], bool)
    mask[::2, 4:8] = False
    y = model.targets(ols, b, r, ols, mask)
    data = [torch.from_numpy(v[use]) for v in (legal, y, mask)]
    for arm in model.ARMS:
        net = model.Head(x.shape[1])
        loss = model.loss(net, arm, x, *data)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters())
        assert model.scores(net(x), arm).shape == data[0].shape
