import numpy as np
import torch

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model import m3w_av2_relative_projection as model


def test_relative_loss_difference_replays_all_edges_and_missing_debits():
    rng = np.random.default_rng(9)
    cv = np.zeros((4, 12, 2))
    raw = cv.copy()
    ix = np.array([0, 2])
    raw[ix] = rng.normal(size=(2, 12, 2))
    y = rng.normal(size=cv.shape)
    v = rng.random(cv.shape[:-1]) > .25
    v[1] = False
    labels = model.labels(dict(cv=cv), raw, ix, y, v)
    args = tuple(torch.from_numpy(np.asarray(z)[None]) for z in (cv[ix], raw[ix], np.ones(2), *labels))
    a = np.repeat(np.array([[.2], [-.1]]), 12, axis=1)
    after = raw.copy()
    after[ix] = model.rotate(raw[ix], a, np)
    x = model.losses(torch.zeros_like(torch.from_numpy(a[None])), *args)[1]
    z = model.losses(torch.from_numpy(a[None]), *args)[1]
    np.testing.assert_allclose(float(x-z), pair_reduction(after[None], raw, y, v)[0, 1], atol=1e-12)


def test_refinement_starts_as_exact_reference_and_uses_bounded_angles():
    m = model.Refine(3)
    x = torch.randn(5, 2, 3)
    assert torch.equal(m(x), torch.zeros(5, 2, 12))
    with torch.no_grad():
        m.head[-1].bias[:] = torch.tensor([100., -100., 100.])
    assert m(x).abs().max().item() <= model.MAX_ANGLE+1e-6


def test_replay_confidence_is_neutral_for_unchanged_path():
    h = np.arange(8, dtype=float)[None, :, None]*np.array([[1., .3], [.1, .7]])[:, None]
    cv, _, _, _ = model.history.motion.rollout(h, 12)
    assert model.confidence(dict(h=h, cv=cv), cv, cv) == 1.
