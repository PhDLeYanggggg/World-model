import numpy as np
import torch  # Preload before the transitive experiment imports.

from src.world_model import m3w_sustained_turn as model
from scripts.audit_m3w_one_edit_radius import reachable, recording, witness


def test_sustained_turn_is_exact_for_a_constant_stepwise_turn():
    phase = .12*np.arange(19)
    chord = np.stack((np.cos(phase), np.sin(phase)), -1)
    points = np.concatenate((np.zeros((1, 2)), chord.cumsum(0)))
    p = model.paths(points[None, :8], 12)[0]
    np.testing.assert_allclose(p[2], points[8:]-points[7], atol=1e-12)
    np.testing.assert_array_equal(p[1, 0], p[2, 0])
    assert not np.allclose(p[1, -1], p[2, -1])


def test_rollouts_translate_rotate_and_degenerate_cleanly():
    h = np.random.default_rng(41).normal(size=(16, 8, 2)).cumsum(1)
    result = model.paths(h, 12)
    np.testing.assert_allclose(model.paths(model.rotate(h, .8)+[1., 2.], 12), model.rotate(result, .8))
    np.testing.assert_array_equal(model.paths(np.ones_like(h), 12), np.zeros_like(result))


def test_prefix_selector_keeps_exact_original_roster_radii_and_fallback():
    h = np.random.default_rng(42).normal(size=(32, 8, 2)).cumsum(1)
    amp = np.linspace(.25, 1., len(h))
    cv, _, bank, extra = model.matched(h, 12, amp)
    anchor, p, x, selected = model.choose(h, amp)
    ix = model.legacy.prefix_rank(h, amp, np.ones(len(h), bool)).mean(0).argmax(1)
    np.testing.assert_array_equal(anchor, bank[np.arange(len(h)), ix])
    np.testing.assert_array_equal(extra, x)
    np.testing.assert_array_equal(p, np.where(selected[..., None, None], x, anchor[:, None]))
    r = model.legacy.peak(anchor-cv)
    np.testing.assert_allclose(model.legacy.peak(p-cv[:, None]), np.broadcast_to(r[:, None], selected.shape))


def test_exact_ceiling_accounts_for_missing_step_tradeoff_and_easy_support():
    cv = np.array([[[1., 0.], [0., 0.]]])
    ols = np.array([[[.8, 0.], [.5, 0.]]])
    y = np.array([[[0., 0.], [np.nan, np.nan]]])
    valid = np.array([[True, False]])
    ok, c, o, m, obs = reachable(cv, ols, y, valid, np.array([.6]), 2.)
    assert ok[0] and o[0] < 0.
    p = witness(cv[0], ols[0], y[0], valid[0], .6, c[0], o[0], m[0], obs[0])
    assert p[1, 0] > 0.
    assert recording(cv, ols, y, valid, np.array([.6]), [2., 3.])
    assert not recording(cv, ols, y, valid, np.array([.2]), [2., 3.])
    assert not recording(cv, ols, y, valid, np.array([.6]), [.1, 3.])
